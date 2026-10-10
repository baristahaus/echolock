// file_page: GPU reads from a memory-mapped file, with its pages resident in and evicted from the
// page cache. Spec: probes/file_page/spec.md.
//
// Ported from ninfer-gb10 tools/gb10/file_page_probe.cu. Changes from the original: JSONL
// records; the file holds a counter-based pattern so the host knows every row's XOR without
// touching the file, and every row the GPU reads is checked; mincore() records the residency
// each phase actually started from; the evict phase defaults to posix_fadvise(DONTNEED) on the
// scratch file (the original read a >RAM evictor file, still available with --evict PATH).
//
// Phases run as separate invocations (--phase) so a hung cold-fault phase can be killed by a
// timeout without taking the rest (run/plans.py sequences them):
//   prep           create + fill + fsync the scratch file
//   warm           host-read the whole mapping (populate the page cache)
//   res-seq        GPU full sequential read (pages resident)
//   res-rows       GPU random rows (resident)
//   evict          drop the scratch file from the page cache
//   cold-serial    1 GPU thread, sequential cold rows (serialized per-fault latency)
//   cold-rows      GPU random cold rows (parallel fault throughput)
//   res-rows-again same rows as res-rows, now resident again (must match res-rows)

#include "common/echolock.cuh"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <functional>

using namespace echolock;

namespace {

constexpr std::uint32_t kRowBytes  = 2560; // one PLE row in ninfer: 16 heads x 160 B
constexpr std::uint32_t kRowWords  = kRowBytes / sizeof(std::uint32_t);
constexpr std::uint32_t kPageBytes = 4096;
constexpr std::uint32_t kPageWords = kPageBytes / sizeof(std::uint32_t);
constexpr std::uint64_t kSeed      = 0x9E3779B97F4A7C15ull;
constexpr std::uint32_t kWordSeed  = 0x45594C4Bu;

struct Options {
    std::string phase, file, evict;
    std::size_t bytes       = std::size_t{512} << 20;
    std::size_t evict_bytes = std::size_t{130} << 30;
    int rows                = 100000;
    int serial_rows         = 512;
};

std::size_t parse_bytes(std::string s) {
    std::size_t mult = 1;
    if (!s.empty()) {
        const char u = s.back();
        if (u == 'G' || u == 'g') { mult = std::size_t{1} << 30; s.pop_back(); }
        else if (u == 'M' || u == 'm') { mult = std::size_t{1} << 20; s.pop_back(); }
        else if (u == 'K' || u == 'k') { mult = std::size_t{1} << 10; s.pop_back(); }
    }
    return std::stoull(s) * mult;
}

Options parse(int argc, char** argv) {
    Args args(argc, argv,
              "usage: file_page --phase prep|warm|res-seq|res-rows|evict|cold-serial|cold-rows|"
              "res-rows-again --file PATH [--bytes 512M] [--evict PATH] [--evict-bytes 130G] "
              "[--rows 100000] [--serial-rows 512]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--phase") { o.phase = args.value(a); }
        else if (a == "--file") { o.file = args.value(a); }
        else if (a == "--evict") { o.evict = args.value(a); }
        else if (a == "--bytes") { o.bytes = parse_bytes(args.value(a)); }
        else if (a == "--evict-bytes") { o.evict_bytes = parse_bytes(args.value(a)); }
        else if (a == "--rows") { o.rows = std::stoi(args.value(a)); }
        else if (a == "--serial-rows") { o.serial_rows = std::stoi(args.value(a)); }
        else { args.usage_error("unknown argument " + a); }
    }
    if (o.file.empty()) { args.usage_error("--file is required"); }
    if (o.phase.empty()) { args.usage_error("--phase is required"); }
    o.bytes = o.bytes / kPageBytes * kPageBytes;
    if (o.bytes == 0 || o.rows <= 0 || o.serial_rows <= 0) { args.usage_error("sizes must be > 0"); }
    return o;
}

// u32 word w of the scratch file (murmur3 fmix32). The host oracle and the prep phase share it.
__host__ __device__ __forceinline__ std::uint32_t pattern_word(std::uint64_t w) {
    std::uint32_t h = static_cast<std::uint32_t>(w) ^ static_cast<std::uint32_t>(w >> 32) ^ kWordSeed;
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

// Row i starts at a 4 KiB-aligned page chosen by splitmix64: identical across phases and runs.
std::uint64_t row_page(std::uint64_t i, std::uint64_t pages) {
    std::uint64_t z = kSeed ^ (i * 0x9E3779B97F4A7C15ull);
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (z ^ (z >> 31)) % pages;
}

std::uint32_t expected_row_xor(std::uint32_t page) {
    std::uint32_t x = 0;
    const std::uint64_t first = std::uint64_t(page) * kPageWords;
    for (std::uint32_t w = 0; w < kRowWords; ++w) { x ^= pattern_word(first + w); }
    return x;
}

// ---- kernels ------------------------------------------------------------------------------------

// One block per row; the row's 640 words are read coalesced and their XOR stored to out[row].
__global__ void row_read_kernel(const std::uint32_t* __restrict__ base,
                                const std::uint32_t* __restrict__ pages, int n,
                                std::uint32_t* out) {
    const int row = blockIdx.x;
    if (row >= n) { return; }
    const std::uint32_t* p = base + std::size_t(pages[row]) * kPageWords;
    std::uint32_t acc      = 0;
    for (std::uint32_t w = threadIdx.x; w < kRowWords; w += blockDim.x) { acc ^= __ldcs(p + w); }
    for (int s = 16; s > 0; s >>= 1) { acc ^= __shfl_xor_sync(0xffffffffu, acc, s); }
    if ((threadIdx.x & 31) == 0) { atomicXor(out + row, acc); }
}

// Single thread, rows in order: serialized per-fault latency.
__global__ void serial_row_kernel(const std::uint32_t* __restrict__ base,
                                  const std::uint32_t* __restrict__ pages, int n,
                                  std::uint32_t* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) { return; }
    for (int row = 0; row < n; ++row) {
        const std::uint32_t* p = base + std::size_t(pages[row]) * kPageWords;
        std::uint32_t acc      = 0;
        for (std::uint32_t w = 0; w < kRowWords; ++w) { acc ^= p[w]; }
        out[row] = acc;
    }
}

__global__ void seq_read_kernel(const std::uint32_t* __restrict__ data, std::size_t words,
                                std::uint32_t* out) {
    std::uint32_t acc = 0;
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < words;
         i += std::size_t(gridDim.x) * blockDim.x) {
        acc ^= __ldcs(data + i);
    }
    for (int s = 16; s > 0; s >>= 1) { acc ^= __shfl_xor_sync(0xffffffffu, acc, s); }
    if ((threadIdx.x & 31) == 0) { atomicXor(out, acc); }
}

// GPU (event) and host wall time of one launch. Resident phases warm up first; cold phases must
// fault inside the timed launch, so they don't.
struct Timing {
    double gpu_s = 0, wall_s = 0;
};
// `reset` runs after the warm launch, so only the timed launch's result is checked.
Timing timed_kernel(cudaStream_t stream, const std::function<void()>& launch, bool warm,
                    const std::function<void()>& reset) {
    cudaEvent_t a, b;
    EL_CUDA(cudaEventCreate(&a));
    EL_CUDA(cudaEventCreate(&b));
    if (warm) {
        launch();
        EL_CUDA(cudaGetLastError());
    }
    reset();
    const std::int64_t h0 = mono_ns();
    EL_CUDA(cudaEventRecord(a, stream));
    launch();
    EL_CUDA(cudaGetLastError());
    EL_CUDA(cudaEventRecord(b, stream));
    EL_CUDA(cudaEventSynchronize(b));
    const std::int64_t h1 = mono_ns();
    float ms              = 0;
    EL_CUDA(cudaEventElapsedTime(&ms, a, b));
    EL_CUDA(cudaEventDestroy(a));
    EL_CUDA(cudaEventDestroy(b));
    return {ms / 1e3, double(h1 - h0) / 1e9};
}

// ---- host side ----------------------------------------------------------------------------------

void* map_file(const std::string& path, std::size_t bytes) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fail("open " + path + ": " + std::strerror(errno)); }
    void* p = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { fail("mmap " + path + ": " + std::strerror(errno)); }
    return p;
}

// Fraction of the mapping's pages resident in the page cache, per mincore().
double resident_fraction(void* base, std::size_t bytes) {
    const long page = sysconf(_SC_PAGESIZE);
    std::vector<unsigned char> vec((bytes + page - 1) / page);
    if (mincore(base, bytes, vec.data()) != 0) { return -1; }
    std::size_t n = 0;
    for (const unsigned char v : vec) { n += v & 1; }
    return double(n) / double(vec.size());
}

void emit_residency(const char* when, double fraction) {
    measure("page_cache_resident", fraction, "fraction", Obj().str("when", when), fraction >= 0);
}

void phase_prep(const Options& o) {
    // Create the scratch file's directory (mkdir -p semantics); --scratch-dir may point somewhere
    // that does not exist yet.
    const std::size_t slash = o.file.rfind('/');
    if (slash != std::string::npos && slash > 0) {
        std::string dir = o.file.substr(0, slash);
        for (std::size_t i = 1; i <= dir.size(); ++i) {
            if (i == dir.size() || dir[i] == '/') {
                const std::string part = dir.substr(0, i);
                if (mkdir(part.c_str(), 0755) != 0 && errno != EEXIST) {
                    fail("mkdir " + part + ": " + std::strerror(errno));
                }
            }
        }
    }
    // A tmpfs scratch would measure RAM, not storage; refuse rather than post a number that looks
    // like disk behaviour. (The file itself may not exist yet, so statfs its parent directory.)
    const std::string stat_dir = slash != std::string::npos && slash > 0 ? o.file.substr(0, slash) : std::string(".");
    struct statfs st {};
    if (statfs(stat_dir.c_str(), &st) == 0 && st.f_type == 0x01021994 /* TMPFS_MAGIC */) {
        fail("scratch path " + o.file + " is on tmpfs; page-cache numbers would measure RAM, "
             "not storage -- pass --scratch-dir on a real filesystem");
    }
    const int fd = open(o.file.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) { fail("open scratch: " + std::string(std::strerror(errno))); }
    std::vector<std::uint32_t> buf((1 << 20) / sizeof(std::uint32_t));
    const std::int64_t t0 = mono_ns();
    for (std::size_t off = 0; off < o.bytes; off += buf.size() * 4) {
        const std::uint64_t w0 = off / 4;
        const std::size_t words = std::min(buf.size(), (o.bytes - off) / 4);
        for (std::size_t i = 0; i < words; ++i) { buf[i] = pattern_word(w0 + i); }
        if (write(fd, buf.data(), words * 4) != static_cast<ssize_t>(words * 4)) {
            fail("write scratch: " + std::string(std::strerror(errno)));
        }
    }
    if (fsync(fd) != 0) { fail("fsync scratch: " + std::string(std::strerror(errno))); }
    close(fd);
    measure("prep_write", double(o.bytes) / (double(mono_ns() - t0) / 1e9) / 1e9, "GB/s",
            Obj().uint("bytes", o.bytes), true);
}

void phase_warm(const Options& o) {
    auto* p = static_cast<const std::uint32_t*>(map_file(o.file, o.bytes));
    emit_residency("before", resident_fraction(const_cast<std::uint32_t*>(p), o.bytes));
    const std::int64_t t0 = mono_ns();
    std::uint32_t acc     = 0;
    for (std::size_t w = 0; w < o.bytes / 4; ++w) { acc ^= p[w]; }
    const double s = double(mono_ns() - t0) / 1e9;
    std::uint32_t want = 0;
    for (std::size_t w = 0; w < o.bytes / 4; ++w) { want ^= pattern_word(w); }
    check("warm.host_xor", acc == want, Obj().uint("got", acc).uint("expected", want));
    measure("host_warm_read", double(o.bytes) / s / 1e9, "GB/s", Obj(), acc == want);
    emit_residency("after", resident_fraction(const_cast<std::uint32_t*>(p), o.bytes));
    munmap(const_cast<std::uint32_t*>(p), o.bytes);
}

void phase_evict(const Options& o) {
    void* base = map_file(o.file, o.bytes);
    emit_residency("before", resident_fraction(base, o.bytes));
    const std::int64_t t0 = mono_ns();
    if (o.evict.empty()) {
        const int fd = open(o.file.c_str(), O_RDONLY);
        if (fd < 0) { fail("open scratch: " + std::string(std::strerror(errno))); }
        const int rc = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
        if (rc != 0) { fail("posix_fadvise(DONTNEED): " + std::string(std::strerror(rc))); }
        note("evicted with posix_fadvise(DONTNEED)");
    } else {
        // The original method: stream a file larger than RAM through the page cache.
        const int fd = open(o.evict.c_str(), O_RDONLY);
        if (fd < 0) { fail("open evictor: " + std::string(std::strerror(errno))); }
        std::vector<char> buf(1 << 18);
        std::uint64_t total = 0;
        while (total < o.evict_bytes) {
            const ssize_t r = read(fd, buf.data(), buf.size());
            if (r <= 0) {
                if (lseek(fd, 0, SEEK_SET) < 0) { break; }
                continue;
            }
            total += std::uint64_t(r);
        }
        close(fd);
        measure("evictor_read", double(total) / (double(mono_ns() - t0) / 1e9) / 1e9, "GB/s",
                Obj().str("evictor", o.evict).uint("bytes", total), true);
    }
    const double after = resident_fraction(base, o.bytes);
    emit_residency("after", after);
    measure("evict_duration", double(mono_ns() - t0) / 1e9, "s",
            Obj().str("method", o.evict.empty() ? "fadvise" : "evictor_file"), true);
    // Not a failed probe, but the cold phases are meaningless if the pages stayed.
    feature("page_cache_eviction", true, -1, after >= 0 && after < 0.01,
            Obj().num("resident_after", after));
    munmap(base, o.bytes);
}

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    probe_start("file_page", Obj()
                                 .str("phase", o.phase)
                                 .str("file", o.file)
                                 .uint("bytes", o.bytes)
                                 .integer("rows", o.rows)
                                 .integer("serial_rows", o.serial_rows)
                                 .str("evict", o.evict)
                                 .uint("row_bytes", kRowBytes));
    if (o.phase == "prep") { phase_prep(o); return probe_end(); }
    if (o.phase == "warm") { phase_warm(o); return probe_end(); }
    if (o.phase == "evict") { phase_evict(o); return probe_end(); }

    const bool cold = o.phase == "cold-serial" || o.phase == "cold-rows";
    if (!cold && o.phase != "res-seq" && o.phase != "res-rows" && o.phase != "res-rows-again") {
        fail("unknown phase " + o.phase);
    }
    // Map and measure residency before CUDA starts, so context creation can't disturb it.
    auto* base = static_cast<const std::uint32_t*>(map_file(o.file, o.bytes));
    const double resident = resident_fraction(const_cast<std::uint32_t*>(base), o.bytes);
    emit_residency("before", resident);
    if (cold && resident > 0.01) {
        feature("page_cache_eviction", true, -1, false, Obj().num("resident_before_cold", resident));
    }

    EL_CUDA(cudaSetDevice(0));
    declare_device(0);
    declared("cuda.pageableMemoryAccess",
             std::to_string(runtime_attr(cudaDevAttrPageableMemoryAccess, 0)), "cudaDeviceGetAttribute");
    const int sms = runtime_attr(cudaDevAttrMultiProcessorCount, 0);
    cudaStream_t stream;
    EL_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    declare_compiled_arch(stream);

    const std::uint64_t pages = o.bytes / kPageBytes;
    const int n               = o.phase == "cold-serial" ? o.serial_rows : o.rows;
    std::vector<std::uint32_t> row_pages(n);
    for (int i = 0; i < n; ++i) { row_pages[i] = static_cast<std::uint32_t>(row_page(i, pages)); }
    // Two rows can land on the same page, and only the first of them faults. `rows` overstates
    // the work a cold phase did, so every row measurement carries the honest counts alongside.
    std::vector<std::uint32_t> sorted_pages(row_pages);
    std::sort(sorted_pages.begin(), sorted_pages.end());
    const int unique_pages = static_cast<int>(std::unique(sorted_pages.begin(), sorted_pages.end())
                                             - sorted_pages.begin());
    const int duplicate_rows = n - unique_pages;
    std::uint32_t *d_pages = nullptr, *d_out = nullptr;
    EL_CUDA(cudaMalloc(&d_pages, row_pages.size() * sizeof(std::uint32_t)));
    EL_CUDA(cudaMalloc(&d_out, std::size_t(n) * sizeof(std::uint32_t)));
    EL_CUDA(cudaMemcpy(d_pages, row_pages.data(), row_pages.size() * sizeof(std::uint32_t),
                       cudaMemcpyHostToDevice));
    EL_CUDA(cudaMemset(d_out, 0, std::size_t(n) * sizeof(std::uint32_t)));

    const Obj tags = Obj().str("phase", o.phase).num("resident_before", resident)
                           .integer("rows", n).integer("unique_pages", unique_pages);
    const auto reset = [&] {
        EL_CUDA(cudaMemsetAsync(d_out, 0, std::size_t(n) * sizeof(std::uint32_t), stream));
    };
    if (o.phase == "res-seq") {
        const std::size_t words = o.bytes / 4;
        const Timing t = timed_kernel(stream, [&] {
            seq_read_kernel<<<sms * 16, 256, 0, stream>>>(base, words, d_out);
        }, true, reset);
        std::uint32_t got = 0;
        EL_CUDA(cudaMemcpy(&got, d_out, sizeof(got), cudaMemcpyDeviceToHost));
        std::uint32_t one_pass = 0;
        for (std::size_t w = 0; w < words; ++w) { one_pass ^= pattern_word(w); }
        const bool ok = got == one_pass;
        check("res-seq.xor", ok, Obj().uint("got", got).uint("expected", one_pass));
        measure("gpu_seq_read", double(o.bytes) / t.gpu_s / 1e9, "GB/s", tags, ok);
        measure("gpu_seq_read_wall", t.wall_s * 1e3, "ms", tags, ok);
    } else {
        const bool serial = o.phase == "cold-serial";
        const Timing t    = timed_kernel(stream, [&] {
            if (serial) {
                serial_row_kernel<<<1, 1, 0, stream>>>(base, d_pages, n, d_out);
            } else {
                row_read_kernel<<<n, 64, 0, stream>>>(base, d_pages, n, d_out);
            }
        }, !cold, reset);
        std::vector<std::uint32_t> got(n);
        EL_CUDA(cudaMemcpy(got.data(), d_out, got.size() * sizeof(std::uint32_t),
                           cudaMemcpyDeviceToHost));
        int bad = 0, first_bad = -1;
        for (int i = 0; i < n; ++i) {
            const std::uint32_t want = expected_row_xor(row_pages[i]);
            if (got[i] != want) {
                ++bad;
                if (first_bad < 0) { first_bad = i; }
            }
        }
        check(o.phase + ".row_xor", bad == 0,
              Obj().integer("rows", n).integer("unique_pages", unique_pages)
                  .integer("duplicate_rows", duplicate_rows).integer("bad_rows", bad)
                  .integer("first_bad_row", first_bad));
        const double bytes = double(n) * kRowBytes;
        measure(serial ? "gpu_serial_rows" : "gpu_rows", t.gpu_s * 1e6 / n, "us/row", tags, bad == 0);
        measure(serial ? "gpu_serial_rows_bw" : "gpu_rows_bw", bytes / t.gpu_s / 1e6, "MB/s", tags,
                bad == 0);
        measure(serial ? "gpu_serial_rows_wall" : "gpu_rows_wall", t.wall_s * 1e3, "ms", tags,
                bad == 0);
        if (cold) {
            // Faults per second, counted in pages: duplicate rows on a faulted page are not faults.
            measure("cold_fault_throughput", double(unique_pages) / t.gpu_s, "pages/s", tags,
                    bad == 0);
        }
    }
    emit_residency("after", resident_fraction(const_cast<std::uint32_t*>(base), o.bytes));

    EL_CUDA(cudaFree(d_pages));
    EL_CUDA(cudaFree(d_out));
    EL_CUDA(cudaStreamDestroy(stream));
    munmap(const_cast<std::uint32_t*>(base), o.bytes);
    return probe_end();
}
