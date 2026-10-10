// pointer_chase: memory latency as a dependent-load chain sees it -- one load per hop, every hop
// dependent on the previous -- through each kind of memory, at footprints that fit and overflow
// the cache, including cold page faults on a memory-mapped file. Spec: probes/pointer_chase/spec.md.
//
// The chain is data (an array of next-indices), so nothing can be prefetched or hoisted: the
// compiler cannot skip the walk, and the memory system cannot look ahead. The host walks the same
// chain to know the visited XOR, the final index and the pages touched, which is the oracle.

#include "common/echolock.cuh"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

using namespace echolock;

namespace {

constexpr std::uint64_t kSeed = 0x9E3779B97F4A7C15ull;
constexpr unsigned kPageBytes  = 4096;

struct Options {
    std::vector<double> footprints_mb = {0.0625, 1, 64}; // chain footprint, MiB
    unsigned hops          = 4096;
    int reps               = 5;
    std::string kinds      = "device,pinned,pageable,file";
    std::string patterns   = "seq,random";
    std::string file       = "pointer_chase.scratch";
    std::size_t file_bytes = std::size_t{512} << 20;
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

std::vector<std::string> split_list(const std::string& s) {
    std::vector<std::string> out;
    std::size_t b = 0;
    for (std::size_t e; (e = s.find(',', b)) != std::string::npos; b = e + 1) {
        out.push_back(s.substr(b, e - b));
    }
    out.push_back(s.substr(b));
    return out;
}

Options parse(int argc, char** argv) {
    Args args(argc, argv,
              "usage: pointer_chase [--footprints 0.0625,1,64 (MiB)] [--hops 4096] [--reps 5] "
              "[--kinds device,pinned,pageable,file] [--patterns seq,random] [--file PATH] "
              "[--bytes 512M]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--footprints") {
            o.footprints_mb.clear();
            for (const auto& v : split_list(args.value(a))) { o.footprints_mb.push_back(std::stod(v)); }
        } else if (a == "--hops") { o.hops = std::stoul(args.value(a)); }
        else if (a == "--reps") { o.reps = std::stoi(args.value(a)); }
        else if (a == "--kinds") { o.kinds = args.value(a); }
        else if (a == "--patterns") { o.patterns = args.value(a); }
        else if (a == "--file") { o.file = args.value(a); }
        else if (a == "--bytes") { o.file_bytes = parse_bytes(args.value(a)); }
        else { args.usage_error("unknown argument " + a); }
    }
    if (o.footprints_mb.empty() || o.hops < 16 || o.reps < 1) {
        args.usage_error("need footprints, hops >= 16, reps >= 1");
    }
    return o;
}

// ---- chains -------------------------------------------------------------------------------------

// seq: hop distance one 4 KiB page (wrapped) -- a cold walk then pays one fault per hop.
// random: one uniform cycle over all nodes (Fisher-Yates, then linked): every node reachable,
// no shortcuts, no repeats before the wrap.
std::vector<unsigned> build_chain(std::size_t n, const std::string& pattern) {
    std::vector<unsigned> chain(n);
    if (pattern == "seq") {
        const std::size_t stride =
            std::max<std::size_t>(1, std::min<std::size_t>(kPageBytes / 4, n / 2));
        for (std::size_t i = 0; i < n; ++i) { chain[i] = static_cast<unsigned>((i + stride) % n); }
        return chain;
    }
    std::vector<unsigned> perm(n);
    for (std::size_t i = 0; i < n; ++i) { perm[i] = static_cast<unsigned>(i); }
    std::uint64_t z = kSeed;
    for (std::size_t i = n - 1; i > 0; --i) {
        z += 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        const std::size_t j = static_cast<std::size_t>((z ^ (z >> 27)) % (i + 1));
        std::swap(perm[i], perm[j]);
    }
    for (std::size_t i = 0; i < n; ++i) { chain[perm[i]] = perm[(i + 1) % n]; }
    return chain;
}

// The host walks the same chain the GPU will: the expected visited XOR, the final index, and the
// pages the walk touches (so a cold walk's fault count is known, not estimated).
struct WalkCheck {
    unsigned acc        = 0;
    unsigned final_idx  = 0;
    std::size_t unique_pages = 0;
};

WalkCheck simulate_walk(const std::vector<unsigned>& chain, unsigned hops) {
    WalkCheck w;
    const std::size_t pages_per_elem = kPageBytes / 4;
    std::vector<bool> seen(chain.size() / pages_per_elem + 1, false);
    unsigned i = 0;
    for (unsigned h = 0; h < hops; ++h) {
        const unsigned next = chain[i];
        w.acc ^= next;
        const std::size_t page = next / pages_per_elem;
        if (page < seen.size() && !seen[page]) {
            seen[page] = true;
            ++w.unique_pages;
        }
        i = next;
    }
    w.final_idx = i;
    return w;
}

// ---- kernel -------------------------------------------------------------------------------------

// One thread, one block: every hop is a dependent load. GPU timestamps bracket the walk, and
// clock64() gives the same walk in SM clocks.
__global__ void chase_kernel(const unsigned* __restrict__ chain, unsigned hops,
                             unsigned long long* out /* t0, t1, c0, c1, idx, acc */) {
    if (threadIdx.x != 0 || blockIdx.x != 0) { return; }
    const unsigned long long t0 = global_ns();
    const long long c0          = clock64();
    unsigned i = 0, acc = 0;
    for (unsigned h = 0; h < hops; ++h) {
        i = chain[i]; // the dependent load: nothing can be prefetched past this
        acc ^= i;
    }
    const long long c1          = clock64();
    const unsigned long long t1 = global_ns();
    out[0] = t0;
    out[1] = t1;
    out[2] = static_cast<unsigned long long>(c0);
    out[3] = static_cast<unsigned long long>(c1);
    out[4] = i;
    out[5] = acc;
}

struct ChaseResult {
    double hop_ns = 0, hop_cycles = 0;
    unsigned acc = 0, final_idx = 0;
};

ChaseResult run_chase(cudaStream_t stream, const unsigned* base, unsigned hops) {
    unsigned long long* slot = nullptr;
    EL_CUDA(cudaMalloc(&slot, 6 * sizeof(unsigned long long)));
    chase_kernel<<<1, 1, 0, stream>>>(base, hops, slot);
    EL_CUDA(cudaGetLastError());
    EL_CUDA(cudaStreamSynchronize(stream));
    std::vector<unsigned long long> got(6);
    EL_CUDA(cudaMemcpy(got.data(), slot, 6 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    EL_CUDA(cudaFree(slot));
    ChaseResult r;
    r.hop_ns     = double(got[1] - got[0]) / hops;
    r.hop_cycles = double(got[3] - got[2]) / hops;
    r.acc        = static_cast<unsigned>(got[5]);
    r.final_idx  = static_cast<unsigned>(got[4]);
    return r;
}

// Warm-up walk, then `reps` timed walks: one measure per rep, a summary, and one check over the
// whole series. Returns the median hop latency and whether every walk matched the oracle.
struct SeriesOutcome {
    double median_ns = 0;
    bool verified    = false;
};

SeriesOutcome chase_series(cudaStream_t stream, const unsigned* base, unsigned hops, int reps,
                           const Obj& tags, const WalkCheck& oracle, const std::string& name) {
    run_chase(stream, base, hops); // untimed warm-up: brings the chain into the caches
    std::vector<double> per;
    int bad = 0;
    for (int r = 0; r < reps; ++r) {
        const ChaseResult res = run_chase(stream, base, hops);
        const bool pass = res.acc == oracle.acc && res.final_idx == oracle.final_idx;
        bad += pass ? 0 : 1;
        per.push_back(res.hop_ns);
        measure(name, res.hop_ns, "ns/hop", Obj(tags).num("hop_cycles", res.hop_cycles), pass, r);
    }
    const Summary s = summarize(per);
    const bool ok   = bad == 0;
    measure(name, s.p50, "ns/hop", Obj(tags).obj("per_rep", summary_obj(s)), ok);
    Obj detail;
    detail.obj("tags", tags).integer("reps", reps).integer("bad_walks", bad)
        .uint("expected_acc", oracle.acc).uint("expected_final", oracle.final_idx);
    check(name + ".walk_xor", ok, detail);
    return {s.p50, ok};
}

// ---- file backing --------------------------------------------------------------------------------

// The file only needs to exist once, at full size and non-sparse; the chain is written into its
// head afterwards. Rewriting 512 MiB per run would only slow every run down.
void prepare_file(const Options& o) {
    struct stat st {};
    if (stat(o.file.c_str(), &st) == 0 && static_cast<std::size_t>(st.st_size) >= o.file_bytes) {
        return;
    }
    const int fd = open(o.file.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) { fail("open scratch: " + std::string(std::strerror(errno))); }
    std::vector<unsigned char> buf(1 << 20, 0xAB);
    for (std::size_t off = 0; off < o.file_bytes; off += buf.size()) {
        const std::size_t n = std::min(buf.size(), o.file_bytes - off);
        if (write(fd, buf.data(), n) != static_cast<ssize_t>(n)) {
            fail("write scratch: " + std::string(std::strerror(errno)));
        }
    }
    if (fsync(fd) != 0) { fail("fsync scratch: " + std::string(std::strerror(errno))); }
    close(fd);
}

void* map_file(const std::string& path, std::size_t bytes) {
    const int fd = open(path.c_str(), O_RDWR);
    if (fd < 0) { fail("open " + path + ": " + std::string(std::strerror(errno))); }
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { fail("mmap " + path + ": " + std::string(std::strerror(errno))); }
    return p;
}

void evict_file(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fail("open " + path + ": " + std::string(std::strerror(errno))); }
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0) {
        fail("posix_fadvise(DONTNEED): " + std::string(std::strerror(errno)));
    }
    close(fd);
}

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    probe_start("pointer_chase",
                Obj().integer("hops", o.hops).integer("reps", o.reps).str("kinds", o.kinds)
                     .str("patterns", o.patterns).str("file", o.file)
                     .uint("file_bytes", o.file_bytes));
    EL_CUDA(cudaSetDevice(0));
    declare_device(0);
    cudaStream_t stream;
    EL_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    declare_compiled_arch(stream);

    const bool pageable_ok = runtime_attr(cudaDevAttrPageableMemoryAccess, 0) == 1;
    declared("cuda.pageableMemoryAccess",
             std::to_string(runtime_attr(cudaDevAttrPageableMemoryAccess, 0)),
             "cudaDeviceGetAttribute");

    prepare_file(o);
    std::size_t max_chain = 0;
    for (const double mb : o.footprints_mb) {
        max_chain = std::max(max_chain, std::size_t(mb * (1ull << 20)));
    }
    if (max_chain > o.file_bytes) { fail("largest footprint exceeds the scratch file (--bytes)"); }
    void* file_map = map_file(o.file, o.file_bytes);

    // Median hop latency per device-memory footprint (random pattern): the smallest footprint is
    // cache latency, the largest is DRAM, and the difference is the delta the spec promises.
    std::vector<std::pair<double, double>> device_random; // (footprint MiB, median ns)

    for (const double mb : o.footprints_mb) {
        const std::size_t chain_bytes = std::size_t(mb * (1ull << 20));
        const std::size_t n           = chain_bytes / 4;
        if (n < 2) { fail("footprint too small for a chain"); }
        for (const std::string& pattern : split_list(o.patterns)) {
            const std::vector<unsigned> chain = build_chain(n, pattern);
            const WalkCheck oracle            = simulate_walk(chain, o.hops);
            const Obj tags = Obj().num("footprint_mib", mb).uint("footprint_bytes", chain_bytes)
                                 .str("pattern", pattern).uint("hops", o.hops);

            if (o.kinds.find("device") != std::string::npos) {
                unsigned* d = nullptr;
                EL_CUDA(cudaMalloc(&d, chain_bytes));
                EL_CUDA(cudaMemcpy(d, chain.data(), chain_bytes, cudaMemcpyHostToDevice));
                const SeriesOutcome s =
                    chase_series(stream, d, o.hops, o.reps, Obj(tags).str("kind", "device"),
                                 oracle, "chase_hop_latency");
                if (pattern == "random") { device_random.emplace_back(mb, s.median_ns); }
                EL_CUDA(cudaFree(d));
            }

            if (o.kinds.find("pinned") != std::string::npos) {
                unsigned* h = nullptr;
                EL_CUDA(cudaHostAlloc(&h, chain_bytes, cudaHostAllocMapped));
                std::memcpy(h, chain.data(), chain_bytes);
                chase_series(stream, h, o.hops, o.reps, Obj(tags).str("kind", "pinned"), oracle,
                             "chase_hop_latency");
                EL_CUDA(cudaFreeHost(h));
            }

            if (o.kinds.find("pageable") != std::string::npos) {
                if (!pageable_ok) {
                    note("pageableMemoryAccess=0: GPU chase of malloc memory skipped");
                } else {
                    auto* p = static_cast<unsigned*>(std::malloc(chain_bytes));
                    std::memcpy(p, chain.data(), chain_bytes);
                    chase_series(stream, p, o.hops, o.reps, Obj(tags).str("kind", "pageable"),
                                 oracle, "chase_hop_latency");
                    std::free(p);
                }
            }

            if (o.kinds.find("file") != std::string::npos) {
                // The chain is the file's head, so chasing it reads pages the page cache owns.
                auto* fm = static_cast<unsigned*>(file_map);
                std::memcpy(fm, chain.data(), chain_bytes);
                chase_series(stream, fm, o.hops, o.reps,
                             Obj(tags).str("kind", "file").str("residency", "resident"), oracle,
                             "chase_hop_latency");
                // Cold walk: every first touch faults, and only the pages the walk touches pay.
                evict_file(o.file);
                const ChaseResult cold = run_chase(stream, fm, o.hops);
                const bool cold_ok = cold.acc == oracle.acc && cold.final_idx == oracle.final_idx;
                measure("cold_fault_hop", cold.hop_ns, "ns/hop",
                        Obj(tags).str("kind", "file").str("residency", "cold")
                            .uint("unique_pages", oracle.unique_pages)
                            .num("hop_cycles", cold.hop_cycles), cold_ok, 0);
                check("cold_fault_hop.walk_xor", cold_ok,
                      Obj(tags).uint("hops", o.hops).uint("got_acc", cold.acc)
                          .uint("expected_acc", oracle.acc)
                          .uint("unique_pages", oracle.unique_pages));
                if (oracle.unique_pages > 0) {
                    measure("cold_fault_cost", cold.hop_ns * o.hops / oracle.unique_pages,
                            "ns/fault",
                            Obj(tags).str("kind", "file").uint("unique_pages", oracle.unique_pages),
                            cold_ok);
                }
            }
        }
    }
    munmap(file_map, o.file_bytes);

    if (device_random.size() >= 2) {
        double smallest = device_random.front().second, largest = device_random.front().second;
        for (const auto& [mb, ns] : device_random) {
            smallest = std::min(smallest, ns);
            largest  = std::max(largest, ns);
        }
        measure("l2_vs_dram_delta", largest - smallest, "ns/hop",
                Obj().str("kind", "device").str("pattern", "random")
                    .num("smallest_footprint_median_ns", smallest)
                    .num("largest_footprint_median_ns", largest), true);
    }
    return probe_end();
}
