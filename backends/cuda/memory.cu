// memory: GPU and CPU bandwidth over each kind of memory an integrated GPU can read, CPU/GPU
// contention, and whether requested generic compression has any effect. Spec: probes/memory/spec.md.
//
// Ported from ninfer-gb10 tools/gb10/memory_probe.cu. Changes from the original: JSONL records
// instead of Markdown; every timed read and copy is checked against an XOR oracle; every launch
// is timed on its own, so each measurement is a series rather than a mean.
//
// Build: make -C backends/cuda (no -lcuda; driver calls go through the runtime entry-point query).

#include "common/echolock.cuh"

#include <atomic>
#include <cstring>

using namespace echolock;

namespace {

struct Options {
    std::size_t bytes    = std::size_t{1} << 30; // per buffer
    double seconds       = 0.5;                  // target duration of one series
    unsigned cpu_threads = hw_threads();
    std::vector<std::pair<std::string, std::string>> samples; // label, file
};

Options parse(int argc, char** argv) {
    Args args(argc, argv,
              "usage: memory [--gib N (1)] [--seconds S (0.5)] [--cpu-threads N (all)] "
              "[--sample LABEL=PATH]...");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--gib") {
            o.bytes = static_cast<std::size_t>(std::stod(args.value(a)) * double(1ull << 30));
        } else if (a == "--seconds") {
            o.seconds = std::stod(args.value(a));
        } else if (a == "--cpu-threads") {
            o.cpu_threads = std::max(1u, static_cast<unsigned>(std::stoul(args.value(a))));
        } else if (a == "--sample") {
            const std::string v = args.value(a);
            const auto eq       = v.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 == v.size()) {
                args.usage_error("--sample expects LABEL=PATH, got " + v);
            }
            o.samples.emplace_back(v.substr(0, eq), v.substr(eq + 1));
        } else {
            args.usage_error("unknown argument " + a);
        }
    }
    o.bytes = o.bytes / 4096 * 4096;
    if (o.bytes == 0) { args.usage_error("--gib too small"); }
    return o;
}

// ---- driver API ---------------------------------------------------------------------------------

struct Driver {
    CUresult (*deviceGetAttribute)(int*, CUdevice_attribute, CUdevice)                      = nullptr;
    CUresult (*memGetAllocationGranularity)(size_t*, const CUmemAllocationProp*,
                                            CUmemAllocationGranularity_flags)                  = nullptr;
    CUresult (*memCreate)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
                          unsigned long long)                                                 = nullptr;
    CUresult (*memGetAllocationPropertiesFromHandle)(CUmemAllocationProp*,
                                                     CUmemGenericAllocationHandle)            = nullptr;
    CUresult (*memAddressReserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr,
                                  unsigned long long)                                          = nullptr;
    CUresult (*memMap)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
                       unsigned long long)                                                     = nullptr;
    CUresult (*memSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t)             = nullptr;
    CUresult (*memUnmap)(CUdeviceptr, size_t)                                                 = nullptr;
    CUresult (*memRelease)(CUmemGenericAllocationHandle)                                      = nullptr;
    CUresult (*memAddressFree)(CUdeviceptr, size_t)                                           = nullptr;
};

bool load_driver(Driver& d) {
    return resolve_driver("cuDeviceGetAttribute", d.deviceGetAttribute) &&
           resolve_driver("cuMemGetAllocationGranularity", d.memGetAllocationGranularity) &&
           resolve_driver("cuMemCreate", d.memCreate) &&
           resolve_driver("cuMemGetAllocationPropertiesFromHandle",
                          d.memGetAllocationPropertiesFromHandle) &&
           resolve_driver("cuMemAddressReserve", d.memAddressReserve) &&
           resolve_driver("cuMemMap", d.memMap) && resolve_driver("cuMemSetAccess", d.memSetAccess) &&
           resolve_driver("cuMemUnmap", d.memUnmap) && resolve_driver("cuMemRelease", d.memRelease) &&
           resolve_driver("cuMemAddressFree", d.memAddressFree);
}

struct VmmBuffer {
    CUdeviceptr ptr                     = 0;
    std::size_t size                    = 0;
    CUmemGenericAllocationHandle handle = 0;
    bool compression_granted            = false;
    std::string error;
};

VmmBuffer vmm_alloc(const Driver& d, std::size_t bytes, bool request_compression) {
    VmmBuffer b;
    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = 0;
    if (request_compression) { prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC; }
    std::size_t gran = 0;
    if (d.memGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) !=
            CUDA_SUCCESS ||
        gran == 0) {
        b.error = "cuMemGetAllocationGranularity failed";
        return b;
    }
    b.size = (bytes + gran - 1) / gran * gran;
    if (d.memCreate(&b.handle, b.size, &prop, 0) != CUDA_SUCCESS) {
        b.handle = 0;
        b.error  = "cuMemCreate failed";
        return b;
    }
    CUmemAllocationProp granted{};
    if (d.memGetAllocationPropertiesFromHandle(&granted, b.handle) == CUDA_SUCCESS) {
        b.compression_granted = granted.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC;
    }
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (d.memAddressReserve(&b.ptr, b.size, gran, 0, 0) != CUDA_SUCCESS) {
        b.ptr   = 0;
        b.error = "cuMemAddressReserve failed";
    } else if (d.memMap(b.ptr, b.size, 0, b.handle, 0) != CUDA_SUCCESS ||
               d.memSetAccess(b.ptr, b.size, &access, 1) != CUDA_SUCCESS) {
        b.error = "cuMemMap/cuMemSetAccess failed";
    }
    return b;
}

void vmm_free(const Driver& d, VmmBuffer& b) {
    if (b.ptr) {
        d.memUnmap(b.ptr, b.size);
        d.memAddressFree(b.ptr, b.size);
    }
    if (b.handle) { d.memRelease(b.handle); }
    b = {};
}

// ---- stimulus and oracle ------------------------------------------------------------------------

constexpr std::uint32_t kFillSeed = 0x1234567u;

// 16-byte word i of the random fill. Shared by the fill kernel and the host oracle.
__host__ __device__ __forceinline__ uint4 fill_word(std::uint64_t i, std::uint32_t seed) {
    std::uint32_t x = seed ^ static_cast<std::uint32_t>(i * 2654435761ull) ^
                      static_cast<std::uint32_t>(i >> 32);
    uint4 v;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; v.x = x;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; v.y = x;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; v.z = x;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; v.w = x;
    return v;
}

Xor128 expected_fill_xor(std::size_t n, std::uint32_t seed, unsigned threads) {
    const std::size_t per = (n + threads - 1) / threads;
    std::vector<Xor128> part(threads);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            Xor128 a;
            for (std::size_t i = t * per, end = std::min(n, (t + 1) * per); i < end; ++i) {
                const uint4 v = fill_word(i, seed);
                a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
            }
            part[t] = a;
        });
    }
    for (auto& th : pool) { th.join(); }
    Xor128 r;
    for (const auto& a : part) { r.add(a); }
    return r;
}

__global__ void fill_kernel(uint4* data, std::size_t n, std::uint32_t seed, bool random) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        data[i] = random ? fill_word(i, seed) : make_uint4(0, 0, 0, 0);
    }
}

// Fills `n` words of `dst` by repeating the `period` words of `src`, with kernel stores so the data
// reaches a compressible allocation the same way the fills do.
__global__ void tile_kernel(uint4* dst, std::size_t n, const uint4* src, std::size_t period) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        dst[i] = src[i % period];
    }
}

// Streams the buffer once and folds every word into out4 (four u32, XOR).
__global__ void read_kernel(const uint4* __restrict__ data, std::size_t n, unsigned* out4) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        const uint4 v = __ldcs(data + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    xor_reduce_store(acc, out4);
}

// ---- GPU timing ---------------------------------------------------------------------------------

constexpr int kMaxReps = 1024;

struct Gpu {
    int sms = 0;
    cudaStream_t stream{};
    unsigned* slots = nullptr; // kMaxReps x 4 u32, one XOR result per launch
    std::vector<cudaEvent_t> events;
    dim3 grid() const { return dim3(sms * 8); }
};

Xor128 slot_value(const std::vector<unsigned>& host, int r) {
    return {host[4 * r], host[4 * r + 1], host[4 * r + 2], host[4 * r + 3]};
}

struct Series {
    double gbps     = 0; // bytes over total time
    bool verified   = false;
    int reps        = 0;
    int bad_reps    = 0;
};

// Times `reps` reads of `bytes` at `ptr`, each launch on its own event pair, and checks every
// launch's XOR against `expected`. Emits one measure per launch, one summary and one check.
Series gpu_read_series(const Gpu& g, const void* ptr, std::size_t bytes, double seconds,
                       const Xor128& expected, const std::string& name, const Obj& tags) {
    const std::size_t n = bytes / sizeof(uint4);
    const auto* data    = static_cast<const uint4*>(ptr);

    // Warm launch, also used to size the series.
    EL_CUDA(cudaMemsetAsync(g.slots, 0, 4 * sizeof(unsigned), g.stream));
    EL_CUDA(cudaEventRecord(g.events[0], g.stream));
    read_kernel<<<g.grid(), 256, 0, g.stream>>>(data, n, g.slots);
    EL_CUDA(cudaGetLastError());
    EL_CUDA(cudaEventRecord(g.events[1], g.stream));
    EL_CUDA(cudaEventSynchronize(g.events[1]));
    float warm_ms = 0;
    EL_CUDA(cudaEventElapsedTime(&warm_ms, g.events[0], g.events[1]));
    const int reps = std::clamp(static_cast<int>(seconds * 1000.0 / std::max(warm_ms, 0.01f)), 3,
                                kMaxReps);

    EL_CUDA(cudaMemsetAsync(g.slots, 0, std::size_t(reps) * 4 * sizeof(unsigned), g.stream));
    EL_CUDA(cudaEventRecord(g.events[0], g.stream));
    for (int r = 0; r < reps; ++r) {
        read_kernel<<<g.grid(), 256, 0, g.stream>>>(data, n, g.slots + 4 * r);
        EL_CUDA(cudaEventRecord(g.events[r + 1], g.stream));
    }
    EL_CUDA(cudaEventSynchronize(g.events[reps]));
    EL_CUDA(cudaGetLastError());
    std::vector<unsigned> got(std::size_t(reps) * 4);
    EL_CUDA(cudaMemcpy(got.data(), g.slots, got.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));

    Series s;
    s.reps = reps;
    std::vector<double> per;
    double total_ms = 0;
    for (int r = 0; r < reps; ++r) {
        float ms = 0;
        EL_CUDA(cudaEventElapsedTime(&ms, g.events[r], g.events[r + 1]));
        const bool ok = slot_value(got, r) == expected;
        s.bad_reps += ok ? 0 : 1;
        total_ms += ms;
        const double gbps = double(bytes) / (double(ms) * 1e-3) / 1e9;
        per.push_back(gbps);
        measure(name, gbps, "GB/s", tags, ok, r);
    }
    s.verified = s.bad_reps == 0;
    s.gbps     = double(bytes) * reps / (total_ms * 1e-3) / 1e9;
    Obj stats  = summary_obj(summarize(per));
    measure(name, s.gbps, "GB/s", Obj(tags).integer("reps", reps).obj("per_rep", stats), s.verified);
    Obj detail;
    detail.str("series", name).obj("tags", tags).integer("reps", reps).integer("bad_reps", s.bad_reps)
        .str("expected_xor", expected.hex());
    if (!s.verified) {
        for (int r = 0; r < reps; ++r) {
            if (!(slot_value(got, r) == expected)) {
                detail.str("first_bad_xor", slot_value(got, r).hex()).integer("first_bad_rep", r);
                break;
            }
        }
    }
    check(name + ".xor", s.verified, detail);
    return s;
}

void gpu_fill(const Gpu& g, void* ptr, std::size_t bytes, bool random) {
    fill_kernel<<<g.grid(), 256, 0, g.stream>>>(static_cast<uint4*>(ptr), bytes / sizeof(uint4),
                                                kFillSeed, random);
    EL_CUDA(cudaGetLastError());
    EL_CUDA(cudaStreamSynchronize(g.stream));
}

// ---- CPU side -----------------------------------------------------------------------------------

void cpu_fill_random(void* ptr, std::size_t bytes, unsigned threads) {
    std::vector<std::thread> pool;
    const std::size_t words = bytes / 8, per = (words + threads - 1) / threads;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([=] {
            auto* p               = static_cast<std::uint64_t*>(ptr);
            std::uint64_t x       = 0x9E3779B97F4A7C15ull * (t + 1);
            const std::size_t end = std::min(words, (t + 1) * per);
            for (std::size_t i = t * per; i < end; ++i) {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                p[i] = x;
            }
        });
    }
    for (auto& th : pool) { th.join(); }
}

struct CpuRead {
    double gbps       = 0;
    std::uint64_t passes = 0, bad_passes = 0;
};

// Each thread reads its slice repeatedly until `stop` or `seconds`; every pass's XOR is compared
// with the slice XOR taken in an untimed first pass.
CpuRead cpu_read(const void* ptr, std::size_t bytes, unsigned threads, double seconds,
                 const std::atomic<bool>* stop = nullptr) {
    std::atomic<std::uint64_t> total{0}, passes{0}, bad{0};
    const std::size_t words = bytes / 8, per = (words + threads - 1) / threads;
    std::vector<std::uint64_t> expected(threads);
    {
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                const auto* p = static_cast<const std::uint64_t*>(ptr);
                std::uint64_t a = 0;
                for (std::size_t i = t * per, end = std::min(words, (t + 1) * per); i < end; ++i) {
                    a ^= p[i];
                }
                expected[t] = a;
            });
        }
        for (auto& th : pool) { th.join(); }
    }
    const auto t0       = std::chrono::steady_clock::now();
    const auto deadline = t0 + std::chrono::duration<double>(seconds);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            const auto* p         = static_cast<const std::uint64_t*>(ptr);
            const std::size_t beg = std::min(words, t * per), end = std::min(words, (t + 1) * per);
            std::uint64_t done = 0, n = 0, wrong = 0;
            while ((stop ? !stop->load(std::memory_order_relaxed) : true) &&
                   std::chrono::steady_clock::now() < deadline) {
                std::uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                std::size_t i = beg;
                for (; i + 4 <= end; i += 4) {
                    a0 ^= p[i]; a1 ^= p[i + 1]; a2 ^= p[i + 2]; a3 ^= p[i + 3];
                }
                for (; i < end; ++i) { a0 ^= p[i]; }
                wrong += ((a0 ^ a1 ^ a2 ^ a3) != expected[t]) ? 1 : 0;
                done += (end - beg) * 8;
                ++n;
            }
            total += done;
            passes += n;
            bad += wrong;
        });
    }
    for (auto& th : pool) { th.join(); }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return {double(total.load()) / s / 1e9, passes.load(), bad.load()};
}

void emit_cpu_read(const std::string& name, const CpuRead& r, const Obj& tags) {
    const bool ok = r.bad_passes == 0 && r.passes > 0;
    measure(name, r.gbps, "GB/s", Obj(tags).uint("passes", r.passes), ok);
    check(name + ".xor", ok,
          Obj().uint("passes", r.passes).uint("bad_passes", r.bad_passes).obj("tags", tags));
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { fail("cannot open sample " + path); }
    std::vector<unsigned char> data;
    unsigned char chunk[1 << 16];
    for (std::size_t got; (got = std::fread(chunk, 1, sizeof(chunk), f)) > 0;) {
        data.insert(data.end(), chunk, chunk + got);
    }
    std::fclose(f);
    data.resize(data.size() / sizeof(uint4) * sizeof(uint4));
    if (data.empty()) { fail("sample " + path + " holds fewer than 16 bytes"); }
    return data;
}

// XOR of `n` words made by repeating `src` (period words): whole periods cancel in pairs.
Xor128 tiled_xor(const std::vector<unsigned char>& src, std::size_t n) {
    const std::size_t period = src.size() / 16, full = n / period, rem = n % period;
    Xor128 r = (full & 1) ? host_xor(src.data(), src.size(), 1) : Xor128{};
    r.add(host_xor(src.data(), rem * 16, 1));
    return r;
}

} // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    const int dev     = 0;
    probe_start("memory", Obj()
                              .uint("bytes", opt.bytes)
                              .num("seconds", opt.seconds)
                              .integer("cpu_threads", opt.cpu_threads)
                              .integer("samples", static_cast<long long>(opt.samples.size())));
    EL_CUDA(cudaSetDevice(dev));
    EL_CUDA(cudaFree(nullptr)); // create the primary context for the driver calls below

    // 1. What CUDA reports.
    declare_device(dev);
    const std::pair<const char*, cudaDeviceAttr> attrs[] = {
        {"integrated", cudaDevAttrIntegrated},
        {"pageableMemoryAccess", cudaDevAttrPageableMemoryAccess},
        {"pageableMemoryAccessUsesHostPageTables", cudaDevAttrPageableMemoryAccessUsesHostPageTables},
        {"concurrentManagedAccess", cudaDevAttrConcurrentManagedAccess},
        {"directManagedMemAccessFromHost", cudaDevAttrDirectManagedMemAccessFromHost},
        {"hostNativeAtomicSupported", cudaDevAttrHostNativeAtomicSupported},
        {"canUseHostPointerForRegisteredMem", cudaDevAttrCanUseHostPointerForRegisteredMem},
        {"unifiedAddressing", cudaDevAttrUnifiedAddressing},
        {"memoryPoolsSupported", cudaDevAttrMemoryPoolsSupported},
    };
    for (const auto& [name, a] : attrs) {
        declared(std::string("cuda.") + name, std::to_string(runtime_attr(a, dev)),
                 "cudaDeviceGetAttribute");
    }
    Driver drv;
    const bool have_driver  = load_driver(drv);
    int generic_compression = -1;
    if (have_driver) {
        drv.deviceGetAttribute(&generic_compression,
                               CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, dev);
    } else {
        note("driver entry points unavailable; compression section skipped");
    }
    declared("cuda.GENERIC_COMPRESSION_SUPPORTED", std::to_string(generic_compression),
             "cuDeviceGetAttribute");

    Gpu g;
    g.sms = runtime_attr(cudaDevAttrMultiProcessorCount, dev);
    EL_CUDA(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
    EL_CUDA(cudaMalloc(&g.slots, std::size_t(kMaxReps) * 4 * sizeof(unsigned)));
    g.events.resize(kMaxReps + 1);
    for (auto& e : g.events) { EL_CUDA(cudaEventCreate(&e)); }
    const unsigned host_threads = hw_threads();

    // 2. Generic compression: granted per the API, and whether zero-filled data reads faster.
    if (have_driver) {
        double zero_gbps[2] = {0, 0};
        int granted          = -1;
        for (const bool want : {false, true}) {
            VmmBuffer b       = vmm_alloc(drv, opt.bytes, want);
            const char* alloc = want ? "vmm_compression_requested" : "vmm_plain";
            if (!b.error.empty()) {
                note(std::string(alloc) + ": " + b.error);
                vmm_free(drv, b);
                continue;
            }
            if (want) { granted = b.compression_granted ? 1 : 0; }
            void* p          = reinterpret_cast<void*>(b.ptr);
            const Obj common = Obj().str("source", alloc).boolean("compression_granted",
                                                                 b.compression_granted);
            gpu_fill(g, p, b.size, false);
            zero_gbps[want] = gpu_read_series(g, p, b.size, opt.seconds, Xor128{}, "gpu_read",
                                              Obj(common).str("fill", "zero"))
                                  .gbps;
            gpu_fill(g, p, b.size, true);
            gpu_read_series(g, p, b.size, opt.seconds,
                            expected_fill_xor(b.size / 16, kFillSeed, host_threads), "gpu_read",
                            Obj(common).str("fill", "random"));
            vmm_free(drv, b);
        }
        if (zero_gbps[0] > 0 && zero_gbps[1] > 0) {
            const double gain = zero_gbps[1] / zero_gbps[0];
            feature("generic_compression", true, granted, gain >= 1.2,
                    Obj().num("zero_fill_gain", gain).num("threshold", 1.2)
                        .num("plain_zero_gbps", zero_gbps[0]).num("requested_zero_gbps", zero_gbps[1]));
        }

        // 2b. Real bytes: each sample tiled over the buffer, plain against compression requested.
        for (const auto& [label, path] : opt.samples) {
            const std::vector<unsigned char> host = read_file(path);
            void* staged                          = nullptr;
            EL_CUDA(cudaMalloc(&staged, host.size()));
            EL_CUDA(cudaMemcpy(staged, host.data(), host.size(), cudaMemcpyHostToDevice));
            double gbps[2] = {0, 0};
            for (const bool want : {false, true}) {
                VmmBuffer b = vmm_alloc(drv, opt.bytes, want);
                if (!b.error.empty() || (want && !b.compression_granted)) {
                    note("sample " + label + ": " +
                         (b.error.empty() ? std::string("compression not granted") : b.error));
                    vmm_free(drv, b);
                    break;
                }
                const std::size_t n = b.size / sizeof(uint4);
                tile_kernel<<<g.grid(), 256, 0, g.stream>>>(reinterpret_cast<uint4*>(b.ptr), n,
                                                            static_cast<const uint4*>(staged),
                                                            host.size() / sizeof(uint4));
                EL_CUDA(cudaGetLastError());
                EL_CUDA(cudaStreamSynchronize(g.stream));
                gbps[want] = gpu_read_series(
                                 g, reinterpret_cast<void*>(b.ptr), b.size, opt.seconds,
                                 tiled_xor(host, n), "gpu_read",
                                 Obj().str("source", want ? "vmm_compression_requested" : "vmm_plain")
                                     .str("fill", "sample:" + label)
                                     .uint("sample_bytes", host.size()))
                                 .gbps;
                vmm_free(drv, b);
            }
            EL_CUDA(cudaFree(staged));
            if (gbps[0] > 0 && gbps[1] > 0) {
                measure("compression_gain", gbps[1] / gbps[0], "x",
                        Obj().str("fill", "sample:" + label), true);
            }
        }
    }

    // 3. GPU reads of each memory kind, and copies.
    void* dev_buf = nullptr;
    EL_CUDA(cudaMalloc(&dev_buf, opt.bytes));
    gpu_fill(g, dev_buf, opt.bytes, true);
    const Xor128 dev_xor = expected_fill_xor(opt.bytes / 16, kFillSeed, host_threads);
    gpu_read_series(g, dev_buf, opt.bytes, opt.seconds, dev_xor, "gpu_read",
                    Obj().str("source", "cudaMalloc").str("fill", "random"));

    void* pinned = nullptr;
    EL_CUDA(cudaHostAlloc(&pinned, opt.bytes, cudaHostAllocMapped));
    cpu_fill_random(pinned, opt.bytes, opt.cpu_threads);
    const Xor128 host_data_xor = host_xor(pinned, opt.bytes, host_threads);
    gpu_read_series(g, pinned, opt.bytes, opt.seconds, host_data_xor, "gpu_read",
                    Obj().str("source", "cudaHostAlloc_mapped").str("fill", "random"));

    void* pageable = std::malloc(opt.bytes);
    if (pageable == nullptr) { fail("malloc of the pageable buffer failed"); }
    cpu_fill_random(pageable, opt.bytes, opt.cpu_threads); // same generator: same XOR as pinned
    const bool pageable_ok = runtime_attr(cudaDevAttrPageableMemoryAccess, dev) == 1;
    if (pageable_ok) {
        gpu_read_series(g, pageable, opt.bytes, opt.seconds, host_data_xor, "gpu_read",
                        Obj().str("source", "malloc_pageable").str("fill", "random"));
    } else {
        note("pageableMemoryAccess=0: GPU read of malloc memory skipped");
    }

    // Copies: every copy is followed by an untimed device read of the destination.
    const auto copy_series = [&](const void* src, const char* kind) {
        const int reps = std::clamp(static_cast<int>(opt.seconds * 20), 3, 64);
        EL_CUDA(cudaMemcpyAsync(dev_buf, src, opt.bytes, cudaMemcpyHostToDevice, g.stream)); // warm
        EL_CUDA(cudaMemsetAsync(g.slots, 0, std::size_t(reps) * 4 * sizeof(unsigned), g.stream));
        for (int r = 0; r < reps; ++r) {
            EL_CUDA(cudaEventRecord(g.events[2 * r], g.stream));
            EL_CUDA(cudaMemcpyAsync(dev_buf, src, opt.bytes, cudaMemcpyHostToDevice, g.stream));
            EL_CUDA(cudaEventRecord(g.events[2 * r + 1], g.stream));
            read_kernel<<<g.grid(), 256, 0, g.stream>>>(static_cast<const uint4*>(dev_buf),
                                                        opt.bytes / 16, g.slots + 4 * r);
        }
        EL_CUDA(cudaStreamSynchronize(g.stream));
        EL_CUDA(cudaGetLastError());
        std::vector<unsigned> got(std::size_t(reps) * 4);
        EL_CUDA(cudaMemcpy(got.data(), g.slots, got.size() * sizeof(unsigned),
                           cudaMemcpyDeviceToHost));
        const Obj tags = Obj().str("source", kind).str("direction", "h2d");
        std::vector<double> per;
        int bad = 0;
        for (int r = 0; r < reps; ++r) {
            float ms = 0;
            EL_CUDA(cudaEventElapsedTime(&ms, g.events[2 * r], g.events[2 * r + 1]));
            const bool ok = slot_value(got, r) == host_data_xor;
            bad += ok ? 0 : 1;
            per.push_back(double(opt.bytes) / (double(ms) * 1e-3) / 1e9);
            measure("copy_payload", per.back(), "GB/s", tags, ok, r);
        }
        const Summary s = summarize(per);
        measure("copy_payload", s.mean, "GB/s", Obj(tags).integer("reps", reps).obj("per_rep", summary_obj(s)),
                bad == 0);
        check(std::string("copy_payload.") + kind + ".xor", bad == 0,
              Obj().integer("reps", reps).integer("bad_reps", bad));
    };
    copy_series(pinned, "cudaHostAlloc_mapped");
    copy_series(pageable, "malloc_pageable");

    // 4. CPU bandwidth, and CPU/GPU contention on the shared memory.
    const Obj cpu_tags = Obj().str("source", "malloc_pageable").integer("threads", opt.cpu_threads);
    emit_cpu_read("cpu_read", cpu_read(pageable, opt.bytes, opt.cpu_threads, opt.seconds * 2),
                  Obj(cpu_tags).str("contention", "none"));
    const double gpu_alone =
        gpu_read_series(g, dev_buf, opt.bytes, opt.seconds * 2, dev_xor, "gpu_read",
                        Obj().str("source", "cudaMalloc").str("fill", "random").str("contention", "none"))
            .gbps;

    std::atomic<bool> stop{false};
    CpuRead cpu_shared;
    std::thread cpu_side([&] {
        cpu_shared = cpu_read(pageable, opt.bytes, opt.cpu_threads, 3600, &stop);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const double gpu_shared =
        gpu_read_series(g, dev_buf, opt.bytes, opt.seconds * 2, dev_xor, "gpu_read",
                        Obj().str("source", "cudaMalloc").str("fill", "random").str("contention", "cpu_read"))
            .gbps;
    stop = true;
    cpu_side.join();
    // The CPU window includes its untimed verification pass and a 100 ms CPU-only lead-in.
    emit_cpu_read("cpu_read", cpu_shared, Obj(cpu_tags).str("contention", "gpu_read"));
    measure("shared_sum", gpu_shared + cpu_shared.gbps, "GB/s",
            Obj().str("note", "GPU read + CPU read while contending"), true);
    measure("gpu_read_contention_ratio", gpu_shared / gpu_alone, "x", Obj(), true);

    std::free(pageable);
    EL_CUDA(cudaFreeHost(pinned));
    EL_CUDA(cudaFree(dev_buf));
    for (auto& e : g.events) { EL_CUDA(cudaEventDestroy(e)); }
    EL_CUDA(cudaFree(g.slots));
    EL_CUDA(cudaStreamDestroy(g.stream));
    return probe_end();
}
