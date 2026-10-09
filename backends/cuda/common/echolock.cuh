// Shared plumbing for the CUDA backend: JSONL records on stdout, error handling, driver-API entry
// points without -lcuda, the host clock, the polled %globaltimer calibration, and the XOR oracle
// helpers the probes verify their reads with.
//
// Record format: docs/record-format.md. Every line on stdout is one JSON object; diagnostics go to
// stderr. Exit codes: 0 ok, 1 runtime error, 2 usage, 3 a verification check failed.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace echolock {

constexpr int kExitOk = 0, kExitError = 1, kExitUsage = 2, kExitVerifyFailed = 3;

// CLOCK_MONOTONIC in ns: the clock every record's t_ns uses (the runner, sampler and kernel log
// share it, so the streams line up).
inline std::int64_t mono_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ---- JSONL --------------------------------------------------------------------------------------

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (const unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

// One JSON object, built field by field. Nested objects are passed as finished Obj values.
class Obj {
  public:
    Obj& str(const char* k, const std::string& v) { return raw(k, "\"" + json_escape(v) + "\""); }
    Obj& num(const char* k, double v) {
        if (!std::isfinite(v)) { return raw(k, "null"); }
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.10g", v);
        return raw(k, buf);
    }
    Obj& integer(const char* k, long long v) { return raw(k, std::to_string(v)); }
    Obj& uint(const char* k, unsigned long long v) { return raw(k, std::to_string(v)); }
    Obj& boolean(const char* k, bool v) { return raw(k, v ? "true" : "false"); }
    Obj& obj(const char* k, const Obj& v) { return raw(k, v.text()); }
    Obj& raw(const char* k, const std::string& json) {
        body_ += (body_.empty() ? "\"" : ",\"") + json_escape(k) + "\":" + json;
        return *this;
    }
    std::string text() const { return "{" + body_ + "}"; }

  private:
    std::string body_;
};

// Process-wide probe identity, set once by Emitter::start.
struct Identity {
    std::string probe = "unknown", backend = "cuda";
    int failed_checks = 0;
};
inline Identity& identity() {
    static Identity id;
    return id;
}

// Writes one record: {"v":1,"t_ns":...,"src":"probe","probe":...,"backend":...,"type":...,<fields>}.
inline void emit(const char* type, const Obj& fields = Obj()) {
    const std::string f = fields.text();
    std::printf("{\"v\":1,\"t_ns\":%lld,\"src\":\"probe\",\"probe\":\"%s\",\"backend\":\"%s\","
                "\"type\":\"%s\"%s%s\n",
                static_cast<long long>(mono_ns()), json_escape(identity().probe).c_str(),
                json_escape(identity().backend).c_str(), type, f.size() > 2 ? "," : "",
                f.size() > 2 ? f.substr(1).c_str() : "}");
    std::fflush(stdout);
}

inline void probe_start(const std::string& probe, const Obj& params) {
    identity().probe = probe;
    emit("probe_start", Obj().obj("params", params));
}

// Ends the probe with the status the checks earned; returns the process exit code.
inline int probe_end() {
    const bool ok = identity().failed_checks == 0;
    emit("probe_end", Obj()
                          .str("status", ok ? "ok" : "verify_failed")
                          .integer("failed_checks", identity().failed_checks));
    return ok ? kExitOk : kExitVerifyFailed;
}

[[noreturn]] inline void fail(const std::string& message) {
    emit("error", Obj().str("message", message));
    emit("probe_end", Obj().str("status", "error").str("message", message));
    std::fprintf(stderr, "echolock: %s\n", message.c_str());
    std::exit(kExitError);
}

// What an API reports. Declared values only choose probes; the report sets them beside measurements.
inline void declared(const std::string& key, const std::string& json_value,
                     const std::string& source) {
    emit("declared", Obj().str("key", key).raw("value", json_value).str("source", source));
}

// One measured value. `tags` identify the series (e.g. source, fill); `rep` is the index within a
// series, -1 for a summary value. `verified` is whether the oracle accepted the work it timed.
inline void measure(const std::string& name, double value, const std::string& unit,
                    const Obj& tags, bool verified, long long rep = -1) {
    Obj o;
    o.str("name", name).num("value", value).str("unit", unit).obj("tags", tags).boolean(
        "verified", verified);
    if (rep >= 0) { o.integer("rep", rep); }
    emit("measure", o);
}

// An oracle verdict. A failed check makes the probe exit with kExitVerifyFailed.
inline bool check(const std::string& name, bool pass, const Obj& detail = Obj()) {
    if (!pass) { ++identity().failed_checks; }
    emit("check", Obj().str("name", name).boolean("pass", pass).obj("detail", detail));
    return pass;
}

// A requested feature whose effect was (or was not) observed. effect_observed=false with
// granted=true is a silent fallback.
inline void feature(const std::string& name, bool requested, int granted, bool effect_observed,
                    const Obj& evidence) {
    Obj o;
    o.str("feature", name).boolean("requested", requested);
    if (granted < 0) { o.raw("granted", "null"); } else { o.boolean("granted", granted != 0); }
    o.boolean("effect_observed", effect_observed).obj("evidence", evidence);
    emit("feature", o);
}

inline void note(const std::string& message) { emit("note", Obj().str("message", message)); }

inline std::string hex64(std::uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "0x%016llx", static_cast<unsigned long long>(v));
    return buf;
}

// ---- errors -------------------------------------------------------------------------------------

#define EL_CUDA(expr)                                                                              \
    do {                                                                                           \
        const cudaError_t e_ = (expr);                                                             \
        if (e_ != cudaSuccess) {                                                                   \
            ::echolock::fail(std::string(#expr) + " -> " + cudaGetErrorName(e_) + ": " +           \
                             cudaGetErrorString(e_) + " (" + __FILE__ + ":" +                      \
                             std::to_string(__LINE__) + ")");                                      \
        }                                                                                          \
    } while (false)

// ---- driver API through the runtime entry-point query (no -lcuda) -------------------------------

template <typename Fn>
bool resolve_driver(const char* symbol, Fn& fn, int version = 12000) {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status{};
    if (cudaGetDriverEntryPointByVersion(symbol, &ptr, version, cudaEnableDefault, &status) !=
            cudaSuccess ||
        status != cudaDriverEntryPointSuccess || ptr == nullptr) {
        (void)cudaGetLastError();
        return false;
    }
    fn = reinterpret_cast<Fn>(ptr);
    return true;
}

inline int runtime_attr(cudaDeviceAttr a, int dev) {
    int v = -1;
    if (cudaDeviceGetAttribute(&v, a, dev) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    return v;
}

// The arch the device code was actually compiled for, read out of the binary at runtime. The
// build stamp (build/cuda/build.json) says what was asked for; this says what the code is.
__global__ void compiled_arch_kernel(int* out) {
#if defined(__CUDA_ARCH__)
    out[0] = static_cast<int>(__CUDA_ARCH__);
#endif
}

inline void declare_compiled_arch(cudaStream_t stream) {
    int* dev = nullptr;
    int host = -1;
    EL_CUDA(cudaMalloc(&dev, sizeof(int)));
    EL_CUDA(cudaMemset(dev, 0, sizeof(int)));
    compiled_arch_kernel<<<1, 1, 0, stream>>>(dev);
    EL_CUDA(cudaStreamSynchronize(stream));
    EL_CUDA(cudaMemcpy(&host, dev, sizeof(int), cudaMemcpyDeviceToHost));
    EL_CUDA(cudaFree(dev));
    declared("cuda.compiled_arch",
             host > 0 ? "\"" + std::to_string(host / 10) + "\"" : "\"unknown\"", "__CUDA_ARCH__");
}

// Declares the device identity every CUDA probe records.
inline void declare_device(int dev) {
    cudaDeviceProp p{};
    EL_CUDA(cudaGetDeviceProperties(&p, dev));
    int drv = 0, rt = 0;
    EL_CUDA(cudaDriverGetVersion(&drv));
    EL_CUDA(cudaRuntimeGetVersion(&rt));
    declared("device.name", "\"" + json_escape(p.name) + "\"", "cudaGetDeviceProperties");
    declared("device.cc", "\"" + std::to_string(p.major) + "." + std::to_string(p.minor) + "\"",
             "cudaGetDeviceProperties");
    declared("device.sm_count", std::to_string(p.multiProcessorCount), "cudaGetDeviceProperties");
    declared("device.l2_bytes", std::to_string(p.l2CacheSize), "cudaGetDeviceProperties");
    declared("device.bus_width_bits",
             std::to_string(runtime_attr(cudaDevAttrGlobalMemoryBusWidth, dev)),
             "cudaDeviceGetAttribute");
    declared("device.clock_khz", std::to_string(runtime_attr(cudaDevAttrClockRate, dev)),
             "cudaDeviceGetAttribute");
    declared("device.memory_clock_khz",
             std::to_string(runtime_attr(cudaDevAttrMemoryClockRate, dev)),
             "cudaDeviceGetAttribute");
    declared("device.total_global_bytes", std::to_string(p.totalGlobalMem),
             "cudaGetDeviceProperties");
    declared("cuda.driver_version", std::to_string(drv), "cudaDriverGetVersion");
    declared("cuda.runtime_version", std::to_string(rt), "cudaRuntimeGetVersion");
}

// ---- GPU clock --------------------------------------------------------------------------------

__device__ __forceinline__ std::uint64_t global_ns() {
    std::uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void calibration_stamp_kernel(volatile std::uint64_t* out, std::uint64_t seq) {
    out[0] = global_ns();
    __threadfence_system();
    out[1] = seq;
}

// Maps %globaltimer to mono_ns without ever calling a CUDA synchronize: the host polls mapped
// memory for the kernel's sequence number and keeps the sample with the smallest round trip.
// host_ns ~= gpu_ns + offset_ns, uncertain by +-rtt_ns/2.
class ClockMap {
  public:
    explicit ClockMap(cudaStream_t stream) : stream_(stream) {
        std::uint64_t* host = nullptr;
        EL_CUDA(cudaHostAlloc(&host, 2 * sizeof(std::uint64_t), cudaHostAllocMapped));
        EL_CUDA(cudaHostGetDevicePointer(&dev_, host, 0));
        host_ = host;
        host_[0] = host_[1] = 0;
    }
    ~ClockMap() { cudaFreeHost(const_cast<std::uint64_t*>(host_)); }
    ClockMap(const ClockMap&)            = delete;
    ClockMap& operator=(const ClockMap&) = delete;

    struct Sample {
        std::int64_t offset_ns = 0, rtt_ns = 0;
    };

    Sample calibrate(int rounds = 200) {
        Sample best{0, INT64_MAX};
        for (int r = 0; r < rounds; ++r) {
            const std::uint64_t seq = ++seq_;
            const std::int64_t t0   = mono_ns();
            calibration_stamp_kernel<<<1, 1, 0, stream_>>>(dev_, seq);
            const std::int64_t deadline = t0 + std::int64_t{5'000'000'000};
            while (host_[1] != seq) {
                if (mono_ns() > deadline) { fail("clock calibration: stamp kernel never landed"); }
            }
            const std::int64_t t1 = mono_ns();
            if (t1 - t0 < best.rtt_ns) {
                best.rtt_ns    = t1 - t0;
                best.offset_ns = (t0 + t1) / 2 - static_cast<std::int64_t>(host_[0]);
            }
        }
        last_ = best;
        return best;
    }
    Sample last() const { return last_; }

  private:
    cudaStream_t stream_;
    volatile std::uint64_t* host_ = nullptr;
    std::uint64_t* dev_           = nullptr;
    std::uint64_t seq_            = 0;
    Sample last_{};
};

// ---- XOR oracle ---------------------------------------------------------------------------------
//
// Read kernels fold everything they load into a 128-bit XOR. The host knows the expected XOR from
// the generator (or from its own copy of the bytes), so a read that returns wrong data, or skips
// data, fails the check instead of posting a fast number.
//
// A slot that was never written holds zeros, and a zero-filled buffer is expected to XOR to zero
// too -- so for zero data the check above cannot tell "read the zeros" from "read nothing". Every
// launch therefore folds a per-rep nonce into lane x of its slot: the kernel's result is
// expected^nonce, and an untouched slot (0) only matches when the nonce is zero, which it never is.
// The host folds the same nonce into the expectation before comparing.

struct Xor128 {
    std::uint32_t x = 0, y = 0, z = 0, w = 0;
    void add(const Xor128& o) { x ^= o.x; y ^= o.y; z ^= o.z; w ^= o.w; }
    bool operator==(const Xor128& o) const { return x == o.x && y == o.y && z == o.z && w == o.w; }
    std::string hex() const {
        char buf[48];
        std::snprintf(buf, sizeof(buf), "%08x%08x%08x%08x", w, z, y, x);
        return buf;
    }
};

// The expectation for a launch that folded `nonce` into its slot.
inline Xor128 fold_nonce(const Xor128& e, unsigned nonce) {
    return {e.x ^ nonce, e.y, e.z, e.w};
}

// A per-launch nonce: never zero, different for every rep of a series.
inline unsigned rep_nonce(long long rep) {
    return 0xA5A5A500u ^ static_cast<unsigned>(rep + 1);
}

__device__ __forceinline__ void xor_reduce_store(uint4 acc, unsigned* out4, unsigned nonce = 0) {
    for (int s = 16; s > 0; s >>= 1) {
        acc.x ^= __shfl_xor_sync(0xffffffffu, acc.x, s);
        acc.y ^= __shfl_xor_sync(0xffffffffu, acc.y, s);
        acc.z ^= __shfl_xor_sync(0xffffffffu, acc.z, s);
        acc.w ^= __shfl_xor_sync(0xffffffffu, acc.w, s);
    }
    if ((threadIdx.x & 31) == 0) {
        // The nonce is folded exactly once per launch (one thread), so it survives the reduction.
        const unsigned n = (threadIdx.x == 0 && blockIdx.x == 0) ? nonce : 0u;
        atomicXor(out4 + 0, acc.x ^ n);
        atomicXor(out4 + 1, acc.y);
        atomicXor(out4 + 2, acc.z);
        atomicXor(out4 + 3, acc.w);
    }
}

// XOR of a host buffer of 16-byte words, split across threads.
inline Xor128 host_xor(const void* data, std::size_t bytes, unsigned threads) {
    const std::size_t n = bytes / 16, per = (n + threads - 1) / threads;
    std::vector<Xor128> part(threads);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            const auto* p = static_cast<const std::uint32_t*>(data);
            Xor128 a;
            for (std::size_t i = t * per, end = std::min(n, (t + 1) * per); i < end; ++i) {
                a.x ^= p[4 * i]; a.y ^= p[4 * i + 1]; a.z ^= p[4 * i + 2]; a.w ^= p[4 * i + 3];
            }
            part[t] = a;
        });
    }
    for (auto& th : pool) { th.join(); }
    Xor128 r;
    for (const auto& a : part) { r.add(a); }
    return r;
}

inline unsigned hw_threads() { return std::max(1u, std::thread::hardware_concurrency()); }

// Summary statistics of a series, for the summary records that accompany per-rep ones.
struct Summary {
    double min = 0, p50 = 0, p90 = 0, max = 0, mean = 0;
};
inline Summary summarize(std::vector<double> v) {
    Summary s;
    if (v.empty()) { return s; }
    std::sort(v.begin(), v.end());
    const auto at = [&](double q) { return v[std::min(v.size() - 1, std::size_t(q * v.size()))]; };
    double sum    = 0;
    for (const double x : v) { sum += x; }
    s.min  = v.front();
    s.p50  = at(0.5);
    s.p90  = at(0.9);
    s.max  = v.back();
    s.mean = sum / double(v.size());
    return s;
}

inline Obj summary_obj(const Summary& s) {
    return Obj().num("min", s.min).num("p50", s.p50).num("p90", s.p90).num("max", s.max).num(
        "mean", s.mean);
}

// Small argument parser: --key value pairs and bare flags, with a usage string on error.
class Args {
  public:
    Args(int argc, char** argv, std::string usage) : usage_(std::move(usage)) {
        for (int i = 1; i < argc; ++i) { args_.emplace_back(argv[i]); }
    }
    bool next(std::string& key) {
        if (pos_ >= args_.size()) { return false; }
        key = args_[pos_++];
        if (key == "--help" || key == "-h") {
            std::fprintf(stdout, "%s\n", usage_.c_str());
            std::exit(kExitOk);
        }
        return true;
    }
    std::string value(const std::string& key) {
        if (pos_ >= args_.size()) { usage_error("missing value after " + key); }
        return args_[pos_++];
    }
    [[noreturn]] void usage_error(const std::string& why) const {
        std::fprintf(stderr, "%s\n%s\n", why.c_str(), usage_.c_str());
        std::exit(kExitUsage);
    }

  private:
    std::vector<std::string> args_;
    std::size_t pos_ = 0;
    std::string usage_;
};

} // namespace echolock
