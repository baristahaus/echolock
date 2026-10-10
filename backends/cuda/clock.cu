// clock: the SM clock the GPU actually runs at, derived from counters the kernel reads itself
// (%globaltimer nanoseconds vs clock64() SM ticks), idle and under a memory load, as a trace.
// Spec: probes/clock/spec.md.
//
// The declared clock (cudaDeviceGetAttribute, nvidia-smi, the sampler) is a number the driver
// reports. This probe measures the ratio of two hardware counters inside a kernel, so the clock
// it posts is the clock the other probes' work actually ran at.

#include "common/echolock.cuh"

#include <cmath>

using namespace echolock;

namespace {

struct Options {
    double seconds      = 2.0;   // how long the loaded trace runs
    double idle_ms      = 100.0; // per-sample window when idle
    double trace_ms     = 200.0; // per-sample window under load
    std::size_t load_gib = 1;
};

Options parse(int argc, char** argv) {
    Args args(argc, argv,
              "usage: clock [--seconds S (2)] [--idle-ms S (100)] [--trace-ms S (200)] "
              "[--load-gib N (1)]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--seconds") { o.seconds = std::stod(args.value(a)); }
        else if (a == "--idle-ms") { o.idle_ms = std::stod(args.value(a)); }
        else if (a == "--trace-ms") { o.trace_ms = std::stod(args.value(a)); }
        else if (a == "--load-gib") {
            o.load_gib = static_cast<std::size_t>(std::stod(args.value(a)) * (1ull << 30));
        } else { args.usage_error("unknown argument " + a); }
    }
    if (o.seconds <= 0 || o.idle_ms < 10 || o.trace_ms < 10 || o.load_gib == 0) {
        args.usage_error("durations must be positive, windows at least 10 ms");
    }
    return o;
}

// ---- kernels ------------------------------------------------------------------------------------

// One block per SM: reads both counters around a spin, and the SM clock falls out of the ratio.
struct Stamp {
    unsigned sm;
    long long clk0, clk1, t0_ns, t1_ns;
};

__global__ void ratio_kernel(unsigned long long spin_ns, Stamp* out) {
    const unsigned long long t0 = global_ns();
    long long c0                = clock64();
    while (global_ns() - t0 < spin_ns) {}
    long long c1                = clock64();
    const unsigned long long t1 = global_ns();
    if (threadIdx.x == 0) {
        out[blockIdx.x] = Stamp{blockIdx.x, c0, c1, static_cast<long long>(t0),
                                 static_cast<long long>(t1)};
    }
}

// Spins for exactly `ns` of %globaltimer. The host measures its wall time: this is what makes
// %globaltimer trustworthy enough to derive a clock from.
__global__ void spin_for_kernel(unsigned long long ns) {
    const unsigned long long t0 = global_ns();
    while (global_ns() - t0 < ns) {}
}

// The load and fill kernels live in echolock.cuh (load_until_kernel, fill_kernel), verified
// against the same fill generator the memory probe uses.

double stamp_mhz(const Stamp& s) {
    const double dt_ns = double(s.t1_ns - s.t0_ns);
    return dt_ns > 0 ? double(s.clk1 - s.clk0) / dt_ns * 1e3 : 0.0; // ticks/ns = GHz -> MHz
}

// Runs the ratio kernel `reps` times and emits one measure per SM window plus a summary.
// SMs run in lockstep, so a large spread between their derived clocks means the measurement is
// wrong, not the GPU; that is checked here, per condition.
double clock_series(cudaStream_t stream, Stamp* d_stamps, int sms, unsigned long long spin_ns,
                    int reps, const std::string& condition, const std::string& measure_name) {
    std::vector<Stamp> host(sms);
    std::vector<double> all;
    for (int r = 0; r < reps; ++r) {
        ratio_kernel<<<sms, 32, 0, stream>>>(spin_ns, d_stamps);
        EL_CUDA(cudaGetLastError());
        EL_CUDA(cudaStreamSynchronize(stream));
        EL_CUDA(cudaMemcpy(host.data(), d_stamps, host.size() * sizeof(Stamp),
                           cudaMemcpyDeviceToHost));
        for (const Stamp& s : host) {
            const double mhz = stamp_mhz(s);
            all.push_back(mhz);
            measure(measure_name, mhz, "MHz",
                    Obj().str("condition", condition).integer("sm", s.sm).uint("window_ns", spin_ns),
                    mhz > 10.0, r);
        }
    }
    const Summary sum = summarize(all);
    measure("derived_sm_clock_mhz", sum.p50, "MHz",
            Obj().str("condition", condition).obj("per_sm", summary_obj(sum)), true);
    const double spread = sum.min > 0 ? sum.max / sum.min : 0.0;
    check("per_sm_agreement", spread > 0 && spread <= 1.25,
          Obj().str("condition", condition).num("min_mhz", sum.min).num("max_mhz", sum.max)
              .num("spread", spread));
    return sum.p50;
}

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    probe_start("clock", Obj().num("seconds", o.seconds).num("idle_ms", o.idle_ms)
                          .num("trace_ms", o.trace_ms).uint("load_bytes", o.load_gib));
    EL_CUDA(cudaSetDevice(0));
    declare_device(0);
    const int sms = runtime_attr(cudaDevAttrMultiProcessorCount, 0);
    const double declared_mhz = double(runtime_attr(cudaDevAttrClockRate, 0)) / 1e3;
    declared("cuda.clock_khz_mhz", "\"" + std::to_string(declared_mhz) + "\"",
             "cudaDeviceGetAttribute(ClockRate)");
    cudaStream_t stream, load_stream;
    EL_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    EL_CUDA(cudaStreamCreateWithFlags(&load_stream, cudaStreamNonBlocking));
    declare_compiled_arch(stream);

    // 1. Is %globaltimer trustworthy? Spin a fixed GPU time and check the host wall time.
    {
        constexpr unsigned long long kSpinUs = 10000;
        const std::int64_t t0 = mono_ns();
        spin_for_kernel<<<1, 1, 0, stream>>>(kSpinUs * 1000);
        EL_CUDA(cudaStreamSynchronize(stream));
        const double wall_ms = double(mono_ns() - t0) / 1e6;
        const bool ok = wall_ms > 0.7 * double(kSpinUs) / 1e3 &&
                        wall_ms < 1.3 * double(kSpinUs) / 1e3;
        measure("globaltimer_vs_host_ms", wall_ms, "ms",
                Obj().uint("requested_us", kSpinUs), ok);
        check("spin_calibration", ok,
              Obj().uint("requested_us", kSpinUs).num("wall_ms", wall_ms));
    }

    Stamp* d_stamps = nullptr;
    EL_CUDA(cudaMalloc(&d_stamps, std::size_t(sms) * sizeof(Stamp)));
    ClockMap clock_map(stream); // maps %globaltimer to the host clock, for the load deadline

    // 2. Idle clock: nothing else running.
    const int idle_reps    = 5;
    const double idle_mhz  = clock_series(stream, d_stamps, sms,
                                          static_cast<unsigned long long>(o.idle_ms * 1e6),
                                          idle_reps, "idle", "derived_sm_clock_mhz");

    // 3. Clock under a verified memory load, as a trace of windows.
    {
        const std::size_t bytes  = o.load_gib;
        const std::size_t n      = bytes / 16;
        uint4* buf               = nullptr;
        unsigned* slot           = nullptr;
        EL_CUDA(cudaMalloc(&buf, bytes));
        EL_CUDA(cudaMalloc(&slot, 4 * sizeof(unsigned)));
        fill_kernel<<<sms * 8, 256, 0, stream>>>(buf, n, 0x1234567u);
        EL_CUDA(cudaGetLastError());
        EL_CUDA(cudaStreamSynchronize(stream));
        const Xor128 expected =
            expected_fill_xor(n, 0x1234567u, hw_threads()); // same generator as memory.cu

        const unsigned long long until_ns =
            static_cast<unsigned long long>(gpu_now_ns(clock_map))
            + static_cast<unsigned long long>(o.seconds * 1.25e9);
        const unsigned nonce = rep_nonce(0);
        EL_CUDA(cudaMemsetAsync(slot, 0, 4 * sizeof(unsigned), load_stream));
        load_until_kernel<<<sms * 8, 256, 0, load_stream>>>(buf, n, slot, nonce, until_ns);
        EL_CUDA(cudaGetLastError());

        const Obj load_tags = Obj().str("condition", "load").uint("load_bytes", bytes);
        const int windows = std::max(1, static_cast<int>(o.seconds * 1000.0 / o.trace_ms));
        const double load_mhz = clock_series(stream, d_stamps, sms,
                                             static_cast<unsigned long long>(o.trace_ms * 1e6),
                                             windows, "load", "clock_trace");

        EL_CUDA(cudaStreamSynchronize(load_stream));
        std::vector<unsigned> got(4);
        EL_CUDA(cudaMemcpy(got.data(), slot, 4 * sizeof(unsigned), cudaMemcpyDeviceToHost));
        const bool load_ok = slot_value(got, 0) == fold_nonce(expected, nonce);
        check("load.xor", load_ok, Obj().str("expected", expected.hex())
                                     .str("got", Xor128{got[0], got[1], got[2], got[3]}.hex())
                                     .uint("load_bytes", bytes));
        measure("clock_drop_ratio", load_mhz / idle_mhz, "x",
                Obj().num("idle_mhz", idle_mhz).num("load_mhz", load_mhz), load_ok);
        EL_CUDA(cudaFree(buf));
        EL_CUDA(cudaFree(slot));
    }

    // 4. The measured clock against the number the driver reports.
    if (idle_mhz > 0) {
        const double ratio = declared_mhz / idle_mhz;
        measure("declared_over_derived_ratio", ratio, "x",
                Obj().num("declared_mhz", declared_mhz).num("derived_mhz", idle_mhz),
                ratio > 0.1 && ratio < 3.0);
        check("clock_sanity", ratio > 0.1 && ratio < 3.0,
              Obj().num("declared_mhz", declared_mhz).num("derived_mhz", idle_mhz));
    }

    EL_CUDA(cudaFree(d_stamps));
    EL_CUDA(cudaStreamDestroy(stream));
    EL_CUDA(cudaStreamDestroy(load_stream));
    return probe_end();
}
