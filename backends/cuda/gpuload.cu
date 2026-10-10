// gpuload: a GPU background load for the grid runner. Streams a buffer with a duty cycle
// (busy for on_ms, idle for off_ms), so "50% GPU" is an occupancy the kernel's own timestamps
// measured, not an intention. Every busy cycle's reads are XOR-checked against the fill
// generator, so the achieved bandwidth is bandwidth of verified work.
//
// Protocol (stdout JSONL, like the Python loads in load/): load_start, load_ready, heartbeat
// {achieved duty, gbps}, load_end. Stops on SIGTERM, on its deadline, or when its parent is
// gone (a load must not outlive the runner that started it).
//
// Usage: gpuload --duty 50 --on-ms 100 --gib 1 --deadline-min 120

#include "common/echolock.cuh"

#include <unistd.h>

#include <csignal>
#include <cstdlib>

using namespace echolock;

namespace {

volatile std::sig_atomic_t g_stop = 0;

void on_term(int) { g_stop = 1; }

// A load whose runner died must not keep holding the GPU. The runner passes --parent-pid; if
// /proc/<pid> disappears the load exits. (getppid() alone fails when the parent dies before the
// load captures it, so the explicit pid is the honest check.)
bool parent_alive(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--parent-pid") {
            return access(("/proc/" + std::string(argv[i + 1])).c_str(), F_OK) == 0;
        }
    }
    return true;
}

struct Options {
    double duty       = 0.5;
    double on_ms      = 100.0;
    std::size_t gib   = 1;
    double deadline_min = 120.0;
};

Options parse(int argc, char** argv) {
    Args args(argc, argv,
              "usage: gpuload --duty 50 --on-ms 100 --gib 1 --deadline-min 120");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--duty") { o.duty = std::stod(args.value(a)) / 100.0; }
        else if (a == "--on-ms") { o.on_ms = std::stod(args.value(a)); }
        else if (a == "--gib") {
            o.gib = static_cast<std::size_t>(std::stod(args.value(a)) * (1ull << 30));
        } else if (a == "--deadline-min") { o.deadline_min = std::stod(args.value(a)); }
        else if (a == "--parent-pid") { /* read below, once options are parsed */ args.value(a); }
        else { args.usage_error("unknown argument " + a); }
    }
    if (o.duty <= 0 || o.duty > 1 || o.on_ms < 1 || o.gib == 0) {
        args.usage_error("duty in (0,1], on-ms >= 1, gib > 0");
    }
    return o;
}

void emit_load(const std::string& type, const Obj& fields = Obj()) {
    const std::string f = fields.text();
    std::printf("{\"v\":1,\"t_ns\":%lld,\"src\":\"load\",\"type\":\"%s\"%s%s\n",
                static_cast<long long>(mono_ns()), type.c_str(), f.size() > 2 ? "," : "",
                f.size() > 2 ? f.substr(1).c_str() : "}");
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    std::signal(SIGTERM, on_term);
    std::signal(SIGINT, on_term);

    emit_load("load_start", Obj().str("load", "gpu")
                  .num("duty", o.duty).num("on_ms", o.on_ms).uint("bytes", o.gib)
                  .num("deadline_min", o.deadline_min));

    EL_CUDA(cudaSetDevice(0));
    const int sms = runtime_attr(cudaDevAttrMultiProcessorCount, 0);
    cudaStream_t stream;
    EL_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ClockMap clock_map(stream); // for GPU-time deadlines, not for timing the load itself

    const std::size_t bytes = o.gib, n = bytes / 16;
    uint4* buf = nullptr;
    unsigned* slot = nullptr;
    EL_CUDA(cudaMalloc(&buf, bytes));
    EL_CUDA(cudaMalloc(&slot, 4 * sizeof(unsigned)));
    fill_kernel<<<sms * 8, 256, 0, stream>>>(buf, n, 0x1234567u);
    EL_CUDA(cudaGetLastError());
    EL_CUDA(cudaStreamSynchronize(stream));
    const Xor128 expected = expected_fill_xor(n, 0x1234567u, hw_threads());

    emit_load("load_ready", Obj().str("load", "gpu").num("duty", o.duty)
                  .num("on_ms", o.on_ms).uint("bytes", bytes));

    const double off_ms      = o.on_ms * (1 - o.duty) / o.duty;
    const double deadline_s  = o.deadline_min * 60.0;
    const std::int64_t t0    = mono_ns();
    const std::int64_t gpu_offset = clock_map.calibrate().offset_ns; // one calibration: the
    // deadline only needs the mapping, not per-cycle precision
    double busy_s = 0.0, verified_bytes = 0.0;
    unsigned long long cycles = 0;
    while (!g_stop && double(mono_ns() - t0) / 1e9 < deadline_s) {
        if (!parent_alive(argc, argv)) { break; } // the runner is gone; do not hold the GPU
        ++cycles;
        EL_CUDA(cudaMemsetAsync(slot, 0, 4 * sizeof(unsigned), stream));
        const unsigned long long until_ns =
            static_cast<unsigned long long>(mono_ns() - gpu_offset +
                                            static_cast<std::int64_t>(o.on_ms * 1e6));
        load_until_kernel<<<sms * 8, 256, 0, stream>>>(buf, n, slot, rep_nonce(cycles), until_ns);
        EL_CUDA(cudaStreamSynchronize(stream));
        EL_CUDA(cudaGetLastError());
        std::vector<unsigned> got(4);
        EL_CUDA(cudaMemcpy(got.data(), slot, 4 * sizeof(unsigned), cudaMemcpyDeviceToHost));
        if (!(slot_value(got, 0) == fold_nonce(expected, rep_nonce(cycles)))) {
            emit_load("load_error", Obj().str("message", "load cycle XOR mismatch"));
            EL_CUDA(cudaFree(buf));
            EL_CUDA(cudaFree(slot));
            return 1;
        }
        busy_s += o.on_ms / 1e3; // the kernel spins until its deadline, so it was GPU-busy
        verified_bytes += double(bytes);
        if (cycles % 20 == 0) { // ~2 s at the default 100 ms on-window
            const double elapsed = double(mono_ns() - t0) / 1e9;
            emit_load("heartbeat", Obj().obj("achieved", Obj().num("duty", busy_s / elapsed)
                                                  .num("gbps", verified_bytes / busy_s)));
        }
        if (off_ms > 0) { usleep(static_cast<useconds_t>(off_ms * 1000)); }
    }

    const double elapsed = double(mono_ns() - t0) / 1e9;
    emit_load("load_end", Obj().obj("achieved", Obj().num("duty", busy_s / elapsed)
                                        .num("gbps", verified_bytes / std::max(busy_s, 1e-9))
                                        .num("cycles", double(cycles)))
                  .num("seconds", elapsed));
    EL_CUDA(cudaFree(buf));
    EL_CUDA(cudaFree(slot));
    EL_CUDA(cudaStreamDestroy(stream));
    return 0;
}
