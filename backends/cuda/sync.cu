// sync: how late a stream synchronize returns after the GPU finishes, per CUDA device schedule,
// and what a graph launch costs against its node count. Spec: probes/sync/spec.md.
//
// Ported from ninfer-gb10 tools/gb10/sync_probe.cu. Changes from the original: JSONL records
// with one record per iteration (a series, not a percentile); the spin kernel records its own
// start and end so the oracle can confirm it ran for the requested time; graph nodes each count
// themselves (one atomic from one thread) so a launch that skipped nodes fails its check; a lag
// more negative than the calibration uncertainty fails the clock-mapping check.
//
// The schedule is fixed per process (cudaSetDeviceFlags before the context exists), so
// run/plans.py runs one process per schedule.

#include "common/echolock.cuh"

#include <sched.h>
#include <time.h>

#include <fstream>

using namespace echolock;

namespace {

// Busy-waits for `ns` of %globaltimer, then records its start and end.
__global__ void spin_kernel(std::uint64_t ns, std::uint64_t* stamps) {
    const std::uint64_t start = global_ns();
    while (global_ns() - start < ns) {}
    stamps[0] = start;
    stamps[1] = global_ns();
}

__global__ void first_node_kernel(std::uint64_t* start, unsigned* count) {
    *start = global_ns();
    atomicAdd(count, 1u);
}

__global__ void count_node_kernel(unsigned* count) {
    if (threadIdx.x == 0) { atomicAdd(count, 1u); }
}

std::string read_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}

// CPU time this thread has consumed, in us. Compared with the wall time of a synchronize wait,
// it tells spin (ratio ~1) from yield (in between) from blocking (ratio ~0) -- the observable
// difference between the schedules, measured instead of assumed from the granted flags.
double thread_cpu_us() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return double(ts.tv_sec) * 1e6 + double(ts.tv_nsec) / 1e3;
}

} // namespace

int main(int argc, char** argv) {
    Args args(argc, argv,
              "usage: sync [--schedule auto|spin|yield|blocking (blocking)] [--iterations N (30)] "
              "[--durations-us 50,1000,10000,60000] [--graph-nodes 100,1000,4000]");
    std::string schedule = "blocking";
    int iterations       = 30;
    std::vector<std::uint64_t> durations_us = {50, 1000, 10000, 60000};
    std::vector<int> graph_nodes            = {100, 1000, 4000};
    const auto split = [](const std::string& s) {
        std::vector<std::string> out;
        std::size_t b = 0;
        for (std::size_t e; (e = s.find(',', b)) != std::string::npos; b = e + 1) {
            out.push_back(s.substr(b, e - b));
        }
        out.push_back(s.substr(b));
        return out;
    };
    for (std::string a; args.next(a);) {
        if (a == "--schedule") {
            schedule = args.value(a);
        } else if (a == "--iterations") {
            iterations = std::max(5, std::stoi(args.value(a)));
        } else if (a == "--durations-us") {
            durations_us.clear();
            for (const auto& v : split(args.value(a))) { durations_us.push_back(std::stoull(v)); }
        } else if (a == "--graph-nodes") {
            graph_nodes.clear();
            for (const auto& v : split(args.value(a))) { graph_nodes.push_back(std::max(1, std::stoi(v))); }
        } else {
            args.usage_error("unknown argument " + a);
        }
    }
    unsigned flag = cudaDeviceScheduleBlockingSync;
    if (schedule == "auto") { flag = cudaDeviceScheduleAuto; }
    else if (schedule == "spin") { flag = cudaDeviceScheduleSpin; }
    else if (schedule == "yield") { flag = cudaDeviceScheduleYield; }
    else if (schedule != "blocking") { args.usage_error("unknown schedule " + schedule); }

    const int cpu = sched_getcpu();
    probe_start("sync", Obj().str("schedule", schedule).integer("iterations", iterations)
                             .integer("host_cpu", cpu));
    EL_CUDA(cudaSetDeviceFlags(flag | cudaDeviceMapHost));
    declare_device(0);
    const std::string cpu_dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
    declared("host.cpu_max_khz", "\"" + json_escape(read_line(cpu_dir + "/cpufreq/cpuinfo_max_freq")) + "\"",
             "sysfs");
    declared("host.cpuidle_governor",
             "\"" + json_escape(read_line("/sys/devices/system/cpu/cpuidle/current_governor_ro")) + "\"",
             "sysfs");
    unsigned granted_flags = 0;
    EL_CUDA(cudaGetDeviceFlags(&granted_flags));
    declared("cuda.device_flags", std::to_string(granted_flags), "cudaGetDeviceFlags");

    cudaStream_t stream;
    EL_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    declare_compiled_arch(stream);
    ClockMap clock(stream);
    std::uint64_t* stamps = nullptr; // host: [0,1] spin start/end, [2] first node start
    EL_CUDA(cudaMallocHost(&stamps, 3 * sizeof(std::uint64_t)));
    std::uint64_t* dev_stamps = nullptr;
    EL_CUDA(cudaMalloc(&dev_stamps, 3 * sizeof(std::uint64_t)));
    unsigned* dev_count = nullptr;
    EL_CUDA(cudaMalloc(&dev_count, sizeof(unsigned)));

    // 1. Completion lag: synchronize return time minus the kernel's end, on the host clock.
    //    The wait's CPU fraction says which schedule the device is actually following.
    const std::uint64_t longest_us =
        *std::max_element(durations_us.begin(), durations_us.end());
    std::vector<double> longest_fracs;
    for (const std::uint64_t us : durations_us) {
        const Obj tags = Obj().str("schedule", schedule).uint("kernel_us", us);
        std::vector<double> lags, fracs;
        int short_spins = 0, impossible = 0;
        double worst_rtt_us = 0;
        for (int it = 0; it < iterations; ++it) {
            const ClockMap::Sample cal = clock.calibrate();
            spin_kernel<<<1, 1, 0, stream>>>(us * 1000, dev_stamps);
            const std::int64_t sync_t0 = mono_ns();
            const double cpu0           = thread_cpu_us();
            EL_CUDA(cudaStreamSynchronize(stream));
            const std::int64_t returned = mono_ns();
            const double wait_wall_us   = double(returned - sync_t0) / 1e3;
            const double frac = wait_wall_us > 0 ? (thread_cpu_us() - cpu0) / wait_wall_us : 0.0;
            fracs.push_back(frac);
            EL_CUDA(cudaMemcpy(stamps, dev_stamps, 2 * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
            const double spun_us = double(stamps[1] - stamps[0]) / 1e3;
            const double lag_us =
                double(returned - (static_cast<std::int64_t>(stamps[1]) + cal.offset_ns)) / 1e3;
            const double rtt_us = double(cal.rtt_ns) / 1e3;
            short_spins += spun_us < double(us) ? 1 : 0;
            // The sync cannot return before the kernel ended, beyond the mapping's uncertainty.
            impossible += lag_us < -rtt_us / 2 ? 1 : 0;
            worst_rtt_us = std::max(worst_rtt_us, rtt_us);
            lags.push_back(lag_us);
            measure("sync_return_lag", lag_us, "us", Obj(tags).num("calibration_rtt_us", rtt_us),
                    spun_us >= double(us) && lag_us >= -rtt_us / 2, it);
        }
        measure("sync_return_lag", summarize(lags).p50, "us",
                Obj(tags).obj("per_rep", summary_obj(summarize(lags))), short_spins == 0 && impossible == 0);
        measure("sync_wait_cpu_fraction", summarize(fracs).p50, "x",
                Obj(tags).obj("per_rep", summary_obj(summarize(fracs))), true);
        check("spin_duration", short_spins == 0, Obj(tags).integer("short_spins", short_spins));
        check("clock_mapping", impossible == 0,
              Obj(tags).integer("negative_lags", impossible).num("worst_rtt_us", worst_rtt_us));
        if (us == longest_us) { longest_fracs = fracs; }
    }

    // The schedule the device actually followed, identified from the wait's CPU fraction on the
    // longest kernel (short waits are noise): spin burns the core (~1), blocking sleeps (~0),
    // yield sits between. `auto` makes no promise, so for it the identification is the result.
    if (!longest_fracs.empty()) {
        const double f           = summarize(longest_fracs).p50;
        const std::string which  = f > 0.9 ? "spin" : f < 0.15 ? "blocking" : "yield";
        bool effect              = true;
        if (schedule == "spin") { effect = f > 0.9; }
        else if (schedule == "blocking") { effect = f < 0.15; }
        else if (schedule == "yield") { effect = f >= 0.15 && f <= 0.9; }
        feature("device_schedule", true, (granted_flags & 7) == flag ? 1 : 0, effect,
                Obj().str("requested", schedule).str("identified", which)
                    .num("wait_cpu_fraction_p50", f).uint("kernel_us", longest_us)
                    .str("method", "thread CPU time / wall time while waiting in "
                                   "cudaStreamSynchronize"));
    }

    // 2. Graph launch cost against node count.
    for (const int nodes : graph_nodes) {
        const Obj tags = Obj().str("schedule", schedule).integer("nodes", nodes);
        cudaGraph_t graph;
        EL_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        first_node_kernel<<<1, 1, 0, stream>>>(dev_stamps + 2, dev_count);
        for (int n = 1; n < nodes; ++n) { count_node_kernel<<<1, 32, 0, stream>>>(dev_count); }
        EL_CUDA(cudaStreamEndCapture(stream, &graph));
        cudaGraphExec_t exec;
        EL_CUDA(cudaGraphInstantiate(&exec, graph, 0));
        EL_CUDA(cudaGraphUpload(exec, stream));
        EL_CUDA(cudaGraphLaunch(exec, stream)); // warm
        EL_CUDA(cudaStreamSynchronize(stream));
        std::vector<double> call, first;
        int bad = 0;
        for (int it = 0; it < iterations; ++it) {
            EL_CUDA(cudaMemsetAsync(dev_count, 0, sizeof(unsigned), stream));
            const ClockMap::Sample cal = clock.calibrate(); // also drains the memset
            const std::int64_t t0      = mono_ns();
            EL_CUDA(cudaGraphLaunch(exec, stream));
            const std::int64_t t1 = mono_ns();
            EL_CUDA(cudaStreamSynchronize(stream));
            unsigned count = 0;
            EL_CUDA(cudaMemcpy(&count, dev_count, sizeof(count), cudaMemcpyDeviceToHost));
            EL_CUDA(cudaMemcpy(stamps + 2, dev_stamps + 2, sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
            const bool ok = count == unsigned(nodes);
            bad += ok ? 0 : 1;
            call.push_back(double(t1 - t0) / 1e3);
            first.push_back(double(static_cast<std::int64_t>(stamps[2]) + cal.offset_ns - t0) / 1e3);
            measure("graph_launch_call", call.back(), "us", tags, ok, it);
            measure("graph_launch_to_first_kernel", first.back(), "us",
                    Obj(tags).num("calibration_rtt_us", double(cal.rtt_ns) / 1e3), ok, it);
        }
        measure("graph_launch_call", summarize(call).p50, "us",
                Obj(tags).obj("per_rep", summary_obj(summarize(call))), bad == 0);
        measure("graph_launch_to_first_kernel", summarize(first).p50, "us",
                Obj(tags).obj("per_rep", summary_obj(summarize(first))), bad == 0);
        check("graph_nodes_executed", bad == 0, Obj(tags).integer("bad_launches", bad));
        EL_CUDA(cudaGraphExecDestroy(exec));
        EL_CUDA(cudaGraphDestroy(graph));
    }

    EL_CUDA(cudaFree(dev_count));
    EL_CUDA(cudaFree(dev_stamps));
    EL_CUDA(cudaFreeHost(stamps));
    EL_CUDA(cudaStreamDestroy(stream));
    return probe_end();
}
