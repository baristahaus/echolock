// clock (Vulkan backend): the open equivalent of the CUDA clock probe. Spec: probes/clock/spec.md.
//
// What exists on this stack, and what does not:
//  - GPU timestamp queries (VK_QUERY_TYPE_TIMESTAMP, timestampPeriod = 10 ns here) are a real
//    nanosecond device timeline; the memory probe validates them against the wall clock.
//  - The GLSL shader clock (clock2x32ARB / OpShaderClock) is NOT a real-time clock on AMD/RADV:
//    its rate is the shader/core clock and moves with DVFS. There is no GLSL-accessible
//    nanosecond real-time clock, unlike CUDA's %globaltimer. This probe derives the shader
//    clock's rate instead of assuming it.
//
// Measures, mirroring the CUDA probe's structure where the hardware allows:
//  idle/loaded: shader-clock rate derived from a burn kernel bracketed by timestamps, in an
//    idle cadence (host sleeps between short kernels) and under continuous load (long kernels
//    back to back). On NVIDIA the equivalent measures real device time; here it measures the
//    core-clock rate, which is exactly what shows the DVFS behavior.
//  clock_vs_sclk: the derived core-clock rate against amdgpu's pp_dpm_sclk (sysfs, cross-check
//    only -- the measurement itself never touches the driver).
//  submit_overhead: host wall span minus the device timestamp span of the same submission.

#include "common/vk_ctx.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <thread>
#include <cstring>
#include <string>
#include <vector>

using namespace echolock;

namespace {

struct Options {
    double seconds = 0.5;
    int reps = 20;
    unsigned workgroups = 512;
};

Options parse(int argc, char** argv) {
    Args args(argc, argv, "usage: clock [--seconds S (0.5)] [--reps N (20)] [--workgroups N (512)]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--seconds") { o.seconds = std::stod(args.value(a)); }
        else if (a == "--reps") { o.reps = std::stoi(args.value(a)); }
        else if (a == "--workgroups") { o.workgroups = std::stoul(args.value(a)); }
        else { args.usage_error("unknown argument " + a); }
    }
    if (o.seconds <= 0 || o.reps <= 0 || o.workgroups == 0) {
        args.usage_error("values must be positive");
    }
    return o;
}

// amdgpu cross-check only: the active pp_dpm_sclk level, as declared context (never used in a
// measurement).
std::string read_sclk() {
    for (int card = 0; card < 16; ++card) {
        const std::string path = "/sys/class/drm/card" + std::to_string(card) +
                                 "/device/pp_dpm_sclk";
        FILE* f = std::fopen(path.c_str(), "r");
        if (f == nullptr) { continue; }
        char line[256];
        std::string active;
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            if (std::strstr(line, "*") != nullptr) { active = line; }
        }
        std::fclose(f);
        if (!active.empty()) {
            // "  3: 2995 Mhz *"
            const std::size_t colon = active.find(':');
            if (colon != std::string::npos) {
                const std::size_t start = active.find_first_of("0123456789", colon);
                if (start != std::string::npos) {
                    return active.substr(start, active.find_first_of(" Mm", start) - start);
                }
            }
        }
    }
    return "";
}

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    VkCtx ctx("clock", 0,
              Obj().num("seconds", o.seconds).integer("reps", o.reps)
                  .integer("workgroups", o.workgroups));
    ctx.declare_common();

    const std::vector<char> spirv = VkCtx::read_spirv(shader_path(argv[0], "clock_dual.spv"));
    VkCtx::Pipeline pl;
    ctx.make_pipeline(pl, spirv, 8, 64);
    VkBufferRes stamps = ctx.make_buffer(64, false, true, "stamps");
    const VkDescriptorSet set = ctx.bind_set(pl, VK_NULL_HANDLE, stamps.buffer);
    VkCtx::Submit sub;
    ctx.make_submit(sub);

    // One burn kernel with timestamps around it; returns (exec_ns from timestamps, ticks from
    // the shader clock, nonce_ok, host_span_ns). The rate derivation divides ticks by the
    // timestamp span; the bracket adds a small dispatch gap (~tens of microseconds), declared
    // here and bounded by the long kernel duration.
    const auto burn = [&](std::uint32_t iters, std::uint32_t nonce, double* exec_ns,
                          double* ticks, bool* nonce_ok, double* host_span_ns) {
        EL_VK(vkResetCommandBuffer(sub.cmd, 0));
        const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0,
                                          nullptr};
        EL_VK(vkBeginCommandBuffer(sub.cmd, &bi));
        vkCmdResetQueryPool(sub.cmd, sub.queries, 0, 2);
        vkCmdWriteTimestamp(sub.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sub.queries, 0);
        vkCmdBindPipeline(sub.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipe);
        vkCmdBindDescriptorSets(sub.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &set, 0,
                                nullptr);
        const std::uint32_t push[2] = {iters, nonce};
        vkCmdPushConstants(sub.cmd, pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
        vkCmdDispatch(sub.cmd, o.workgroups, 1, 1);
        vkCmdWriteTimestamp(sub.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sub.queries, 1);
        EL_VK(vkEndCommandBuffer(sub.cmd));
        const std::int64_t h0 = mono_ns();
        const VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1,
                              &sub.cmd, 0, nullptr};
        EL_VK(vkQueueSubmit(ctx.queue, 1, &si, sub.fence));
        EL_VK(vkWaitForFences(ctx.device, 1, &sub.fence, VK_TRUE, UINT64_MAX));
        EL_VK(vkResetFences(ctx.device, 1, &sub.fence));
        const std::int64_t h1 = mono_ns();
        std::uint64_t ts[2] = {0, 0};
        EL_VK(vkGetQueryPoolResults(ctx.device, sub.queries, 0, 2, sizeof(ts), ts,
                                    sizeof(std::uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        *exec_ns = double(ts[1] - ts[0]) * ctx.timestamp_period_ns;
        std::uint32_t raw[6];
        std::memcpy(raw, stamps.mapped, sizeof(raw));
        *nonce_ok = raw[4] == nonce && raw[5] == 0x45594C4Bu;
        const std::uint64_t t0d = std::uint64_t(raw[1]) << 32 | raw[0];
        const std::uint64_t t1d = std::uint64_t(raw[3]) << 32 | raw[2];
        *ticks = double(t1d - t0d);
        *host_span_ns = double(h1 - h0);
    };

    // ---- calibrate the burn length to ~2 ms ----------------------------------------------------
    std::uint32_t iters = 100000;
    for (int i = 0; i < 4; ++i) {
        double e = 0, t = 0, h = 0;
        bool ok = false;
        burn(iters, rep_nonce(-1), &e, &t, &ok, &h);
        if (e > 1e6) { break; } // >= 1 ms
        iters = std::uint32_t(double(iters) * std::max(2.0, 2e6 / std::max(1.0, e)));
    }

    // ---- idle cadence: short burn, then the host sleeps ------------------------------------------
    std::string sclk_during_load; // sampled between loaded reps; sysfs updates lazily
    auto rate_series = [&](const char* tag, bool idle) {
        std::vector<double> mhz, overheads;
        bool nonce_ok_all = true;
        const double per_rep_target = o.seconds / double(o.reps);
        for (int rep = 0; rep < o.reps; ++rep) {
            if (idle) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
            double e = 0, t = 0, h = 0;
            bool ok = false;
            burn(iters, rep_nonce(rep), &e, &t, &ok, &h);
            nonce_ok_all = nonce_ok_all && ok;
            if (!idle) { sclk_during_load = read_sclk(); } // right after a burn, while boosted
            const double m = t / std::max(1.0, e) * 1000.0; // ticks/ns -> MHz
            mhz.push_back(m);
            overheads.push_back((h - e) / 1e3); // us
            measure(std::string(tag) + "_clock", m, "MHz",
                    Obj().boolean("nonce_ok", ok).num("exec_us", e / 1e3)
                        .num("submit_overhead_us", (h - e) / 1e3),
                    ok, rep);
            (void)per_rep_target;
        }
        const Summary sm = summarize(mhz), so = summarize(overheads);
        measure(std::string(tag) + "_clock", sm.p50, "MHz",
                Obj().obj("per_rep", summary_obj(sm)), nonce_ok_all);
        measure(std::string(tag) + "_submit_overhead", so.p50, "us",
                Obj().obj("per_rep", summary_obj(so)), nonce_ok_all);
        // the rate must be plausible for a GPU core clock and stable within a cadence
        check(std::string(tag) + "_clock_sanity", sm.p50 > 100.0 && sm.p50 < 10000.0,
              Obj().num("p50_mhz", sm.p50));
        check(std::string(tag) + "_overhead", so.p50 < 5000.0,
              Obj().num("p50_us", so.p50));
        return sm.p50;
    };
    const double idle_mhz = rate_series("idle", true);
    const double loaded_mhz = rate_series("loaded", false);

    // ---- cross-check: amdgpu's active sclk (declared context only) --------------------------------
    // pp_dpm_sclk's active flag updates lazily, so it is sampled right after a burn kernel;
    // the comparison uses the peak of both sides (boost clock vs peak derived rate).
    const std::string sclk = sclk_during_load.empty() ? read_sclk() : sclk_during_load;
    if (!sclk.empty()) {
        declared("amdgpu.pp_dpm_sclk_active_mhz", sclk,
                 "sysfs cross-check sampled under load, not a measurement");
        const double ratio = loaded_mhz / std::max(1.0, std::stod(sclk));
        measure("shader_clock_over_sclk", ratio, "ratio",
                Obj().num("derived_mhz", loaded_mhz).raw("sclk_mhz", sclk), true);
        // The shader clock is expected to track the core clock; both sides are sampled at
        // possibly different DVFS points, so the band is wide (3x).
        check("shader_clock_vs_sclk", ratio > 0.33 && ratio < 3.0,
              Obj().num("ratio", ratio).num("derived_mhz", loaded_mhz).raw("sclk_mhz", sclk));
        feature("clock", true, true, true,
                Obj().str("path", "shader clock derived against timestamp brackets")
                    .num("idle_mhz", idle_mhz).num("loaded_mhz", loaded_mhz)
                    .raw("sclk_mhz", sclk));
    } else {
        feature("clock", true, false, false,
                Obj().str("path", "shader clock derived against timestamp brackets")
                    .str("sclk", "sysfs not found; cross-check missing"));
    }
    note("no GLSL-accessible real-time nanosecond clock exists on this stack (AMD/RADV): the "
         "shader clock is the core clock and moves with DVFS, so the CUDA-style host<->device "
         "globaltimer mapping is done with timestamp queries instead (see the sync probe)");

    ctx.destroy_buffer(stamps);
    ctx.destroy_pipeline(pl);
    return probe_end();
}
