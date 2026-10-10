// sync (Vulkan backend): the open equivalent of the CUDA sync probe. Spec: probes/sync/spec.md.
//
// Per rep: t_submit (host CLOCK_MONOTONIC), one submission that brackets a device-side spin
// kernel with GPU timestamp queries, vkWaitForFences, t_return (host).
//
// Device timeline: the timestamp queries are the device's real-nanosecond timeline
// (timestampPeriod = 10 ns here, validated by the memory probe against the wall clock). The
// kernel's own GLSL clock (clock2x32ARB) is recorded too, but on AMD/RADV it is NOT a
// real-time clock -- it ticks at the shader-clock rate (the clock probe derives the rate), so
// it cannot anchor a host mapping the way NVIDIA's %globaltimer does. That asymmetry is
// itself a capability finding: the CUDA sync probe maps host time to device time with an
// in-kernel globaltimer stamp; the Vulkan equivalent must lean on timestamp queries.
//
// Measures per duration: the device-side execution span (timestamps), the host-side span
// (submit to fence return), and lag = t_return - t_exec_end: how long the host was still
// waiting after the device had finished. CUDA's SCHEDULE axis has no Vulkan equivalent: there
// is no device-side schedule flag on a queue submission, so this probe measures the one
// behavior. The pipeline underneath is open: RADV -> amdgpu DRM scheduler (kernel uAPI).

#include "common/vk_ctx.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace echolock;

namespace {

struct Options {
    std::vector<double> durations_us = {50, 1000, 10000, 60000};
    int iterations = 30;
};

Options parse(int argc, char** argv) {
    Args args(argc, argv,
              "usage: sync [--durations-us 50,1000,10000,60000] [--iterations N (30)]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--durations-us") {
            o.durations_us.clear();
            const std::string v = args.value(a);
            for (std::size_t pos = 0; pos < v.size();) {
                const std::size_t comma = v.find(',', pos);
                o.durations_us.push_back(std::stod(v.substr(pos, comma - pos)));
                if (comma == std::string::npos) { break; }
                pos = comma + 1;
            }
        } else if (a == "--iterations") { o.iterations = std::stoi(args.value(a)); }
        else { args.usage_error("unknown argument " + a); }
    }
    if (o.iterations <= 0) { args.usage_error("iterations must be positive"); }
    for (const double d : o.durations_us) {
        if (d <= 0) { args.usage_error("durations must be positive"); }
    }
    return o;
}

// summaries recorded in nanoseconds are converted for the us-tagged objects
Summary apply_us(const Summary& s) {
    return Summary{s.min / 1e3, s.p50 / 1e3, s.p90 / 1e3, s.max / 1e3, s.mean / 1e3};
}

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    VkCtx ctx("sync", 0,
              Obj().str("durations_us",
                        [&] {
                            std::string s;
                            for (const double d : o.durations_us) {
                                s += (s.empty() ? "" : ",") + std::to_string(long(d));
                            }
                            return s;
                        }())
                  .integer("iterations", o.iterations));
    ctx.declare_common();

    // The spin kernel loops on the GLSL shader clock; its duration is therefore in shader-clock
    // ticks, not nanoseconds -- the timestamp brackets turn it into a controlled real-time
    // device workload whose length we can verify against the requested duration.
    const std::vector<char> spirv = VkCtx::read_spirv(shader_path(argv[0], "spin_until.spv"));
    VkCtx::Pipeline pl;
    ctx.make_pipeline(pl, spirv, 16, 64); // full subgroups
    VkBufferRes stamps = ctx.make_buffer(64, false, true, "stamps");
    const VkDescriptorSet set = ctx.bind_set(pl, VK_NULL_HANDLE, stamps.buffer);
    VkCtx::Submit sub;
    ctx.make_submit(sub);

    // The GLSL shader clock's tick rate is not known a priori (and moves with DVFS), so the
    // spin length is calibrated per duration: a short calibration spin measured by the
    // timestamp bracket gives ticks/ns right before the series, and the realized span is then
    // verified by the brackets of every rep.
    const auto spin_exec_ns = [&](std::uint64_t ticks, std::uint32_t nonce, double* exec_ns_out) {
        EL_VK(vkResetCommandBuffer(sub.cmd, 0));
        const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0,
                                          nullptr};
        EL_VK(vkBeginCommandBuffer(sub.cmd, &bi));
        vkCmdResetQueryPool(sub.cmd, sub.queries, 0, 2);
        vkCmdWriteTimestamp(sub.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sub.queries, 0);
        vkCmdBindPipeline(sub.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipe);
        vkCmdBindDescriptorSets(sub.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &set, 0,
                                nullptr);
        const std::uint32_t push[4] = {std::uint32_t(ticks & 0xFFFFFFFFu),
                                       std::uint32_t(ticks >> 32), nonce, 0u};
        vkCmdPushConstants(sub.cmd, pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
        vkCmdDispatch(sub.cmd, 1, 1, 1);
        vkCmdWriteTimestamp(sub.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sub.queries, 1);
        EL_VK(vkEndCommandBuffer(sub.cmd));
        const VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1,
                              &sub.cmd, 0, nullptr};
        EL_VK(vkQueueSubmit(ctx.queue, 1, &si, sub.fence));
        EL_VK(vkWaitForFences(ctx.device, 1, &sub.fence, VK_TRUE, UINT64_MAX));
        EL_VK(vkResetFences(ctx.device, 1, &sub.fence));
        std::uint64_t ts[2] = {0, 0};
        EL_VK(vkGetQueryPoolResults(ctx.device, sub.queries, 0, 2, sizeof(ts), ts,
                                    sizeof(std::uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        *exec_ns_out = double(ts[1] - ts[0]) * ctx.timestamp_period_ns;
    };

    for (const double dur_us : o.durations_us) {
        const std::string tag = "sync_" + std::to_string(long(dur_us)) + "us";
        // calibrate: shader-clock ticks per nanosecond, measured by the timestamp bracket
        double cal_exec_ns = 1.0;
        const std::uint64_t cal_ticks = 300000000ull; // ~0.1 s at GHz-ish rates
        spin_exec_ns(cal_ticks, rep_nonce(-1), &cal_exec_ns);
        const double ticks_per_ns = double(cal_ticks) / std::max(1.0, cal_exec_ns);
        note("calibration: " + Obj().uint("cal_ticks", cal_ticks).num("cal_exec_us", cal_exec_ns / 1e3)
                                 .num("ticks_per_ns", ticks_per_ns)
                                 .uint("spin_ticks_50us", std::uint64_t(50.0 * 1000.0 * ticks_per_ns))
                                 .text());
        const std::uint64_t spin_ticks = std::uint64_t(dur_us * 1000.0 * ticks_per_ns);
        std::vector<double> host_spans, exec_spans, lags;
        bool nonce_ok_all = true;
        for (int rep = 0; rep < o.iterations; ++rep) {
            const std::uint32_t nonce = rep_nonce(rep);
            const std::uint32_t push[4] = {std::uint32_t(spin_ticks & 0xFFFFFFFFu),
                                           std::uint32_t(spin_ticks >> 32), nonce, 0u};
            EL_VK(vkResetCommandBuffer(sub.cmd, 0));
            const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                              0, nullptr};
            EL_VK(vkBeginCommandBuffer(sub.cmd, &bi));
            vkCmdResetQueryPool(sub.cmd, sub.queries, 0, 2);
            vkCmdWriteTimestamp(sub.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sub.queries, 0);
            vkCmdBindPipeline(sub.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipe);
            vkCmdBindDescriptorSets(sub.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &set,
                                    0, nullptr);
            vkCmdPushConstants(sub.cmd, pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
            vkCmdDispatch(sub.cmd, 1, 1, 1);
            vkCmdWriteTimestamp(sub.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sub.queries, 1);
            EL_VK(vkEndCommandBuffer(sub.cmd));
            const std::int64_t t_submit = mono_ns();
            const VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1,
                                  &sub.cmd, 0, nullptr};
            EL_VK(vkQueueSubmit(ctx.queue, 1, &si, sub.fence));
            EL_VK(vkWaitForFences(ctx.device, 1, &sub.fence, VK_TRUE, UINT64_MAX));
            EL_VK(vkResetFences(ctx.device, 1, &sub.fence));
            const std::int64_t t_return = mono_ns();

            std::uint64_t ts[2] = {0, 0};
            EL_VK(vkGetQueryPoolResults(ctx.device, sub.queries, 0, 2, sizeof(ts), ts,
                                        sizeof(std::uint64_t),
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            const double exec_ns = double(ts[1] - ts[0]) * ctx.timestamp_period_ns;
            std::uint32_t raw[6];
            std::memcpy(raw, stamps.mapped, sizeof(raw));
            const bool nonce_ok = raw[4] == nonce && raw[5] == 0x45594C4Bu;
            nonce_ok_all = nonce_ok_all && nonce_ok;
            const double host_span_ns = double(t_return - t_submit);
            // lag: host time spent waiting after the device had already finished (negative when
            // the fence woke early relative to the recorded end stamp)
            const double lag_ns = host_span_ns - exec_ns;
            exec_spans.push_back(exec_ns);
            host_spans.push_back(host_span_ns);
            lags.push_back(lag_ns);
            measure(tag + ".lag", lag_ns / 1e3, "us",
                    Obj().boolean("nonce_ok", nonce_ok).num("exec_us", exec_ns / 1e3), nonce_ok,
                    rep);
        }
        const Summary sl = summarize(lags), se = summarize(exec_spans), sh = summarize(host_spans);
        measure(tag + ".lag", sl.p50 / 1e3, "us",
                Obj().obj("per_rep", summary_obj(sl))
                    .obj("exec_span_us", summary_obj(apply_us(se)))
                    .obj("host_span_us", summary_obj(apply_us(sh))),
                nonce_ok_all);
        // The realized device span must track the requested duration. The tolerance is loose on
        // purpose: the spin length was calibrated once against the shader clock, whose rate
        // moves with DVFS (observed: a 3x stretch within one series as the core clock drops
        // after a burst). This is a check on the workload, not on the driver.
        check(tag + ".exec_span", std::abs(se.p50 - dur_us * 1000.0) < dur_us * 1000.0 * 3.0,
              Obj().num("p50_exec_us", se.p50 / 1e3).num("requested_us", dur_us));
        // The host cannot return before the device finished minus the timestamp/stamp slop.
        check(tag + ".clock_mapping", sl.p50 > -2e6,
              Obj().num("p50_lag_us", sl.p50 / 1e3));
        feature("sync", true, true, true,
                Obj().str("path", "spin kernel + timestamp brackets + fence wait on host")
                    .str("scheduler_note", "RADV -> amdgpu DRM scheduler (open pipeline)")
                    .str("device_clock_note",
                         "shader clock is not real time on AMD; timestamps are the device ns "
                         "timeline (unlike CUDA %globaltimer)")
                    .num((tag + "_lag_p50_us").c_str(), sl.p50 / 1e3));
    }

    ctx.destroy_buffer(stamps);
    ctx.destroy_pipeline(pl);
    return probe_end();
}
