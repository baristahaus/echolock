// memory (Vulkan backend): what the GPU can read from each kind of Vulkan memory, and what a
// copy costs, timed with GPU timestamp queries and checked with the XOR oracle.
//
// Same record contract as the CUDA memory probe: declared values first, then one measure per
// dispatch plus a summary, with `verified` true only when the dispatch's result matched the
// expectation (the shader folds a per-dispatch nonce into its result slot, so an unwritten slot
// cannot pass). Copies are verified by a read dispatch of the destination.
//
// Memory kinds: a device-local buffer (VRAM on a discrete GPU) and a host-visible buffer (system
// memory). On an integrated GPU every heap can be both; the heap flags are recorded either way,
// so the report says which case it was. CPU/GPU contention: host threads read the host-visible
// buffer while the GPU streams the device-local one, and both sides are measured.
//
// Device setup, buffers (HOST_CACHED preference for CPU-streamed memory), the pipeline and the
// timestamped submissions come from common/vk_ctx.h, the same way the other probes use it.
//
// Build: make -C backends/vulkan (glslc for the shader, g++ for the probe, -lvulkan).

#include "common/vk_ctx.h"

#include <vulkan/vulkan.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>

using namespace echolock;

namespace {

struct Options {
    std::size_t mib      = 256;
    double seconds       = 0.5;
    int device           = 0;
    unsigned cpu_threads = std::max(1u, hw_threads() / 2);
    unsigned workgroups  = 512;
};

Options parse(int argc, char** argv) {
    Args args(argc, argv,
              "usage: memory [--mib N (256)] [--seconds S (0.5)] [--device N (0)] "
              "[--cpu-threads N (half)] [--workgroups N (512)]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--mib") { o.mib = std::size_t(std::stoul(args.value(a))); }
        else if (a == "--seconds") { o.seconds = std::stod(args.value(a)); }
        else if (a == "--device") { o.device = std::stoi(args.value(a)); }
        else if (a == "--cpu-threads") { o.cpu_threads = std::stoul(args.value(a)); }
        else if (a == "--workgroups") { o.workgroups = std::stoul(args.value(a)); }
        else { args.usage_error("unknown argument " + a); }
    }
    if (o.mib == 0 || o.seconds <= 0 || o.workgroups == 0) {
        args.usage_error("values must be positive");
    }
    return o;
}

std::string num_json(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.10g", v);
    return buf;
}

// ---- command submission --------------------------------------------------------------------------

struct Push {
    std::uint32_t words;
    std::uint32_t nonce;
};

struct SeriesResult {
    double gbps = 0;
    bool verified = false;
};

// Warm-up, then `reps` timed verified reads: one measure per rep, a summary, one check.
// `read(set, nonce)` is one verified dispatch on `set` (the shared machinery fills the slot,
// barriers transfer writes into shader visibility and brackets the dispatch with timestamps).
SeriesResult read_series(
    const std::function<std::pair<double, std::uint32_t>(VkDescriptorSet, std::uint32_t)>& read,
    VkDescriptorSet set, std::uint32_t words, int reps, std::uint32_t expected, const Obj& tags,
    const std::string& name) {
    read(set, rep_nonce(-1)); // warm-up
    std::vector<double> per;
    int bad = 0;
    for (int r = 0; r < reps; ++r) {
        const unsigned nonce = rep_nonce(r);
        const auto [gpu_us, slot] = read(set, nonce);
        const bool ok = slot == fold_nonce(expected, nonce);
        bad += ok ? 0 : 1;
        const double gbps = double(words) * 4.0 / (gpu_us * 1e-6) / 1e9;
        per.push_back(gbps);
        measure(name, gbps, "GB/s", Obj(tags).num("gpu_us", gpu_us), ok, r);
    }
    const Summary s = summarize(per);
    const bool ok = bad == 0;
    measure(name, s.p50, "GB/s", Obj(tags).obj("per_rep", summary_obj(s)), ok);
    check(name + ".xor", ok,
          Obj().obj("tags", tags).integer("reps", reps).integer("bad_reps", bad)
              .uint("expected_xor", expected));
    return {s.p50, ok};
}

// ---- CPU side ------------------------------------------------------------------------------------

struct CpuRead {
    double gbps = 0;
    std::uint64_t passes = 0, bad_passes = 0;
};

// Each thread reads its slice repeatedly until `stop`, checking every pass against the slice
// XOR taken in an untimed first pass -- the same contract as the CUDA probe's cpu_read.
CpuRead cpu_read(const void* ptr, std::size_t bytes, unsigned threads,
                 const std::atomic<bool>* stop) {
    const auto* p = static_cast<const std::uint32_t*>(ptr);
    const std::uint64_t words = bytes / 4, per = (words + threads - 1) / threads;
    std::vector<std::uint32_t> expected(threads, 0);
    {
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                std::uint32_t a = 0;
                for (std::uint64_t i = std::uint64_t(t) * per,
                                    end = std::min(words, std::uint64_t(t + 1) * per);
                     i < end; ++i) {
                    a ^= p[i];
                }
                expected[t] = a;
            });
        }
        for (auto& th : pool) { th.join(); }
    }
    std::atomic<std::uint64_t> total{0}, passes{0}, bad{0};
    const std::int64_t t0 = mono_ns();
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            const std::uint64_t beg = std::uint64_t(t) * per;
            const std::uint64_t end = std::min(words, std::uint64_t(t + 1) * per);
            std::uint64_t done = 0, n = 0, wrong = 0;
            while (!stop->load(std::memory_order_relaxed)) {
                std::uint32_t a = 0;
                for (std::uint64_t i = beg; i < end; ++i) { a ^= p[i]; }
                wrong += a != expected[t] ? 1 : 0;
                done += (end - beg) * 4;
                ++n;
            }
            total += done;
            passes += n;
            bad += wrong;
        });
    }
    for (auto& th : pool) { th.join(); }
    const double s = double(mono_ns() - t0) / 1e9;
    return {double(total.load()) / s / 1e9, passes.load(), bad.load()};
}

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    VkCtx vk("memory", o.device,
             Obj().uint("bytes", std::uint64_t(o.mib) << 20).num("seconds", o.seconds)
                 .integer("cpu_threads", o.cpu_threads).integer("workgroups", o.workgroups)
                 .integer("device", o.device));
    vk.declare_common();
    // The probe dispatches o.workgroups groups; declare the device limit it is checked against.
    declared("vulkan.max_compute_workgroups",
             "[" + std::to_string(vk.props.limits.maxComputeWorkGroupCount[0]) + ", " +
                 std::to_string(vk.props.limits.maxComputeWorkGroupCount[1]) + ", " +
                 std::to_string(vk.props.limits.maxComputeWorkGroupCount[2]) + "]",
             "vkGetPhysicalDeviceProperties");

    // Shader lives next to the binary; the Makefile puts it there.
    const std::vector<char> spirv = VkCtx::read_spirv(shader_path(argv[0], "read_fold.spv"));

    // Pipeline + descriptor machinery.
    VkCtx::Pipeline pl;
    vk.make_pipeline(pl, spirv, sizeof(Push));

    const std::size_t bytes = std::uint64_t(o.mib) << 20;
    const std::uint32_t words = std::uint32_t(bytes / 4);
    VkBufferRes host_buf = vk.make_buffer(bytes, false, true, "host-visible");
    VkBufferRes dev_buf = vk.make_buffer(bytes, true, false, "device-local");
    VkBufferRes slot_buf = vk.make_buffer(4, false, true, "result-slot");

    // Fill the host buffer with the pattern; its XOR is the oracle for everything downstream.
    {
        auto* p = static_cast<std::uint32_t*>(host_buf.mapped);
        for (std::uint32_t i = 0; i < words; ++i) { p[i] = pattern_word(i); }
    }
    const std::uint32_t host_expected = expected_xor(words);

    VkCtx::Submit sub;
    vk.make_submit(sub);

    // Descriptor sets: one binding the device-local buffer, one the host-visible buffer.
    VkDescriptorSet dev_set = vk.bind_set(pl, dev_buf.buffer, slot_buf.buffer);
    VkDescriptorSet host_set = vk.bind_set(pl, host_buf.buffer, slot_buf.buffer);

    // Read dispatch with the shared machinery (slot fill, barrier, timestamps, fence, slot read).
    const auto read = [&](VkDescriptorSet set, std::uint32_t nonce) {
        Push push{words, nonce};
        return sub.dispatch(pl, slot_buf, [&](VkCommandBuffer cmd, const VkCtx::Pipeline& p) {
            vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0,
                                    nullptr);
            vkCmdDispatch(cmd, o.workgroups, 1, 1);
        });
    };

    // Fill the device-local buffer by a copy, then verify it with a read dispatch: the copy's
    // payload is checked, not assumed.
    {
        const double fill_us = sub.copy(host_buf, dev_buf, bytes);
        note("device-local buffer filled by copy in " + num_json(fill_us) + " us");
        const auto [gpu_us, slot] = read(dev_set, rep_nonce(-2));
        check("dev_fill.xor", slot == fold_nonce(host_expected, rep_nonce(-2)),
              Obj().num("gpu_us", gpu_us).uint("got", slot)
                  .uint("expected", fold_nonce(host_expected, rep_nonce(-2))));
    }

    // Size the series from one warm dispatch, like the CUDA probe does (gpu_us is in
    // microseconds; a series targets `seconds` of dispatches).
    int reps = 3;
    {
        const auto [gpu_us, slot] = read(dev_set, rep_nonce(-1));
        reps = std::clamp(int(o.seconds * 1e6 / std::max(gpu_us, 0.001)), 3, 1024);
        (void)slot;
    }

    // 1. GPU reads of device-local memory.
    const SeriesResult dev_alone =
        read_series(read, dev_set, words, reps, host_expected,
                    Obj().str("source", "vulkan_device_local")
                        .boolean("heap_device_local", dev_buf.device_local)
                        .boolean("heap_host_visible", dev_buf.host_visible)
                        .str("contention", "none"),
                    "gpu_read");

    // 2. GPU reads of host-visible memory.
    read_series(read, host_set, words, reps, host_expected,
                Obj().str("source", "vulkan_host_visible")
                    .boolean("heap_device_local", host_buf.device_local)
                    .boolean("heap_host_visible", host_buf.host_visible)
                    .str("contention", "none"),
                "gpu_read");

    // 3. Copies host-visible -> device-local, each verified by a read dispatch of the
    // destination (the barrier in the read makes the copy visible to the shader).
    {
        const int copy_reps = std::clamp(int(o.seconds * 1000.0 / 2.0), 3, 64);
        std::vector<double> per;
        int bad = 0;
        for (int r = 0; r < copy_reps; ++r) {
            const double copy_us = sub.copy(host_buf, dev_buf, bytes);
            const unsigned nonce = rep_nonce(r);
            const auto [gpu_us, slot] = read(dev_set, nonce);
            const bool ok = slot == fold_nonce(host_expected, nonce);
            bad += ok ? 0 : 1;
            const double gbps = double(bytes) / (copy_us * 1e-6) / 1e9;
            per.push_back(gbps);
            measure("copy_payload", gbps, "GB/s",
                    Obj().str("source", "vulkan_host_visible").str("direction", "h2d")
                        .num("copy_us", copy_us).num("verify_read_us", gpu_us),
                    ok, r);
        }
        const Summary s = summarize(per);
        measure("copy_payload", s.p50, "GB/s",
                Obj().str("source", "vulkan_host_visible").str("direction", "h2d")
                    .obj("per_rep", summary_obj(s)),
                bad == 0);
        check("copy_payload.xor", bad == 0,
              Obj().integer("reps", copy_reps).integer("bad_copies", bad)
                  .uint("expected_xor", host_expected));
    }

    // 4. CPU/GPU contention: host threads stream the host-visible buffer while the GPU streams
    // the device-local one; both sides are measured, and each CPU pass is checked.
    std::atomic<bool> stop{false};
    CpuRead cpu_shared;
    std::thread cpu_side([&] {
        cpu_shared = cpu_read(host_buf.mapped, bytes, o.cpu_threads, &stop);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // CPU-only lead-in
    const SeriesResult dev_contended =
        read_series(read, dev_set, words, reps, host_expected,
                    Obj().str("source", "vulkan_device_local").str("contention", "cpu_read"),
                    "gpu_read");
    stop = true;
    cpu_side.join();
    {
        const Obj cpu_tags = Obj().str("source", "vulkan_host_visible")
                                 .integer("threads", o.cpu_threads);
        const bool cpu_ok = cpu_shared.bad_passes == 0 && cpu_shared.passes > 0;
        measure("cpu_read", cpu_shared.gbps, "GB/s",
                Obj(cpu_tags).str("contention", "gpu_read")
                    .uint("passes", cpu_shared.passes)
                    .uint("bad_passes", cpu_shared.bad_passes),
                cpu_ok);
        check("cpu_read.xor", cpu_ok,
              Obj(cpu_tags).uint("passes", cpu_shared.passes)
                  .uint("bad_passes", cpu_shared.bad_passes));
        measure("cpu_read", cpu_shared.gbps, "GB/s", Obj(cpu_tags).str("contention", "gpu_read"),
                cpu_ok);
        measure("shared_sum", dev_contended.gbps + cpu_shared.gbps, "GB/s",
                Obj().str("note", "GPU read + CPU read while contending"), true);
        if (dev_alone.gbps > 0) {
            measure("gpu_read_contention_ratio", dev_contended.gbps / dev_alone.gbps, "x",
                    Obj().num("alone_gbps", dev_alone.gbps)
                        .num("contended_gbps", dev_contended.gbps),
                    dev_contended.verified && dev_alone.verified);
        }
    }

    vk.destroy_pipeline(pl);
    vk.destroy_submit(sub);
    vk.destroy_buffer(slot_buf);
    vk.destroy_buffer(dev_buf);
    vk.destroy_buffer(host_buf);
    return probe_end();
}
