// pointer_chase (Vulkan backend): memory latency as a dependent-load chain sees it -- one load
// per hop, every hop depending on the previous -- through each kind of Vulkan memory, at
// footprints that fit and overflow the cache. Spec: probes/pointer_chase/spec.md.
//
// The chain is data (an array of next-indices), so nothing can be prefetched or hoisted. The
// host walks the same chain to know the visited XOR and the pages the walk touches, which is the
// oracle: the shader folds the visited XOR plus a dispatch nonce into its result slot, and an
// unwritten slot cannot match.
//
// File-backed pages have no Vulkan equivalent (buffers are not page cache), so the cold-fault
// measurement of the CUDA backend stays CUDA-side on GB10; everything else the spec asks of this
// probe works here: footprints, seq (page stride) and random (one uniform cycle) patterns, and
// device-local vs host-visible memory.
//
// Device setup, buffers (HOST_CACHED preference first, as the chain buffers and the result slot
// are host-written/read), the pipeline and the timestamped submissions come from common/vk_ctx.h,
// the same way the other probes use it.

#include "common/vk_ctx.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstring>

using namespace echolock;

namespace {

constexpr std::uint64_t kSeed = 0x9E3779B97F4A7C15ull;
constexpr unsigned kPageBytes  = 4096;

struct Options {
    std::vector<double> footprints_mb = {0.0625, 1, 64};
    unsigned hops        = 4096;
    int reps             = 5;
    std::string kinds    = "device,host";
    std::string patterns = "seq,random";
    int device           = 0;
};

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
              "[--kinds device,host] [--patterns seq,random] [--device N (0)]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--footprints") {
            o.footprints_mb.clear();
            for (const auto& v : split_list(args.value(a))) { o.footprints_mb.push_back(std::stod(v)); }
        } else if (a == "--hops") { o.hops = std::stoul(args.value(a)); }
        else if (a == "--reps") { o.reps = std::stoi(args.value(a)); }
        else if (a == "--kinds") { o.kinds = args.value(a); }
        else if (a == "--patterns") { o.patterns = args.value(a); }
        else if (a == "--device") { o.device = std::stoi(args.value(a)); }
        else { args.usage_error("unknown argument " + a); }
    }
    if (o.footprints_mb.empty() || o.hops < 16 || o.reps < 1) {
        args.usage_error("need footprints, hops >= 16, reps >= 1");
    }
    return o;
}

// seq: hop distance one 4 KiB page (wrapped). random: one uniform cycle over all nodes.
std::vector<std::uint32_t> build_chain(std::size_t n, const std::string& pattern) {
    std::vector<std::uint32_t> chain(n);
    if (pattern == "seq") {
        const std::size_t stride = std::max<std::size_t>(1, std::min<std::size_t>(kPageBytes / 4, n / 2));
        for (std::size_t i = 0; i < n; ++i) { chain[i] = std::uint32_t((i + stride) % n); }
        return chain;
    }
    std::vector<std::uint32_t> perm(n);
    for (std::size_t i = 0; i < n; ++i) { perm[i] = std::uint32_t(i); }
    std::uint64_t z = kSeed;
    for (std::size_t i = n - 1; i > 0; --i) {
        z += 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        const std::size_t j = std::size_t((z ^ (z >> 27)) % (i + 1));
        std::swap(perm[i], perm[j]);
    }
    for (std::size_t i = 0; i < n; ++i) { chain[perm[i]] = perm[(i + 1) % n]; }
    return chain;
}

// The host walks the same chain the shader will: expected visited XOR and final index.
struct WalkCheck {
    std::uint32_t acc = 0;
    std::uint32_t final_idx = 0;
};

WalkCheck simulate_walk(const std::vector<std::uint32_t>& chain, unsigned hops) {
    WalkCheck w;
    std::uint32_t i = 0;
    for (unsigned h = 0; h < hops; ++h) {
        const std::uint32_t next = chain[i];
        w.acc ^= next;
        i = next;
    }
    w.final_idx = i;
    return w;
}

struct Push {
    std::uint32_t hops;
    std::uint32_t nonce;
};

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    VkCtx vk("pointer_chase", o.device,
             Obj().integer("hops", o.hops).integer("reps", o.reps).str("kinds", o.kinds)
                 .str("patterns", o.patterns).integer("device", o.device));
    vk.declare_common();

    // Shader + pipeline: chain buffer + result slot, hops/nonce push constants.
    const std::vector<char> spirv = VkCtx::read_spirv(shader_path(argv[0], "chase.spv"));
    VkCtx::Pipeline pl;
    vk.make_pipeline(pl, spirv, sizeof(Push));

    VkCtx::Submit sub;
    vk.make_submit(sub);

    // The slot: host-visible + cached (the host reads it after every dispatch).
    VkBufferRes slot_buf = vk.make_buffer(4, false, true, "result-slot");

    // One verified chase dispatch: the shared machinery fills the slot fresh, barriers transfer
    // writes into shader visibility, brackets the dispatch with timestamps and reads the slot.
    const auto chase = [&](VkDescriptorSet set, std::uint32_t nonce) {
        Push push{o.hops, nonce};
        return sub.dispatch(pl, slot_buf, [&](VkCommandBuffer cmd, const VkCtx::Pipeline& p) {
            vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0,
                                    nullptr);
            vkCmdDispatch(cmd, 1, 1, 1); // one invocation: the walk is serial by design
        });
    };

    // Footprints x patterns x kinds, one series each: measures per rep, a summary, one check.
    for (const double mb : o.footprints_mb) {
        const std::size_t chain_bytes = std::size_t(mb * (1ull << 20));
        const std::size_t n = chain_bytes / 4;
        if (n < 2) { fail("footprint too small for a chain"); }
        for (const std::string& pattern : split_list(o.patterns)) {
            const std::vector<std::uint32_t> chain = build_chain(n, pattern);
            const WalkCheck oracle = simulate_walk(chain, o.hops);
            for (const std::string& kind : split_list(o.kinds)) {
                // The host fills and reads the chain buffer through its mapping, so keep the
                // HOST_CACHED-first type selection the probe has always used for it.
                VkBufferRes buf = vk.make_buffer(chain_bytes, kind == "device", true, kind);
                std::memcpy(buf.mapped, chain.data(), chain_bytes);
                VkDescriptorSet set = vk.bind_set(pl, buf.buffer, slot_buf.buffer);

                // Warm-up, then reps of timed verified walks.
                chase(set, rep_nonce(-1));
                std::vector<double> per;
                int bad = 0;
                for (int r = 0; r < o.reps; ++r) {
                    const unsigned nonce = rep_nonce(r);
                    const auto [gpu_us, slot] = chase(set, nonce);
                    const bool ok = slot == fold_nonce(oracle.acc, nonce);
                    bad += ok ? 0 : 1;
                    const double hop_ns = gpu_us * 1e3 / o.hops;
                    per.push_back(hop_ns);
                    measure("chase_hop_latency", hop_ns, "ns/hop",
                            Obj().str("kind", kind == "device" ? "vulkan_device_local"
                                                               : "vulkan_host_visible")
                                .num("footprint_mib", mb)
                                .uint("footprint_bytes", chain_bytes)
                                .str("pattern", pattern)
                                .uint("hops", o.hops),
                            ok, r);
                }
                const Summary s = summarize(per);
                measure("chase_hop_latency", s.p50, "ns/hop",
                        Obj().str("kind", kind == "device" ? "vulkan_device_local"
                                                           : "vulkan_host_visible")
                            .num("footprint_mib", mb)
                            .str("pattern", pattern)
                            .obj("per_rep", summary_obj(s)),
                        bad == 0);
                check("chase_hop_latency.walk_xor", bad == 0,
                      Obj().str("kind", kind).str("pattern", pattern)
                          .num("footprint_mib", mb)
                          .integer("reps", o.reps)
                          .integer("bad_walks", bad)
                          .uint("expected_acc", oracle.acc)
                          .uint("expected_final", oracle.final_idx));

                vkFreeDescriptorSets(vk.device, pl.pool, 1, &set);
                vk.destroy_buffer(buf);
            }
        }
    }

    vk.destroy_pipeline(pl);
    vk.destroy_submit(sub);
    vk.destroy_buffer(slot_buf);
    return probe_end();
}
