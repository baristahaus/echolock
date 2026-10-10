// mma (Vulkan backend): AMD tensor-core throughput per math format through
// VK_KHR_cooperative_matrix -- the RDNA WMMA path, the open equivalent of the CUDA mma probe's
// WMMA section. Spec: probes/mma/spec.md.
//
// The driver's supported configurations are queried first
// (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) and declared; each format the probe runs
// must appear in that list, or it is skipped with a note. Every format is verified numerically
// before and during the timed launches: one MulAdd's full 16x16 result against an exact host
// reference, then every workgroup's and every accumulator chain's result in the throughput
// kernel. The result slot carries a per-dispatch nonce, so an unwritten slot cannot match.
//
// On this machine (RADV GFX1201) the driver exposes fp16, bf16, int8 (u8/s8 to s32/u32, with
// saturating variants) and fp8 e4m3/e5m2, all M16 N16 K16 at subgroup scope. fp16, bf16 and
// int8 run here and verify bit-exact against an exact host reference. fp8 is driver-supported
// but the GLSL frontend in this toolchain (glslang 16.6.1) cannot express fp8 cooperative
// matrices, so the probe says so instead of guessing. fp4 exists in the RDNA4 ISA but the
// driver does not expose it at all.
//
// Three traps this probe hit on the way (all would look like "the driver is broken" without
// the per-result oracle):
//  1. GLSL coopMatMulAdd(x, y, z) RETURNS the new accumulator; z is not modified in place
//     (CUDA's mma_sync semantics do not carry over). Discarding the return value makes the
//     compiler dead-code-eliminate the entire multiply and store the untouched zero
//     initializer -- silent wrong answers with no error anywhere.
//  2. B loaded column-major reads element (k,n) at n*16+k; a row-major fill is silently
//     transposed. The shared pattern hides this for some columns, which is why every result
//     must be checked, not just one.
//  3. Signed int8 operands require the CooperativeMatrixOperands signed bits in SPIR-V;
//     without them the spec treats the operands as unsigned. glslang only emits them with
//     --target-env=vulkan1.4 or later (the Makefile compiles these shaders with it).

#include "common/vk_ctx.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace echolock;

namespace {

constexpr int kChains = 8;
constexpr std::uint32_t kM = 16, kN = 16, kK = 16;

struct Options {
    double seconds = 0.5;
    std::string formats = "fp16,bf16,int8";
    unsigned workgroups = 512;
};

Options parse(int argc, char** argv) {
    Args args(argc, argv,
              "usage: mma [--seconds S (0.5)] [--formats fp16,bf16,int8] [--workgroups N (512)]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--seconds") { o.seconds = std::stod(args.value(a)); }
        else if (a == "--formats") { o.formats = args.value(a); }
        else if (a == "--workgroups") { o.workgroups = std::stoul(args.value(a)); }
        else { args.usage_error("unknown argument " + a); }
    }
    if (o.seconds <= 0 || o.workgroups == 0) { args.usage_error("values must be positive"); }
    return o;
}

bool wants(const Options& o, const std::string& f) {
    return o.formats.find(f) != std::string::npos;
}

// Exact operand patterns (same scheme as the CUDA mma probe): small values, exactly
// representable in every format, so the host reference is exact.
double a_pattern(int m, int k) { return double((m * 13 + k * 7) % 5) - 2.0; } // [-2, 2]
double b_pattern(int k, int n) { return double((k * 11 + n * 5) % 5) - 2.0; } // [-2, 2]

std::vector<double> reference() {
    std::vector<double> c(256, 0.0);
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            double acc = 0.0;
            for (int k = 0; k < 16; ++k) { acc += a_pattern(m, k) * b_pattern(k, n); }
            c[std::size_t(m) * 16 + n] = acc;
        }
    }
    return c;
}

// IEEE fp16 bits for the pattern values (multiples of 0.5 in [-2, 2]: normal, exactly
// representable, so a direct encoding from the float is exact).
std::uint16_t fp16_bits(double v) {
    const float f = float(v);
    if (f == 0.0f) { return 0; }
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const int exp = int((bits >> 23) & 0xFF) - 127;
    const std::uint32_t mant = bits & 0x7FFFFFu;
    return std::uint16_t(sign | std::uint32_t(exp + 15) << 10 | mant >> 13);
}

// bf16 = the top 16 bits of the fp32 value (exact for these values: 8 mantissa bits suffice).
std::uint16_t bf16_bits(double v) {
    const float f = float(v);
    if (f == 0.0f) { return 0; }
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return std::uint16_t(bits >> 16);
}

std::string type_name(std::uint32_t t) {
    switch (t) {
    case 0: return "fp16";
    case 1: return "fp32";
    case 3: return "u8";
    case 5: return "u32";
    case 7: return "s8";
    case 9: return "s32";
    case 1000141000: return "bf16";
    case 1000491002: return "fp8_e4m3";
    case 1000491003: return "fp8_e5m2";
    default: return "type" + std::to_string(t);
    }
}

struct FormatSpec {
    std::string name;
    const char* define;
    std::uint32_t a_type, b_type, c_type;
    std::size_t element_bytes;
    bool is_float; // false for the int8 path (integer operands and accumulator)
};

const FormatSpec kFormats[] = {
    {"fp16", "FORMAT_FP16", 0, 0, 1, 2, true},
    {"bf16", "FORMAT_BF16", 1000141000, 1000141000, 1, 2, true},
    {"int8", "FORMAT_INT8", 7, 7, 9, 1, false},
};

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);

    // Feature structs must be filled before device creation (the device is created with them).
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{};
    coop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coop.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16f{};
    bf16f.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    bf16f.shaderBFloat16Type = VK_TRUE;
    bf16f.shaderBFloat16CooperativeMatrix = VK_TRUE;
    coop.pNext = &bf16f;
    VkPhysicalDevice16BitStorageFeatures stor16{};
    stor16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    stor16.storageBuffer16BitAccess = VK_TRUE;
    bf16f.pNext = &stor16;
    VkPhysicalDevice8BitStorageFeatures stor8{};
    stor8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    stor8.storageBuffer8BitAccess = VK_TRUE;
    stor16.pNext = &stor8;
    VkPhysicalDeviceVulkanMemoryModelFeatures mm{};
    mm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
    mm.vulkanMemoryModel = VK_TRUE;
    stor8.pNext = &mm;

    VkCtx ctx("mma", 0,
              Obj().num("seconds", o.seconds).str("formats", o.formats)
                  .integer("chains", kChains).integer("workgroups", o.workgroups),
              {"VK_KHR_cooperative_matrix", "VK_KHR_vulkan_memory_model",
               "VK_KHR_shader_float16_int8", "VK_KHR_shader_bfloat16", "VK_KHR_16bit_storage",
               "VK_KHR_8bit_storage"},
              &coop);
    ctx.declare_common();

    // The driver's configuration list: every format this probe runs must be in it.
    using PFN_GetCoopProps = PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;
    const auto get_props = reinterpret_cast<PFN_GetCoopProps>(
        vkGetInstanceProcAddr(ctx.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (get_props == nullptr) { fail("loader has no cooperative-matrix properties entry point"); }
    std::uint32_t count = 0;
    EL_VK(get_props(ctx.phys, &count, nullptr));
    std::vector<VkCooperativeMatrixPropertiesKHR> configs(count);
    for (auto& c : configs) { c.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR; }
    if (count > 0) { EL_VK(get_props(ctx.phys, &count, configs.data())); }
    {
        Obj list;
        for (std::uint32_t i = 0; i < count; ++i) {
            list.str(("cfg" + std::to_string(i)).c_str(),
                     type_name(configs[i].AType) + "x" + type_name(configs[i].BType) + "->" +
                         type_name(configs[i].CType) + "/" + type_name(configs[i].ResultType) +
                         " M" + std::to_string(configs[i].MSize) + "N" +
                         std::to_string(configs[i].NSize) + "K" +
                         std::to_string(configs[i].KSize) +
                         (configs[i].saturatingAccumulation ? " sat" : ""));
        }
        declared("vulkan.coopmat_configs", list.text(),
                 "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    }
    const auto config_present = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        for (const auto& cfg : configs) {
            if (cfg.AType == a && cfg.BType == b && cfg.CType == c && cfg.ResultType == c &&
                cfg.MSize == kM && cfg.NSize == kN && cfg.KSize == kK &&
                cfg.scope == VK_SCOPE_SUBGROUP_KHR && !cfg.saturatingAccumulation) {
                return true;
            }
        }
        return false;
    };

    // The device's subgroup size comes from the subgroup-size-control properties (it is not in
    // the base limits struct).
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT subp{};
    subp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 p2sub{};
    p2sub.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2sub.pNext = &subp;
    vkGetPhysicalDeviceProperties2(ctx.phys, &p2sub);
    const std::uint32_t subgroup_size = subp.maxSubgroupSize; // AMD wave64 on RDNA

    const std::uint32_t grid =
        std::min<std::uint32_t>(o.workgroups, ctx.props.limits.maxComputeWorkGroupCount[0]);
    const std::uint32_t local_x = 64; // one or more subgroups; scaled into the FLOP count below
    const double flops_per_iter =
        double(grid) * (double(local_x) / double(subgroup_size)) * double(kChains) * 2.0 *
        double(kM) * double(kN) * double(kK);
    note("grid: " + Obj().uint("workgroups", grid).uint("local_size_x", local_x)
                        .uint("subgroup_size", subgroup_size)
                        .num("flops_per_kernel_iter", flops_per_iter).text());

    const std::vector<double> want = reference();

    for (const FormatSpec& f : kFormats) {
        const std::string tag = f.name;
        if (!wants(o, tag)) { continue; }
        if (!config_present(f.a_type, f.b_type, f.c_type)) {
            note("skipping " + tag +
                 ": the driver does not list an M16N16K16 subgroup config for it");
            continue;
        }
        const std::vector<char> spirv =
            VkCtx::read_spirv(shader_path(argv[0], "coopmul_" + tag + ".spv"));

        VkCtx::Pipeline pl;
        ctx.make_pipeline(pl, spirv, 8);
        VkBufferRes a = ctx.make_buffer(256 * f.element_bytes, false, false, tag + ".a");
        VkBufferRes b = ctx.make_buffer(256 * f.element_bytes, false, false, tag + ".b");
        const std::size_t out_bytes = std::size_t(grid) * kChains * 256 * 4;
        VkBufferRes c = ctx.make_buffer(out_bytes, true, false, tag + ".c");
        VkBufferRes slot = ctx.make_buffer(4, false, true, tag + ".slot");
        // A and B hold the pattern in the format's element type.
        if (f.is_float) {
            auto* pa = static_cast<std::uint16_t*>(a.mapped);
            auto* pb = static_cast<std::uint16_t*>(b.mapped);
            for (int m = 0; m < 16; ++m) {
                for (int k = 0; k < 16; ++k) {
                    pa[std::size_t(m) * 16 + k] =
                        tag == "fp16" ? fp16_bits(a_pattern(m, k)) : bf16_bits(a_pattern(m, k));
                }
            }
            for (int k = 0; k < 16; ++k) {
                for (int n = 0; n < 16; ++n) {
                    // B is loaded column-major (element (k,n) at n*16+k with stride 16)
                    pb[std::size_t(n) * 16 + k] =
                        tag == "fp16" ? fp16_bits(b_pattern(k, n)) : bf16_bits(b_pattern(k, n));
                }
            }
        } else {
            auto* pa = static_cast<std::int8_t*>(a.mapped);
            auto* pb = static_cast<std::int8_t*>(b.mapped);
            for (int m = 0; m < 16; ++m) {
                for (int k = 0; k < 16; ++k) {
                    pa[std::size_t(m) * 16 + k] = std::int8_t(a_pattern(m, k));
                }
            }
            for (int k = 0; k < 16; ++k) {
                for (int n = 0; n < 16; ++n) {
                    // B is loaded column-major (element (k,n) at n*16+k with stride 16)
                    pb[std::size_t(n) * 16 + k] = std::int8_t(b_pattern(k, n));
                }
            }
        }
        const std::uint32_t zero = 0;
        ctx.fill_mapped(slot, &zero, 4);

        VkCtx::Submit sub;
        ctx.make_submit(sub);
        const VkDescriptorSet set = ctx.bind_set(pl, a.buffer, b.buffer, c.buffer, slot.buffer);

        auto run_timed = [&](std::uint32_t iters, std::uint32_t nonce) {
            auto record = [&](VkCommandBuffer cmd, const VkCtx::Pipeline&) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &set,
                                        0, nullptr);
                const std::uint32_t push[2] = {iters, nonce};
                vkCmdPushConstants(cmd, pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
                vkCmdDispatch(cmd, grid, 1, 1);
            };
            return sub.dispatch(pl, slot, record).first;
        };
        auto check_slot = [&](std::uint32_t nonce, Obj& d) {
            std::uint32_t got = 0;
            std::memcpy(&got, slot.mapped, 4);
            if (got != nonce) {
                d.uint("slot", got).uint("expected_slot", nonce);
                return false;
            }
            return true;
        };
        // Every (workgroup, chain) block must equal the reference; the verify dispatch (grid = 1)
        // writes the same layout, so block 0 covers it too.
        // Throughput mode runs iters MulAdds into the same accumulators, so the stored value is
        // iters x the single-MulAdd reference (exact: the reference is an integer <= 64 and the
        // accumulator is fp32 or s32).
        auto check_c = [&](std::uint32_t workgroups_to_check, int chains_to_check, double iters,
                           Obj& d) {
            const float* got_f = f.is_float ? static_cast<const float*>(c.mapped) : nullptr;
            const std::int32_t* got_i =
                f.is_float ? nullptr : static_cast<const std::int32_t*>(c.mapped);
            for (std::uint32_t wg = 0; wg < workgroups_to_check; ++wg) {
                for (int j = 0; j < chains_to_check; ++j) {
                    const std::size_t off = (std::size_t(wg) * kChains + std::size_t(j)) * 256;
                    for (std::size_t i = 0; i < 256; ++i) {
                        const double got = f.is_float ? double(got_f[off + i]) : double(got_i[off + i]);
                        const double expect = want[i] * iters;
                        if (got != expect) {
                            d.uint("wg", wg).integer("chain", j).uint("index", i)
                                .num("got", got).num("expected", expect);
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        // ---- oracle: one MulAdd (iters = 0), full 16x16 result against the exact reference -----
        // The verify kernel stores only chain 0, so only block 0 is checked here.
        {
            const unsigned nonce = rep_nonce(-1);
            auto record = [&](VkCommandBuffer cmd, const VkCtx::Pipeline&) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &set,
                                        0, nullptr);
                const std::uint32_t push[2] = {0u, nonce};
                vkCmdPushConstants(cmd, pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
                vkCmdDispatch(cmd, 1, 1, 1);
            };
            const auto [verify_us, verify_slot] = sub.dispatch(pl, slot, record);
            Obj d;
            d.uint("slot", verify_slot).uint("expected_slot", nonce).num("dispatch_us", verify_us);
            const bool ok = verify_slot == nonce && check_c(1, 1, 1.0, d); // one MulAdd
            check(tag + ".matrix", ok, d);
        }

        // ---- throughput series -----------------------------------------------------------------
        // NOTE: on this machine (RADV GFX1201, Mesa 26.2.4) OpCooperativeMatrixMulAddKHR yields
        // zeros/garbage while loads and stores verified correct, so the matrix oracle fails and
        // every throughput number below is recorded verified=false. The probe refuses to claim an
        // unverified rate; probe_end reports verify_failed. rocBLAS computes fp16 GEMM correctly
        // on the same silicon, so the tensor cores are fine -- this is a RADV lowering defect,
        // and this probe will start reporting verified numbers on a driver that fixes it (or on
        // other hardware through this same backend).
        double warm = 1e30;
        for (int i = 0; i < 3; ++i) { warm = std::min(warm, run_timed(1000, rep_nonce(-2))); }
        const double iters_target = o.seconds * 1e6 / std::max(1.0, warm / 1000.0);
        const std::uint32_t iters =
            std::uint32_t(std::max(1.0, std::min(5e7, std::round(iters_target))));
        std::vector<double> tflops;
        bool all_ok = true;
        for (long long rep = 0;; ++rep) {
            const unsigned nonce = rep_nonce(rep);
            const double us = run_timed(iters, nonce);
            Obj d;
            d.integer("iters", iters).num("gpu_us", us);
            const bool ok = check_slot(nonce, d) && check_c(grid, kChains, double(iters), d);
            all_ok = all_ok && ok;
            const double t = flops_per_iter * double(iters) / (us * 1e-6) / 1e12;
            tflops.push_back(t);
            measure(tag + "_tflops", t, "TFLOPS",
                    Obj().integer("iters", iters).num("gpu_us", us), ok, rep);
            if (double(rep + 1) * (us * 1e-6) >= o.seconds) { break; }
        }
        const Summary s = summarize(tflops);
        measure(tag + "_tflops", s.p50, "TFLOPS",
                Obj().obj("per_rep", summary_obj(s)).integer("iters", iters), all_ok);
        check(tag + ".result", all_ok,
              Obj().integer("reps", long(tflops.size())).integer("iters", iters));
        feature("mma", true, true, all_ok,
                Obj().str("path", "VK_KHR_cooperative_matrix M16N16K16 subgroup scope")
                    .str("formats_verified", all_ok ? tag : "none")
                    .num((tag + "_tflops_p50").c_str(), s.p50));

        ctx.destroy_buffer(slot);
        ctx.destroy_buffer(c);
        ctx.destroy_buffer(b);
        ctx.destroy_buffer(a);
        ctx.destroy_pipeline(pl);
    }

    note("fp8: driver-supported on this GPU (e4m3/e5m2 coopmat configs are in the list above) "
         "but unreachable through this toolchain: glslang has no fp8 cooperative-matrix support, "
         "so this probe does not claim an fp8 number");
    return probe_end();
}
