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

#include "common/echolock_vk.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstring>

using namespace echolock;

namespace {

#define EL_VK(expr)                                                                              \
    do {                                                                                          \
        const VkResult r_ = (expr);                                                               \
        if (r_ != VK_SUCCESS) {                                                                   \
            ::echolock::fail(std::string(#expr) + " -> VkResult " +                               \
                             std::to_string(int(r_)) + " (" + __FILE__ + ":" +                     \
                             std::to_string(__LINE__) + ")");                                      \
        }                                                                                         \
    } while (false)

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

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    std::size_t size = 0;
};

struct Push {
    std::uint32_t hops;
    std::uint32_t nonce;
};

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    probe_start("pointer_chase",
                Obj().integer("hops", o.hops).integer("reps", o.reps).str("kinds", o.kinds)
                     .str("patterns", o.patterns).integer("device", o.device));

    // Instance / device (same requirements as the memory probe: compute queue with timestamps).
    VkInstance instance = VK_NULL_HANDLE;
    {
        const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "echolock", 1,
                                   "echolock", 1, VK_API_VERSION_1_3};
        const VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0,
                                     nullptr, 0, nullptr};
        EL_VK(vkCreateInstance(&ci, nullptr, &instance));
    }
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    {
        std::uint32_t n = 0;
        EL_VK(vkEnumeratePhysicalDevices(instance, &n, nullptr));
        std::vector<VkPhysicalDevice> devices(n);
        EL_VK(vkEnumeratePhysicalDevices(instance, &n, devices.data()));
        if (std::size_t(o.device) >= devices.size()) { fail("device index out of range"); }
        phys = devices[o.device];
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    const double timestamp_period_ns = double(props.limits.timestampPeriod);
    declared("device.name", "\"" + json_escape(props.deviceName) + "\"",
             "vkGetPhysicalDeviceProperties");
    declared("vulkan.timestamp_period_ns", std::to_string(timestamp_period_ns),
             "vkGetPhysicalDeviceProperties");

    std::uint32_t family = 0;
    {
        std::uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &families, nullptr);
        std::vector<VkQueueFamilyProperties> q(families);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &families, q.data());
        bool found = false;
        for (std::uint32_t i = 0; i < families; ++i) {
            if ((q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && q[i].timestampValidBits > 0) {
                family = i;
                found = true;
                break;
            }
        }
        if (!found) { fail("no compute queue family with timestamp support"); }
    }
    VkDevice device = VK_NULL_HANDLE;
    {
        const float priority = 1.0f;
        const VkDeviceQueueCreateInfo queue_ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr,
                                               0, family, 1, &priority};
        const VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queue_ci,
                                     0, nullptr, 0, nullptr, nullptr};
        EL_VK(vkCreateDevice(phys, &dci, nullptr, &device));
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &queue);

    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);

    // Shader + pipeline: chain buffer + result slot, hops/nonce push constants.
    std::string exe = argc > 0 ? argv[0] : "pointer_chase";
    const std::size_t slash = exe.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "." : exe.substr(0, slash);
    std::vector<char> spirv;
    {
        FILE* f = std::fopen((dir + "/chase.spv").c_str(), "rb");
        if (f == nullptr) { fail("cannot open shader " + dir + "/chase.spv (build it with make)"); }
        char chunk[1 << 16];
        for (std::size_t got; (got = std::fread(chunk, 1, sizeof(chunk), f)) > 0;) {
            spirv.insert(spirv.end(), chunk, chunk + got);
        }
        std::fclose(f);
    }
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    {
        const VkDescriptorSetLayoutBinding bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        const VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                 nullptr, 0, 2, bindings};
        EL_VK(vkCreateDescriptorSetLayout(device, &sl, nullptr, &set_layout));
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        const VkPipelineLayoutCreateInfo plc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr,
                                             0, 1, &set_layout, 1, &push};
        EL_VK(vkCreatePipelineLayout(device, &plc, nullptr, &pipe_layout));
        const VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                          spirv.size(),
                                          reinterpret_cast<const std::uint32_t*>(spirv.data())};
        VkShaderModule module = VK_NULL_HANDLE;
        EL_VK(vkCreateShaderModule(device, &sm, nullptr, &module));
        const VkPipelineShaderStageCreateInfo stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
            VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        const VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr,
                                             0, stage, pipe_layout, VK_NULL_HANDLE, 0};
        EL_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
        vkDestroyShaderModule(device, module, nullptr);
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64};
        // FREE_DESCRIPTOR_SET_BIT: the probe frees each series' set after use, and without the
        // flag that free is invalid (the pool would exhaust after a few series on RADV).
        const VkDescriptorPoolCreateInfo pc{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,
            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 32, 1, &size};
        EL_VK(vkCreateDescriptorPool(device, &pc, nullptr, &pool));
    }
    const VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, family};
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    EL_VK(vkCreateCommandPool(device, &pci, nullptr, &cmd_pool));
    const VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                          cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    EL_VK(vkAllocateCommandBuffers(device, &cai, &cmd));
    const VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    VkFence fence = VK_NULL_HANDLE;
    EL_VK(vkCreateFence(device, &fci, nullptr, &fence));
    const VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
                                    VK_QUERY_TYPE_TIMESTAMP, 2, 0};
    VkQueryPool queries = VK_NULL_HANDLE;
    EL_VK(vkCreateQueryPool(device, &qci, nullptr, &queries));

    // The slot: host-visible + cached (the host reads it after every dispatch).
    Buffer slot_buf{};
    {
        const VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, 4,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        EL_VK(vkCreateBuffer(device, &bi, nullptr, &slot_buf.buffer));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, slot_buf.buffer, &req);
        std::uint32_t type = UINT32_MAX;
        for (std::uint32_t t = 0; t < mem_props.memoryTypeCount; ++t) {
            const auto& m = mem_props.memoryTypes[t];
            if ((m.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (m.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
                (m.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
                type = t;
                break;
            }
        }
        if (type == UINT32_MAX) {
            for (std::uint32_t t = 0; t < mem_props.memoryTypeCount; ++t) {
                if (mem_props.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
                    type = t;
                    break;
                }
            }
        }
        const VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                                      type};
        EL_VK(vkAllocateMemory(device, &ai, nullptr, &slot_buf.memory));
        EL_VK(vkBindBufferMemory(device, slot_buf.buffer, slot_buf.memory, 0));
        EL_VK(vkMapMemory(device, slot_buf.memory, 0, 4, 0, &slot_buf.mapped));
    }

    auto make_set = [&](VkBuffer chain) {
        const VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                             pool, 1, &set_layout};
        VkDescriptorSet set = VK_NULL_HANDLE;
        EL_VK(vkAllocateDescriptorSets(device, &ai, &set));
        const VkDescriptorBufferInfo info_chain{chain, 0, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo info_out{slot_buf.buffer, 0, VK_WHOLE_SIZE};
        const VkWriteDescriptorSet writes[] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1,
             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &info_chain, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1,
             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &info_out, nullptr},
        };
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
        return set;
    };

    // One verified chase dispatch: fresh slot, timestamps, result read back.
    auto chase = [&](VkDescriptorSet set, const Push& push) {
        EL_VK(vkResetCommandBuffer(cmd, 0));
        const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0,
                                          nullptr};
        EL_VK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdFillBuffer(cmd, slot_buf.buffer, 0, 4, 0);
        const VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
                             nullptr);
        vkCmdResetQueryPool(cmd, queries, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
        vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &set, 0,
                                nullptr);
        vkCmdDispatch(cmd, 1, 1, 1); // one invocation: the walk is serial by design
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
        EL_VK(vkEndCommandBuffer(cmd));
        const VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmd,
                              0, nullptr};
        EL_VK(vkQueueSubmit(queue, 1, &si, fence));
        EL_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        EL_VK(vkResetFences(device, 1, &fence));
        std::uint64_t ts[2] = {0, 0};
        EL_VK(vkGetQueryPoolResults(device, queries, 0, 2, sizeof(ts), ts,
                                    sizeof(std::uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        std::uint32_t slot = 0;
        std::memcpy(&slot, slot_buf.mapped, 4);
        return std::make_pair(double(ts[1] - ts[0]) * timestamp_period_ns / 1e3, slot);
    };

    // A chain buffer of `bytes` from the requested heap; the host fills it through the mapping.
    auto make_chain_buffer = [&](std::size_t bytes, bool want_device_local,
                                 const std::string& tag) {
        Buffer b;
        b.size = bytes;
        std::uint32_t type = UINT32_MAX;
        bool host_visible = false, device_local = false;
        for (int pass = 0; pass < 2 && type == UINT32_MAX; ++pass) {
            for (std::uint32_t t = 0; t < mem_props.memoryTypeCount; ++t) {
                const auto& m = mem_props.memoryTypes[t];
                const bool host_ok = m.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                const bool coherent = m.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                const bool cached = m.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
                const bool local = mem_props.memoryHeaps[m.heapIndex].flags &
                                   VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
                if (!host_ok || !coherent) { continue; }
                if (pass == 0 && !cached) { continue; }
                if (want_device_local && !local) { continue; }
                type = t;
                host_visible = host_ok;
                device_local = local;
                break;
            }
        }
        if (type == UINT32_MAX) { fail("no usable memory type for " + tag); }
        note("buffer " + tag + ": " + Obj().uint("size_bytes", bytes)
                                                .boolean("device_local", device_local)
                                                .boolean("host_visible", host_visible)
                                                .uint("memory_type", type)
                                                .text());
        const VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        EL_VK(vkCreateBuffer(device, &bi, nullptr, &b.buffer));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, b.buffer, &req);
        const VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                                      type};
        EL_VK(vkAllocateMemory(device, &ai, nullptr, &b.memory));
        EL_VK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
        EL_VK(vkMapMemory(device, b.memory, 0, bytes, 0, &b.mapped));
        return b;
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
                const bool want_dev = kind == "device";
                Buffer buf = make_chain_buffer(chain_bytes, want_dev, kind);
                std::memcpy(buf.mapped, chain.data(), chain_bytes);
                VkDescriptorSet set = make_set(buf.buffer);

                // Warm-up, then reps of timed verified walks.
                chase(set, Push{o.hops, rep_nonce(-1)});
                std::vector<double> per;
                int bad = 0;
                for (int r = 0; r < o.reps; ++r) {
                    const unsigned nonce = rep_nonce(r);
                    const auto [gpu_us, slot] = chase(set, Push{o.hops, nonce});
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

                vkFreeDescriptorSets(device, pool, 1, &set);
                vkUnmapMemory(device, buf.memory);
                vkDestroyBuffer(device, buf.buffer, nullptr);
                vkFreeMemory(device, buf.memory, nullptr);
            }
        }
    }

    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
    vkDestroyPipeline(device, pipe, nullptr);
    vkDestroyPipelineLayout(device, pipe_layout, nullptr);
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyQueryPool(device, queries, nullptr);
    vkUnmapMemory(device, slot_buf.memory);
    vkDestroyBuffer(device, slot_buf.buffer, nullptr);
    vkFreeMemory(device, slot_buf.memory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return probe_end();
}
