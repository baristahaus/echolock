// Shared Vulkan machinery for the backend's probes: instance/device/queue, buffers with memory
// type selection, a compute pipeline from SPIR-V, descriptor sets, and timestamped submissions.
//
// The older probes (memory.cpp, pointer_chase.cpp) predate this header and carry their own copy;
// everything here follows the rules they learned:
//   - buffers the CPU streams are allocated from a HOST_CACHED type when one exists (RADV's
//     non-cached host-visible type maps write-combined and makes CPU reads pathologically slow);
//   - descriptor pools carry VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, because freeing
//     a set from a pool without it is invalid and RADV does not return such sets;
//   - a pipeline barrier between transfer writes (fill/copy) and a dispatch, so the shader sees
//     the data.

#pragma once

#include "echolock_vk.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstring>
#include <functional>

namespace echolock {

#define EL_VK(expr)                                                                              \
    do {                                                                                         \
        const VkResult r_ = (expr);                                                             \
        if (r_ != VK_SUCCESS) {                                                                 \
            ::echolock::fail(std::string(#expr) + " -> VkResult " +                             \
                             std::to_string(int(r_)) + " (" + __FILE__ + ":" +                   \
                             std::to_string(__LINE__) + ")");                                    \
        }                                                                                        \
    } while (false)

struct VkBufferRes {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    std::size_t size = 0;
    std::uint32_t memory_type = UINT32_MAX;
    bool device_local = false;
    bool host_visible = false;
    bool host_cached = false;
};

struct VkCtx {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    std::uint32_t timestamp_bits = 0;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem_props{};
    double timestamp_period_ns = 1.0;

    explicit VkCtx(const std::string& probe, int device_index, const Obj& params,
                   const std::vector<std::string>& want_extensions = {},
                   void* device_features_pnext = nullptr) {
        identity().probe = probe;
        identity().backend = "vulkan";
        probe_start(probe, params);

        const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "echolock", 1,
                                   "echolock", 1, VK_API_VERSION_1_3};
        const VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0,
                                     nullptr, 0, nullptr};
        EL_VK(vkCreateInstance(&ci, nullptr, &instance));

        std::uint32_t n = 0;
        EL_VK(vkEnumeratePhysicalDevices(instance, &n, nullptr));
        if (n == 0) { fail("no Vulkan devices"); }
        std::vector<VkPhysicalDevice> devices(n);
        EL_VK(vkEnumeratePhysicalDevices(instance, &n, devices.data()));
        if (std::size_t(device_index) >= devices.size()) {
            fail("device index out of range (" + std::to_string(devices.size()) + " present)");
        }
        phys = devices[device_index];
        vkGetPhysicalDeviceProperties(phys, &props);
        vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
        timestamp_period_ns = double(props.limits.timestampPeriod);

        std::uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &families, nullptr);
        std::vector<VkQueueFamilyProperties> q(families);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &families, q.data());
        int chosen = -1;
        for (std::uint32_t i = 0; i < families; ++i) {
            if ((q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && q[i].timestampValidBits > 0) {
                chosen = int(i);
                timestamp_bits = q[i].timestampValidBits;
                break;
            }
        }
        if (chosen < 0) { fail("no compute queue family with timestamp support"); }
        family = std::uint32_t(chosen);

        const float priority = 1.0f;
        const VkDeviceQueueCreateInfo queue_ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr,
                                               0, family, 1, &priority};
        std::vector<const char*> ext;
        for (const std::string& e : want_extensions) { ext.push_back(e.c_str()); }
        const VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, device_features_pnext,
                                     0, 1, &queue_ci, 0, nullptr,
                                     static_cast<std::uint32_t>(ext.size()),
                                     ext.empty() ? nullptr : ext.data(), nullptr};
        EL_VK(vkCreateDevice(phys, &dci, nullptr, &device));
        vkGetDeviceQueue(device, family, 0, &queue);
    }

    ~VkCtx() {
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
    VkCtx(const VkCtx&) = delete;
    VkCtx& operator=(const VkCtx&) = delete;

    void declare_common() const {
        declared("vulkan.api_version", "\"" + api_string(props.apiVersion) + "\"",
                 "vkGetPhysicalDeviceProperties");
        declared("vulkan.driver_version", "\"" + api_string(props.driverVersion) + "\"",
                 "vkGetPhysicalDeviceProperties");
        declared("device.name", "\"" + json_escape(props.deviceName) + "\"",
                 "vkGetPhysicalDeviceProperties");
        declared("vulkan.timestamp_period_ns", num_json(timestamp_period_ns),
                 "vkGetPhysicalDeviceProperties");
        declared("vulkan.timestamp_valid_bits", std::to_string(timestamp_bits),
                 "vkGetPhysicalDeviceQueueFamilyProperties");
        VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sub{};
        sub.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 p2s{};
        p2s.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2s.pNext = &sub;
        vkGetPhysicalDeviceProperties2(phys, &p2s);
        declared("vulkan.subgroup_size_range",
                 "[" + std::to_string(sub.minSubgroupSize) + ", " +
                     std::to_string(sub.maxSubgroupSize) + "]",
                 "vkGetPhysicalDeviceProperties2");
        VkPhysicalDeviceDriverProperties driver{};
        driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        VkPhysicalDeviceProperties2 p2{};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &driver;
        vkGetPhysicalDeviceProperties2(phys, &p2);
        declared("vulkan.driver_name", "\"" + json_escape(driver.driverName) + "\"",
                 "vkGetPhysicalDeviceProperties2");
        declared("vulkan.driver_info", "\"" + json_escape(driver.driverInfo) + "\"",
                 "vkGetPhysicalDeviceProperties2");
        for (std::uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
            const auto& heap = mem_props.memoryHeaps[i];
            Obj h;
            h.uint("size_bytes", heap.size)
                .boolean("device_local", heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
            declared("vulkan.memory_heap[" + std::to_string(i) + "]", h.text(),
                     "vkGetPhysicalDeviceMemoryProperties");
        }
    }

    std::uint32_t pick_memory_type(bool want_device_local, bool want_host_cached,
                                   bool& out_device_local, bool& out_host_visible,
                                   bool& out_host_cached) const {
        // Two passes: with the HOST_CACHED preference, then without (see the header comment).
        for (int pass = 0; pass < 2; ++pass) {
            const bool need_cached = want_host_cached && pass == 0;
            for (std::uint32_t t = 0; t < mem_props.memoryTypeCount; ++t) {
                const auto& type = mem_props.memoryTypes[t];
                const bool host_ok = type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                const bool coherent = type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                const bool cached = type.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
                const bool local = mem_props.memoryHeaps[type.heapIndex].flags &
                                   VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
                if (!host_ok || !coherent) { continue; }
                if (need_cached && !cached) { continue; }
                if (want_device_local && !local) { continue; }
                out_host_visible = host_ok;
                out_device_local = local;
                out_host_cached = cached;
                return t;
            }
        }
        fail("no host-visible coherent memory type found");
        return 0;
    }

    VkBufferRes make_buffer(std::size_t bytes, bool want_device_local, bool want_host_cached,
                            const std::string& tag, VkBufferUsageFlags extra_usage = 0) {
        VkBufferRes buf;
        buf.size = bytes;
        buf.memory_type = pick_memory_type(want_device_local, want_host_cached, buf.device_local,
                                          buf.host_visible, buf.host_cached);
        note("buffer " + tag + ": " + Obj().uint("size_bytes", bytes)
                                                .boolean("device_local", buf.device_local)
                                                .boolean("host_visible", buf.host_visible)
                                                .boolean("host_cached", buf.host_cached)
                                                .uint("memory_type", buf.memory_type)
                                                .text());
        const VkBufferCreateInfo bi{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra_usage,
            VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        EL_VK(vkCreateBuffer(device, &bi, nullptr, &buf.buffer));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, buf.buffer, &req);
        const VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                                      buf.memory_type};
        EL_VK(vkAllocateMemory(device, &ai, nullptr, &buf.memory));
        EL_VK(vkBindBufferMemory(device, buf.buffer, buf.memory, 0));
        if (buf.host_visible) { EL_VK(vkMapMemory(device, buf.memory, 0, bytes, 0, &buf.mapped)); }
        return buf;
    }

    void destroy_buffer(VkBufferRes& b) {
        if (b.mapped) { vkUnmapMemory(device, b.memory); }
        vkDestroyBuffer(device, b.buffer, nullptr);
        vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }

    // Fill through the mapping (host-visible buffers only; host-coherent, so no flush needed).
    void fill_mapped(VkBufferRes& b, const void* data, std::size_t bytes) const {
        if (!b.mapped) { fail("fill_mapped on a non-host-visible buffer"); }
        std::memcpy(b.mapped, data, bytes);
    }

    // Import host memory (e.g. an mmap'd file) as device memory: the GPU then reads the very
    // pages the page cache owns, so posix_fadvise eviction and mincore residency apply to it.
    VkBufferRes import_host_pointer(void* host, std::size_t bytes, const std::string& tag,
                                    VkBufferUsageFlags extra_usage = 0, bool fatal = true) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_props{};
        host_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 p2{};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &host_props;
        vkGetPhysicalDeviceProperties2(phys, &p2);
        declared("vulkan.min_imported_host_pointer_alignment",
                 std::to_string(host_props.minImportedHostPointerAlignment),
                 "vkGetPhysicalDeviceProperties2");
        if (reinterpret_cast<std::uintptr_t>(host) % host_props.minImportedHostPointerAlignment) {
            fail("host pointer not aligned to " +
                 std::to_string(host_props.minImportedHostPointerAlignment));
        }
        VkBufferRes buf;
        buf.size = bytes;
        buf.host_visible = true;
        buf.host_cached = true; // host memory keeps its own cacheability
        buf.device_local = false;
        note("buffer " + tag + ": imported host pointer, " + Obj().uint("size_bytes", bytes).text());
        const VkBufferCreateInfo bi{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra_usage,
            VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        EL_VK(vkCreateBuffer(device, &bi, nullptr, &buf.buffer));
        VkImportMemoryHostPointerInfoEXT import{};
        import.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
        import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        import.pHostPointer = host;
        // The memory type for an imported host pointer: one with the matching external handle
        // type; RADV exposes exactly one such type (host memory, cacheable as the host mapped it).
        // Only the memory types vkGetMemoryHostPointerPropertiesEXT reports accept the import;
        // guessing a host-visible type fails at vkAllocateMemory.
        using PFN_GetHostPointerProps = PFN_vkGetMemoryHostPointerPropertiesEXT;
        const auto get_host_props = reinterpret_cast<PFN_GetHostPointerProps>(
            vkGetDeviceProcAddr(device, "vkGetMemoryHostPointerPropertiesEXT"));
        if (get_host_props == nullptr) {
            fail("loader has no vkGetMemoryHostPointerPropertiesEXT (missing "
                 "VK_EXT_external_memory_host?)");
        }
        VkMemoryHostPointerPropertiesEXT hp{};
        hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
        EL_VK(get_host_props(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &hp));
        std::uint32_t type = UINT32_MAX;
        for (std::uint32_t t = 0; t < mem_props.memoryTypeCount; ++t) {
            if ((hp.memoryTypeBits & (1u << t)) &&
                (mem_props.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                type = t;
                break;
            }
        }
        if (type == UINT32_MAX) { fail("no memory type accepts the host pointer import"); }
        note("import memory type: " +
             Obj().uint("type", type)
                 .uint("property_flags", mem_props.memoryTypes[type].propertyFlags)
                 .uint("heap", mem_props.memoryTypes[type].heapIndex)
                 .uint("hp_type_bits", hp.memoryTypeBits).text());
        buf.memory_type = type;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &import, bytes, type};
        const VkResult alloc_r = vkAllocateMemory(device, &ai, nullptr, &buf.memory);
        if (alloc_r != VK_SUCCESS) {
            if (fatal) {
                EL_VK(alloc_r);
            } else {
                buf = {};
                note("import rejected with VkResult " + std::to_string(int(alloc_r)));
                return buf;
            }
        }
        EL_VK(vkBindBufferMemory(device, buf.buffer, buf.memory, 0));
        buf.mapped = host; // the import is the same memory; the host mapping is the pointer itself
        return buf;
    }

    // ---- pipeline ---------------------------------------------------------------------------------

    struct Pipeline {
        VkPipeline pipe = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::uint32_t push_bytes = 0;
    };

    // required_subgroup_size: cooperative-matrix pipelines must run full subgroups; RADV needs
    // the size pinned via VK_EXT_subgroup_size_control or the WMMA path silently misbehaves.
    void make_pipeline(Pipeline& pl, const std::vector<char>& spirv,
                       std::uint32_t push_constant_bytes,
                       std::uint32_t required_subgroup_size = 0) const {
        const VkDescriptorSetLayoutBinding bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        const VkDescriptorSetLayoutCreateInfo sl{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, bindings};
        EL_VK(vkCreateDescriptorSetLayout(device, &sl, nullptr, &pl.set_layout));
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_bytes};
        const VkPipelineLayoutCreateInfo plc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr,
                                             0, 1, &pl.set_layout, 1, &push};
        EL_VK(vkCreatePipelineLayout(device, &plc, nullptr, &pl.layout));
        const VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                          spirv.size(),
                                          reinterpret_cast<const std::uint32_t*>(spirv.data())};
        VkShaderModule module = VK_NULL_HANDLE;
        EL_VK(vkCreateShaderModule(device, &sm, nullptr, &module));
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req{};
        if (required_subgroup_size != 0) {
            req.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
            req.requiredSubgroupSize = required_subgroup_size;
        }
        const VkPipelineShaderStageCreateInfo stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            required_subgroup_size != 0 ? &req : nullptr,
            required_subgroup_size != 0
                ? VkPipelineShaderStageCreateFlags(
                      VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT)
                : VkPipelineShaderStageCreateFlags(0),
            VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        const VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr,
                                             0, stage, pl.layout, VK_NULL_HANDLE, 0};
        EL_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pl.pipe));
        vkDestroyShaderModule(device, module, nullptr);
        // FREE_DESCRIPTOR_SET_BIT: probes free their sets, and without the flag that free is
        // invalid (RADV does not return freed sets, so the pool would exhaust).
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128};
        const VkDescriptorPoolCreateInfo pc{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,
            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 32, 1, &size};
        EL_VK(vkCreateDescriptorPool(device, &pc, nullptr, &pl.pool));
        pl.push_bytes = push_constant_bytes;
    }

    // Up to four storage bindings; a buffer passed as VK_NULL_HANDLE keeps that binding unwritten
    // (the shader must not access it).
    VkDescriptorSet bind_set(const Pipeline& pl, VkBuffer b0, VkBuffer b1, VkBuffer b2 = VK_NULL_HANDLE,
                             VkBuffer b3 = VK_NULL_HANDLE) const {
        const VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                             pl.pool, 1, &pl.set_layout};
        VkDescriptorSet set = VK_NULL_HANDLE;
        EL_VK(vkAllocateDescriptorSets(device, &ai, &set));
        const VkBuffer bufs[4] = {b0, b1, b2, b3};
        VkDescriptorBufferInfo info[4]{};
        VkWriteDescriptorSet writes[4]{};
        std::uint32_t n = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            if (bufs[i] == VK_NULL_HANDLE) { continue; }
            info[n] = VkDescriptorBufferInfo{bufs[i], 0, VK_WHOLE_SIZE};
            writes[n] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, i,
                                             0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr,
                                             &info[n], nullptr};
            ++n;
        }
        vkUpdateDescriptorSets(device, n, writes, 0, nullptr);
        return set;
    }

    void destroy_pipeline(Pipeline& pl) const {
        vkDestroyDescriptorPool(device, pl.pool, nullptr);
        vkDestroyDescriptorSetLayout(device, pl.set_layout, nullptr);
        vkDestroyPipeline(device, pl.pipe, nullptr);
        vkDestroyPipelineLayout(device, pl.layout, nullptr);
        pl = {};
    }

    // ---- submission --------------------------------------------------------------------------------

    struct Submit {
        VkCtx* vk = nullptr;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkQueryPool queries = VK_NULL_HANDLE;

        // One dispatch with two timestamps around it. `record` fills the command buffer (bindings,
        // push constants, dispatch size); the slot buffer is cleared and transfer writes are
        // barriered into shader visibility before the dispatch. Returns (gpu_us, slot value).
        std::pair<double, std::uint32_t> dispatch(
            const Pipeline& pl, VkBufferRes& slot_buf,
            const std::function<void(VkCommandBuffer, const Pipeline&)>& record) const {
            EL_VK(vkResetCommandBuffer(cmd, 0));
            const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                              0, nullptr};
            EL_VK(vkBeginCommandBuffer(cmd, &bi));
            if (slot_buf.buffer != VK_NULL_HANDLE) {
                vkCmdFillBuffer(cmd, slot_buf.buffer, 0, 4, 0); // fresh slot: an unwritten slot
                // cannot match the expectation
            }
            const VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                     VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
                                 nullptr);
            vkCmdResetQueryPool(cmd, queries, 0, 2);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
            record(cmd, pl);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
            EL_VK(vkEndCommandBuffer(cmd));
            const double us = finish();
            std::uint32_t slot = 0;
            if (slot_buf.mapped != nullptr) { std::memcpy(&slot, slot_buf.mapped, 4); }
            return {us, slot};
        }

        double finish() const {
            const VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1,
                                  &cmd, 0, nullptr};
            EL_VK(vkQueueSubmit(vk->queue, 1, &si, fence));
            EL_VK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
            EL_VK(vkResetFences(vk->device, 1, &fence));
            std::uint64_t ts[2] = {0, 0};
            EL_VK(vkGetQueryPoolResults(vk->device, queries, 0, 2, sizeof(ts), ts,
                                        sizeof(std::uint64_t),
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            return double(ts[1] - ts[0]) * vk->timestamp_period_ns / 1e3;
        }

        double copy(const VkBufferRes& src, const VkBufferRes& dst, std::size_t bytes) const {
            EL_VK(vkResetCommandBuffer(cmd, 0));
            const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                              0, nullptr};
            EL_VK(vkBeginCommandBuffer(cmd, &bi));
            vkCmdResetQueryPool(cmd, queries, 0, 2);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
            const VkBufferCopy region{0, 0, bytes};
            vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &region);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
            EL_VK(vkEndCommandBuffer(cmd));
            return finish();
        }
    };

    void make_submit(Submit& sub) {
        sub.vk = this;
        const VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0,
                                          family};
        EL_VK(vkCreateCommandPool(device, &pci, nullptr, &sub_cmd_pool));
        const VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                              sub_cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        EL_VK(vkAllocateCommandBuffers(device, &cai, &sub.cmd));
        const VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
        EL_VK(vkCreateFence(device, &fci, nullptr, &sub.fence));
        const VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
                                        VK_QUERY_TYPE_TIMESTAMP, 2, 0};
        EL_VK(vkCreateQueryPool(device, &qci, nullptr, &sub.queries));
    }

    void destroy_submit(const Submit&) const {
        vkDestroyCommandPool(device, sub_cmd_pool, nullptr);
    }

    static std::vector<char> read_spirv(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) { fail("cannot open shader " + path + " (build it with make)"); }
        std::vector<char> out;
        char chunk[1 << 16];
        for (std::size_t got; (got = std::fread(chunk, 1, sizeof(chunk), f)) > 0;) {
            out.insert(out.end(), chunk, chunk + got);
        }
        std::fclose(f);
        if (out.size() < 4 || out.size() % 4 != 0) {
            fail("shader " + path + " is not valid SPIR-V");
        }
        return out;
    }

  private:
    mutable VkCommandPool sub_cmd_pool = VK_NULL_HANDLE;

    static std::string api_string(std::uint32_t v) {
        return std::to_string(VK_API_VERSION_MAJOR(v)) + "." +
               std::to_string(VK_API_VERSION_MINOR(v)) + "." +
               std::to_string(VK_API_VERSION_PATCH(v));
    }
    static std::string num_json(double v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.10g", v);
        return buf;
    }
};

std::string shader_path(const std::string& exe_path, const std::string& name) {
    const std::size_t slash = exe_path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "." : exe_path.substr(0, slash);
    return dir + "/" + name;
}

} // namespace echolock
