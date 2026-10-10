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
// Build: make -C backends/vulkan (glslc for the shader, g++ for the probe, -lvulkan).

#include "common/echolock_vk.h"

#include <vulkan/vulkan.h>

#include <atomic>
#include <cmath>
#include <cstring>

using namespace echolock;

namespace {

#define EL_VK(expr)                                                                              \
    do {                                                                                         \
        const VkResult r_ = (expr);                                                             \
        if (r_ != VK_SUCCESS) {                                                                 \
            ::echolock::fail(std::string(#expr) + " -> VkResult " +                             \
                             std::to_string(int(r_)) + " (" + __FILE__ + ":" +                   \
                             std::to_string(__LINE__) + ")");                                    \
        }                                                                                        \
    } while (false)

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

std::string api_string(std::uint32_t v) {
    return std::to_string(VK_API_VERSION_MAJOR(v)) + "." + std::to_string(VK_API_VERSION_MINOR(v)) +
           "." + std::to_string(VK_API_VERSION_PATCH(v));
}

// ---- device --------------------------------------------------------------------------------------

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    std::size_t size = 0;
    std::uint32_t memory_type = UINT32_MAX;
    bool device_local = false;
    bool host_visible = false;
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

    explicit VkCtx(const Options& o) {
        const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
                                   "echolock", 1, "echolock", 1, VK_API_VERSION_1_3};
        const VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0,
                                     nullptr, 0, nullptr};
        EL_VK(vkCreateInstance(&ci, nullptr, &instance));

        std::uint32_t n = 0;
        EL_VK(vkEnumeratePhysicalDevices(instance, &n, nullptr));
        if (n == 0) { fail("no Vulkan devices"); }
        std::vector<VkPhysicalDevice> devices(n);
        EL_VK(vkEnumeratePhysicalDevices(instance, &n, devices.data()));
        if (std::size_t(o.device) >= devices.size()) {
            fail("device index out of range (" + std::to_string(devices.size()) + " present)");
        }
        phys = devices[o.device];
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
        const VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queue_ci,
                                     0, nullptr, 0, nullptr, nullptr};
        EL_VK(vkCreateDevice(phys, &dci, nullptr, &device));
        vkGetDeviceQueue(device, family, 0, &queue);
    }

    ~VkCtx() {
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
    VkCtx(const VkCtx&) = delete;
    VkCtx& operator=(const VkCtx&) = delete;

    void declare_device() const {
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
        declared("vulkan.max_compute_workgroups",
                 "[" + std::to_string(props.limits.maxComputeWorkGroupCount[0]) + ", " +
                     std::to_string(props.limits.maxComputeWorkGroupCount[1]) + ", " +
                     std::to_string(props.limits.maxComputeWorkGroupCount[2]) + "]",
                 "vkGetPhysicalDeviceProperties");
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
                                   bool& out_device_local, bool& out_host_visible) const {
        // Two passes: with the HOST_CACHED preference, then without. A host-visible type
        // without HOST_CACHED maps write-combined on many drivers (RADV's GTT here), which is
        // fine for writes but makes CPU *reads* pathologically slow -- exactly what the CPU
        // side of the contention series does, so that side asks for cached.
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
                return t;
            }
        }
        fail("no host-visible coherent memory type found");
        return 0;
    }

    Buffer make_buffer(std::size_t bytes, bool want_device_local, bool want_host_cached,
                       const std::string& tag) {
        Buffer buf;
        buf.size = bytes;
        buf.memory_type = pick_memory_type(want_device_local, want_host_cached, buf.device_local,
                                          buf.host_visible);
        note("buffer " + tag + ": " + Obj().uint("size_bytes", bytes)
                                             .boolean("device_local", buf.device_local)
                                             .boolean("host_visible", buf.host_visible)
                                             .uint("memory_type", buf.memory_type)
                                             .text());
        const VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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

    void destroy_buffer(Buffer& b) {
        if (b.mapped) { vkUnmapMemory(device, b.memory); }
        vkDestroyBuffer(device, b.buffer, nullptr);
        vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }
};

std::vector<char> read_spirv(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { fail("cannot open shader " + path + " (build it with make)"); }
    std::vector<char> out;
    char chunk[1 << 16];
    for (std::size_t got; (got = std::fread(chunk, 1, sizeof(chunk), f)) > 0;) {
        out.insert(out.end(), chunk, chunk + got);
    }
    std::fclose(f);
    if (out.size() < 4 || out.size() % 4 != 0) { fail("shader " + path + " is not valid SPIR-V"); }
    return out;
}

struct Push {
    std::uint32_t words;
    std::uint32_t nonce;
};

// ---- command submission --------------------------------------------------------------------------

struct Submit {
    VkCtx& vk;
    VkCommandBuffer cmd;
    VkFence fence;
    VkQueryPool queries;

    // One verified read dispatch: fills the slot, barriers transfer writes into shader
    // visibility, brackets the dispatch with timestamps, and returns (gpu_us, slot).
    std::pair<double, std::uint32_t> read(VkPipeline pipe, VkPipelineLayout layout,
                                          VkDescriptorSet set, const Push& push,
                                          std::uint32_t workgroups, Buffer& slot_buf) {
        EL_VK(vkResetCommandBuffer(cmd, 0));
        const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0,
                                          nullptr};
        EL_VK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdFillBuffer(cmd, slot_buf.buffer, 0, 4, 0); // fresh slot each dispatch
        // The barrier makes both this fill and any earlier transfer writes on the queue (a copy
        // from a previous submission) visible to the shader before it runs.
        const VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
                             nullptr);
        vkCmdResetQueryPool(cmd, queries, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(cmd, workgroups, 1, 1);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
        EL_VK(vkEndCommandBuffer(cmd));
        const VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmd,
                              0, nullptr};
        EL_VK(vkQueueSubmit(vk.queue, 1, &si, fence));
        EL_VK(vkWaitForFences(vk.device, 1, &fence, VK_TRUE, UINT64_MAX));
        EL_VK(vkResetFences(vk.device, 1, &fence));
        std::uint64_t ts[2] = {0, 0};
        EL_VK(vkGetQueryPoolResults(vk.device, queries, 0, 2, sizeof(ts), ts,
                                    sizeof(std::uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        std::uint32_t slot = 0;
        std::memcpy(&slot, slot_buf.mapped, 4); // host-coherent: fence wait makes it visible
        return {double(ts[1] - ts[0]) * vk.timestamp_period_ns / 1e3, slot};
    }

    double copy(const Buffer& src, const Buffer& dst, std::size_t bytes) {
        EL_VK(vkResetCommandBuffer(cmd, 0));
        const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0,
                                          nullptr};
        EL_VK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdResetQueryPool(cmd, queries, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
        const VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &region);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
        EL_VK(vkEndCommandBuffer(cmd));
        const VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmd,
                              0, nullptr};
        EL_VK(vkQueueSubmit(vk.queue, 1, &si, fence));
        EL_VK(vkWaitForFences(vk.device, 1, &fence, VK_TRUE, UINT64_MAX));
        EL_VK(vkResetFences(vk.device, 1, &fence));
        std::uint64_t ts[2] = {0, 0};
        EL_VK(vkGetQueryPoolResults(vk.device, queries, 0, 2, sizeof(ts), ts,
                                    sizeof(std::uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        return double(ts[1] - ts[0]) * vk.timestamp_period_ns / 1e3;
    }
};

struct SeriesResult {
    double gbps = 0;
    bool verified = false;
};

// Warm-up, then `reps` timed verified reads: one measure per rep, a summary, one check.
SeriesResult read_series(Submit& sub, VkPipeline pipe, VkPipelineLayout layout,
                         VkDescriptorSet set, Buffer& slot_buf, std::uint32_t workgroups,
                         std::uint32_t words, int reps, std::uint32_t expected, const Obj& tags,
                         const std::string& name) {
    sub.read(pipe, layout, set, Push{words, rep_nonce(-1)}, workgroups, slot_buf); // warm-up
    std::vector<double> per;
    int bad = 0;
    for (int r = 0; r < reps; ++r) {
        const unsigned nonce = rep_nonce(r);
        const auto [gpu_us, slot] =
            sub.read(pipe, layout, set, Push{words, nonce}, workgroups, slot_buf);
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
    probe_start("memory", Obj().uint("bytes", std::uint64_t(o.mib) << 20)
                                 .num("seconds", o.seconds)
                                 .integer("cpu_threads", o.cpu_threads)
                                 .integer("workgroups", o.workgroups)
                                 .integer("device", o.device));

    VkCtx vk(o);
    vk.declare_device();

    // Shader lives next to the binary; the Makefile puts it there.
    std::string exe = argc > 0 ? argv[0] : "memory";
    const std::size_t slash = exe.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "." : exe.substr(0, slash);
    const std::vector<char> spirv = read_spirv(dir + "/read_fold.spv");

    // Pipeline + descriptor machinery.
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    {
        const VkDescriptorSetLayoutBinding bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        const VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                  nullptr, 0, 2, bindings};
        EL_VK(vkCreateDescriptorSetLayout(vk.device, &sl, nullptr, &set_layout));
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        const VkPipelineLayoutCreateInfo plc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr,
                                             0, 1, &set_layout, 1, &push};
        EL_VK(vkCreatePipelineLayout(vk.device, &plc, nullptr, &pipe_layout));
        const VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                          spirv.size(),
                                          reinterpret_cast<const std::uint32_t*>(spirv.data())};
        VkShaderModule module = VK_NULL_HANDLE;
        EL_VK(vkCreateShaderModule(vk.device, &sm, nullptr, &module));
        const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                   nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module,
                                                   "main", nullptr};
        const VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr,
                                             0, stage, pipe_layout, VK_NULL_HANDLE, 0};
        EL_VK(vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
        vkDestroyShaderModule(vk.device, module, nullptr);
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
        const VkDescriptorPoolCreateInfo pc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,
                                            0, 4, 1, &size};
        EL_VK(vkCreateDescriptorPool(vk.device, &pc, nullptr, &pool));
    }

    const std::size_t bytes = std::uint64_t(o.mib) << 20;
    const std::uint32_t words = std::uint32_t(bytes / 4);
    Buffer host_buf = vk.make_buffer(bytes, false, true, "host-visible");
    Buffer dev_buf = vk.make_buffer(bytes, true, false, "device-local");
    Buffer slot_buf = vk.make_buffer(4, false, true, "result-slot");

    // Fill the host buffer with the pattern; its XOR is the oracle for everything downstream.
    {
        auto* p = static_cast<std::uint32_t*>(host_buf.mapped);
        for (std::uint32_t i = 0; i < words; ++i) { p[i] = pattern_word(i); }
    }
    const std::uint32_t host_expected = expected_xor(words);

    // Command pool / buffer / fence / timestamp queries.
    const VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0,
                                      vk.family};
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    EL_VK(vkCreateCommandPool(vk.device, &pci, nullptr, &cmd_pool));
    const VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                          cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    EL_VK(vkAllocateCommandBuffers(vk.device, &cai, &cmd));
    const VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    VkFence fence = VK_NULL_HANDLE;
    EL_VK(vkCreateFence(vk.device, &fci, nullptr, &fence));
    const VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
                                    VK_QUERY_TYPE_TIMESTAMP, 2, 0};
    VkQueryPool queries = VK_NULL_HANDLE;
    EL_VK(vkCreateQueryPool(vk.device, &qci, nullptr, &queries));
    Submit sub{vk, cmd, fence, queries};

    // Descriptor sets: one binding the device-local buffer, one the host-visible buffer.
    auto bind_set = [&](VkBuffer data) {
        const VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                             pool, 1, &set_layout};
        VkDescriptorSet set = VK_NULL_HANDLE;
        EL_VK(vkAllocateDescriptorSets(vk.device, &ai, &set));
        const VkDescriptorBufferInfo info_data{data, 0, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo info_out{slot_buf.buffer, 0, VK_WHOLE_SIZE};
        const VkWriteDescriptorSet writes[] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1,
             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &info_data, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1,
             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &info_out, nullptr},
        };
        vkUpdateDescriptorSets(vk.device, 2, writes, 0, nullptr);
        return set;
    };
    VkDescriptorSet dev_set = bind_set(dev_buf.buffer);
    VkDescriptorSet host_set = bind_set(host_buf.buffer);

    // Fill the device-local buffer by a copy, then verify it with a read dispatch: the copy's
    // payload is checked, not assumed.
    {
        const double fill_us = sub.copy(host_buf, dev_buf, bytes);
        note("device-local buffer filled by copy in " + num_json(fill_us) + " us");
        const auto [gpu_us, slot] =
            sub.read(pipe, pipe_layout, dev_set, Push{words, rep_nonce(-2)}, o.workgroups, slot_buf);
        check("dev_fill.xor", slot == fold_nonce(host_expected, rep_nonce(-2)),
              Obj().num("gpu_us", gpu_us).uint("got", slot)
                  .uint("expected", fold_nonce(host_expected, rep_nonce(-2))));
    }

    // Size the series from one warm dispatch, like the CUDA probe does (gpu_us is in
    // microseconds; a series targets `seconds` of dispatches).
    int reps = 3;
    {
        const auto [gpu_us, slot] =
            sub.read(pipe, pipe_layout, dev_set, Push{words, rep_nonce(-1)}, o.workgroups, slot_buf);
        reps = std::clamp(int(o.seconds * 1e6 / std::max(gpu_us, 0.001)), 3, 1024);
        (void)slot;
    }

    // 1. GPU reads of device-local memory.
    const SeriesResult dev_alone =
        read_series(sub, pipe, pipe_layout, dev_set, slot_buf, o.workgroups, words, reps,
                    host_expected,
                    Obj().str("source", "vulkan_device_local")
                        .boolean("heap_device_local", dev_buf.device_local)
                        .boolean("heap_host_visible", dev_buf.host_visible)
                        .str("contention", "none"),
                    "gpu_read");

    // 2. GPU reads of host-visible memory.
    read_series(sub, pipe, pipe_layout, host_set, slot_buf, o.workgroups, words, reps,
                host_expected,
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
            const auto [gpu_us, slot] =
                sub.read(pipe, pipe_layout, dev_set, Push{words, nonce}, o.workgroups, slot_buf);
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
        read_series(sub, pipe, pipe_layout, dev_set, slot_buf, o.workgroups, words, reps,
                    host_expected,
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

    vkDestroyDescriptorPool(vk.device, pool, nullptr);
    vkDestroyDescriptorSetLayout(vk.device, set_layout, nullptr);
    vkDestroyPipeline(vk.device, pipe, nullptr);
    vkDestroyPipelineLayout(vk.device, pipe_layout, nullptr);
    vkDestroyCommandPool(vk.device, cmd_pool, nullptr); // frees the command buffer
    vkDestroyFence(vk.device, fence, nullptr);
    vkDestroyQueryPool(vk.device, queries, nullptr);
    vk.destroy_buffer(slot_buf);
    vk.destroy_buffer(dev_buf);
    vk.destroy_buffer(host_buf);
    return probe_end();
}
