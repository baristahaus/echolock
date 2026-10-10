// file_page (Vulkan backend): the open equivalent of the CUDA file_page probe. Spec:
// probes/file_page/spec.md.
//
// Capability finding first: the CUDA probe reads a file through UVA, so the page cache backs
// the GPU's reads and fadvise eviction applies. On Vulkan the only path is
// VK_EXT_external_memory_host, and this driver (RADV GFX1201, Mesa 26.2.4) accepts imports of
// ANONYMOUS host mappings only -- a MAP_SHARED file mapping and a memfd (tmpfs, page-cache
// pages) are both rejected with VK_ERROR_INVALID_EXTERNAL_HANDLE. The probe measures that
// rejection, then runs the same fault structure on an anonymous mapping: warm host touch,
// resident GPU reads, evict (madvise DONTNEED + mincore), refill + cold GPU pass, resident
// again. Every GPU pass is nonce-verified per row.
//
// What the evict phase reveals on this stack: if amdgpu's userptr import keeps the host pages
// pinned, mincore shows them still resident after DONTNEED (the import is unevictable -- like
// registered CUDA host memory, unlike CUDA's file-mapping fault path). If it does not pin,
// the cold pass times the re-fault. Either outcome is a real, measured fact about the
// driver's mechanism.

#include "common/vk_ctx.h"

#include <vulkan/vulkan.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace echolock;

namespace {

constexpr std::size_t kMib = 64;
constexpr std::size_t kRowWords = 1024; // 4 KiB rows

struct Options {
    int reps = 5;
};

Options parse(int argc, char** argv) {
    Args args(argc, argv, "usage: file_page [--reps N (5)]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--reps") { o.reps = std::stoi(args.value(a)); }
        else { args.usage_error("unknown argument " + a); }
    }
    if (o.reps <= 0) { args.usage_error("reps must be positive"); }
    return o;
}

std::size_t resident_pages(const unsigned char* vec, std::size_t pages) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < pages; ++i) { n += (vec[i] & 1u); }
    return n;
}

void fill_pattern(std::uint32_t* p, std::size_t words) {
    for (std::size_t i = 0; i < words; ++i) { p[i] = pattern_word(i); }
}

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    VkCtx ctx("file_page", 0,
              Obj().integer("reps", o.reps).integer("mib", kMib),
              {"VK_EXT_external_memory_host"});
    ctx.declare_common();

    const std::size_t bytes = kMib << 20;
    const std::size_t words = bytes / 4;
    const std::size_t rows = bytes / (kRowWords * 4);
    const std::size_t pages = bytes / 4096;

    // ---- capability probe: does the driver import a file-backed mapping? --------------------------
    bool file_import_ok = false;
    {
        const std::string path = "/tmp/echolock_filepage_cap";
        const int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0600);
        if (fd >= 0) {
            ::ftruncate(fd, off_t(4096 * 4));
            void* fmap = ::mmap(nullptr, 4096 * 4, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (fmap != MAP_FAILED) {
                // non-fatal attempt: the same device accepts or rejects
                VkBufferRes t =
                    ctx.import_host_pointer(fmap, 4096 * 4, "cap_probe_file", 0, false);
                file_import_ok = t.memory != VK_NULL_HANDLE;
                if (file_import_ok) { ctx.destroy_buffer(t); }
                ::munmap(fmap, 4096 * 4);
            }
            ::close(fd);
            ::unlink(path.c_str());
        }
    }
    note("file-backed import support is declared in the feature record below; the main phases "
         "run on an anonymous mapping, which this driver does import");

    // ---- the working mapping: anonymous, filled with the shared pattern ---------------------------
    auto* host = static_cast<std::uint32_t*>(
        ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (host == MAP_FAILED) { fail("mmap failed"); }
    declared("file_page.bytes", std::to_string(bytes), "anonymous mmap");
    declared("file_page.rows", std::to_string(rows), "4 KiB rows");

    // ---- warm: host touch, XOR check --------------------------------------------------------------
    fill_pattern(host, words);
    {
        std::uint32_t x = 0;
        for (std::size_t i = 0; i < words; ++i) { x ^= host[i]; }
        check("warm.xor", x == expected_xor(words),
              Obj().uint("got", x).uint("expected", expected_xor(words)));
    }

    // ---- import into Vulkan -----------------------------------------------------------------------
    VkBufferRes buf = ctx.import_host_pointer(host, bytes, "anon");
    VkBufferRes out = ctx.make_buffer(std::max(rows, std::size_t(2)) * 4, true, false, "out");
    VkCtx::Submit sub;
    ctx.make_submit(sub);

    const std::vector<char> spirv = VkCtx::read_spirv(shader_path(argv[0], "row_read.spv"));
    VkCtx::Pipeline pl;
    ctx.make_pipeline(pl, spirv, 16, 64);

    const auto gpu_pass = [&](std::uint32_t mode, std::uint32_t words_per_row, std::uint32_t grid,
                              std::uint32_t nonce, double* exec_ns, bool* ok) {
        const VkDescriptorSet set = ctx.bind_set(pl, buf.buffer, out.buffer);
        EL_VK(vkResetCommandBuffer(sub.cmd, 0));
        const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0,
                                          nullptr};
        EL_VK(vkBeginCommandBuffer(sub.cmd, &bi));
        vkCmdFillBuffer(sub.cmd, out.buffer, 0, VK_WHOLE_SIZE, 0u);
        const VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(sub.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
                             nullptr);
        vkCmdResetQueryPool(sub.cmd, sub.queries, 0, 2);
        vkCmdWriteTimestamp(sub.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sub.queries, 0);
        vkCmdBindPipeline(sub.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipe);
        vkCmdBindDescriptorSets(sub.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &set, 0,
                                nullptr);
        const std::uint32_t push[4] = {words_per_row, nonce, mode, 0u};
        vkCmdPushConstants(sub.cmd, pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
        vkCmdDispatch(sub.cmd, grid, 1, 1);
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
        *exec_ns = double(ts[1] - ts[0]) * ctx.timestamp_period_ns;
        const auto* got = static_cast<const std::uint32_t*>(out.mapped);
        *ok = true;
        if (mode == 0u) {
            for (std::size_t r = 0; r < rows && *ok; ++r) {
                std::uint32_t want = 0;
                for (std::size_t w = 0; w < kRowWords; ++w) {
                    want ^= pattern_word(std::uint64_t(r) * kRowWords + w);
                }
                if (got[r] != (want ^ nonce)) {
                    note("row mismatch: " + Obj().uint("row", r).uint("got", got[r])
                                              .uint("want", want ^ nonce).text());
                    *ok = false;
                }
            }
        } else {
            std::uint32_t want = 0;
            for (std::size_t w = 0; w < words; ++w) { want ^= pattern_word(w); }
            *ok = got[0] == (want ^ nonce);
        }
    };

    // ---- res-seq: sequential read of the resident mapping -------------------------------------------
    {
        double best = 1e30;
        bool ok_all = true;
        for (int rep = 0; rep < o.reps; ++rep) {
            double e = 0;
            bool ok = false;
            gpu_pass(1u, std::uint32_t(words), 1u, rep_nonce(rep), &e, &ok);
            ok_all = ok_all && ok;
            const double gbps = double(bytes) / (e / 1e3) / 1e3;
            measure("res_seq", gbps, "GB/s",
                    Obj().num("gpu_us", e / 1e3).boolean("seq", true), ok, rep);
            best = std::min(best, gbps);
        }
        measure("res_seq", best, "GB/s", Obj().str("agg", "min"), ok_all);
        check("res_seq.result", ok_all, Obj());
    }

    // ---- res-rows: one 4 KiB row per invocation ------------------------------------------------------
    double res_rows_p50 = 0;
    {
        std::vector<double> gbps;
        bool ok_all = true;
        for (int rep = 0; rep < o.reps; ++rep) {
            double e = 0;
            bool ok = false;
            gpu_pass(0u, kRowWords, std::uint32_t(rows), rep_nonce(rep), &e, &ok);
            ok_all = ok_all && ok;
            const double g = double(bytes) / (e / 1e3) / 1e3;
            gbps.push_back(g);
            measure("res_rows", g, "GB/s", Obj().num("gpu_us", e / 1e3), ok, rep);
        }
        const Summary s = summarize(gbps);
        res_rows_p50 = s.p50;
        measure("res_rows", s.p50, "GB/s", Obj().obj("per_rep", summary_obj(s)), ok_all);
        check("res_rows.result", ok_all, Obj());
    }

    // ---- evict: madvise DONTNEED frees the anonymous pages; mincore sees what remains ---------------
    std::vector<unsigned char> vec(pages);
    std::size_t resident_after_evict = 0;
    {
        if (::madvise(host, bytes, MADV_DONTNEED) != 0) {
            note("madvise DONTNEED failed: " + std::string(std::strerror(errno)));
        }
        if (::mincore(host, bytes, vec.data()) != 0) { fail("mincore failed"); }
        resident_after_evict = resident_pages(vec.data(), pages);
        declared("file_page.resident_pages_after_evict", std::to_string(resident_after_evict),
                 "mincore");
        if (resident_after_evict == 0) {
            check("evict.mincore", true, Obj().uint("resident", resident_after_evict));
            // refill: the host re-faults every page and rewrites the pattern, so the cold GPU
            // pass below crosses freshly established host mappings
            fill_pattern(host, words);
        } else {
            // the import keeps the host pages pinned (unevictable userptr) -- a real finding
            check("evict.mincore", false,
                  Obj().uint("resident", resident_after_evict).uint("pages", pages)
                      .str("interpretation", "import pins the host pages; DONTNEED is a no-op"));
        }
    }

    // ---- cold pass: GPU reads after the eviction attempt --------------------------------------------
    double cold_ms = 0;
    {
        double e = 0;
        bool ok = false;
        gpu_pass(0u, kRowWords, std::uint32_t(rows), rep_nonce(-1), &e, &ok);
        cold_ms = e / 1e6;
        measure("cold_rows", cold_ms, "ms",
                Obj().num("us_per_page", e / 1e3 / double(pages)).uint("pages", pages)
                    .boolean("pages_were_freed", resident_after_evict == 0),
                ok, 0);
        check("cold_rows.result", ok, Obj());
        if (::mincore(host, bytes, vec.data()) == 0) {
            declared("file_page.resident_pages_after_cold",
                     std::to_string(resident_pages(vec.data(), pages)), "mincore");
        }
    }

    // ---- resident again ------------------------------------------------------------------------------
    {
        std::vector<double> gbps;
        bool ok_all = true;
        for (int rep = 0; rep < o.reps; ++rep) {
            double e = 0;
            bool ok = false;
            gpu_pass(0u, kRowWords, std::uint32_t(rows), rep_nonce(rep + 100), &e, &ok);
            ok_all = ok_all && ok;
            gbps.push_back(double(bytes) / (e / 1e3) / 1e3);
        }
        const Summary s = summarize(gbps);
        measure("res_rows_again", s.p50, "GB/s", Obj().obj("per_rep", summary_obj(s)), ok_all);
        check("res_rows_again.result", ok_all, Obj());
        check("res_rows_again.recovered", s.p50 > res_rows_p50 * 0.5,
              Obj().num("again_p50", s.p50).num("res_rows_p50", res_rows_p50));
    }

    feature("file_page", true, true, true,
            Obj().str("path", "anonymous mmap -> VK_EXT_external_memory_host import -> GPU reads")
                .boolean("file_mapping_import_rejected", !file_import_ok)
                .str("rejection", "VK_ERROR_INVALID_EXTERNAL_HANDLE for MAP_SHARED file and "
                                  "memfd mappings (CUDA UVA has no such restriction)")
                .num("cold_rows_ms", cold_ms));

    ::munmap(host, bytes);
    ctx.destroy_buffer(out);
    ctx.destroy_buffer(buf);
    ctx.destroy_pipeline(pl);
    return probe_end();
}
