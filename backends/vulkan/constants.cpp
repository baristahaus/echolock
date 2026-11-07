// constants (Vulkan backend): the open-stack equivalent of the CUDA constants probe. Spec:
// probes/constants/spec.md.
//
// Every device fact is read from every independent source the open stack exposes: the Vulkan
// properties, the KFD topology sysfs, and the amdgpu drm sysfs. The selected device's PCI
// device ID (VkPhysicalDeviceProperties.deviceID) is what picks the KFD node (the node whose
// properties device_id is that ID) and the drm card (the one whose PCI device ID is that ID;
// first match wins). A missing path is a note, never a guess.

#include "common/vk_ctx.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace echolock;

namespace {

struct Options {
    int device = 0;
    std::string against; // --against RUN_DIR, optional
};

Options parse(int argc, char** argv) {
    Args args(argc, argv, "usage: constants [--device N (0)] [--against RUN_DIR]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--device") { o.device = std::stoi(args.value(a)); }
        else if (a == "--against") { o.against = args.value(a); }
        else { args.usage_error("unknown argument " + a); }
    }
    if (o.device < 0) { args.usage_error("device must be a non-negative index"); }
    return o;
}

// ---- small access to the JSONL records this toolkit emits ------------------------------------
// Just enough to read another run's probe.jsonl (our own format v1, docs/record-format.md);
// not a general JSON parser.

// The value position for `"key":` (whitespace around the colon allowed); npos when absent.
// The quoted token keeps `"per_rep"` from matching `"rep"`.
std::size_t json_value_pos(const std::string& line, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    std::size_t p = 0;
    while ((p = line.find(pat, p)) != std::string::npos) {
        std::size_t c = p + pat.size();
        while (c < line.size() && std::isspace(static_cast<unsigned char>(line[c]))) { ++c; }
        if (c < line.size() && line[c] == ':') { return c + 1; }
        p += pat.size();
    }
    return std::string::npos;
}

bool json_field_str(const std::string& line, const std::string& key, std::string* out) {
    std::size_t p = json_value_pos(line, key);
    if (p == std::string::npos) { return false; }
    while (p < line.size() && std::isspace(static_cast<unsigned char>(line[p]))) { ++p; }
    if (p >= line.size() || line[p] != '"') { return false; }
    ++p;
    std::string v;
    for (; p < line.size(); ++p) {
        const char c = line[p];
        if (c == '\\' && p + 1 < line.size()) { v += line[++p]; continue; }
        if (c == '"') { break; }
        v += c;
    }
    if (p >= line.size()) { return false; }
    *out = v;
    return true;
}

bool json_field_num(const std::string& line, const std::string& key, double* out) {
    std::size_t p = json_value_pos(line, key);
    if (p == std::string::npos) { return false; }
    while (p < line.size() && std::isspace(static_cast<unsigned char>(line[p]))) { ++p; }
    char* end = nullptr;
    const double v = std::strtod(line.c_str() + p, &end);
    if (end == line.c_str() + p) { return false; }
    *out = v;
    return true;
}

// ---- host file helpers -------------------------------------------------------------------------

bool read_whole(const std::string& path, std::string* out) {
    std::ifstream f(path);
    if (!f) { return false; }
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

// A "key value" line from a KFD topology properties file (decimal values).
bool prop_u64(const std::string& text, const std::string& key, unsigned long long* out) {
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.rfind(key + " ", 0) == 0) {
            *out = std::strtoull(line.c_str() + key.size() + 1, nullptr, 10);
            return true;
        }
    }
    return false;
}

// ---- KFD topology: the node matching the device's PCI device ID --------------------------------

struct KfdMatch {
    bool found = false;
    unsigned long long vram_bytes = 0;
    bool have_vram = false;
    long long cu_count = -1;
};

KfdMatch kfd_match(unsigned int pci_device_id) {
    KfdMatch m;
    for (int node = 0; node < 64; ++node) {
        const std::string dir = "/sys/class/kfd/kfd/topology/nodes/" + std::to_string(node);
        std::string props;
        if (!read_whole(dir + "/properties", &props)) { continue; }
        unsigned long long vendor_id = 0, device_id = 0;
        // The CPU node has vendor_id 0; the GPU nodes carry their PCI device ID.
        if (!prop_u64(props, "vendor_id", &vendor_id) || vendor_id == 0) { continue; }
        if (!prop_u64(props, "device_id", &device_id) || device_id != pci_device_id) { continue; }
        m.found = true;
        // VRAM: the mem bank with heap_type 1 (KFD_HEAP_TYPE_VRAM).
        for (int bank = 0; bank < 8; ++bank) {
            std::string bp;
            if (!read_whole(dir + "/mem_banks/" + std::to_string(bank) + "/properties", &bp)) { break; }
            unsigned long long type = 0, size = 0;
            if (prop_u64(bp, "heap_type", &type) && type == 1 && prop_u64(bp, "size_in_bytes", &size)) {
                m.vram_bytes += size;
                m.have_vram = true;
            }
        }
        // CU count: the cu_count file when the kernel exposes it, else the product the KFD
        // topology derives it from.
        std::string cc;
        if (read_whole(dir + "/cu_count", &cc)) {
            const std::size_t p = cc.find_first_of("0123456789");
            if (p != std::string::npos) { m.cu_count = std::strtoll(cc.c_str() + p, nullptr, 10); }
        } else {
            unsigned long long arrays = 0, cu_per = 0;
            if (prop_u64(props, "array_count", &arrays) && prop_u64(props, "cu_per_simd_array", &cu_per)) {
                m.cu_count = static_cast<long long>(arrays) * static_cast<long long>(cu_per);
            }
        }
        break;
    }
    return m;
}

// ---- drm: the card matching the device's PCI device ID ------------------------------------------

struct DrmMatch {
    bool found = false;
    unsigned long long vram_bytes = 0;
    bool have_vram = false;
    unsigned long long sclk_mhz = 0;
    bool have_sclk = false;
};

DrmMatch drm_match(unsigned int pci_device_id) {
    DrmMatch m;
    for (int card = 0; card < 16; ++card) {
        const std::string dev = "/sys/class/drm/card" + std::to_string(card) + "/device";
        std::string id_text;
        if (!read_whole(dev + "/device", &id_text)) { continue; } // "0x7550"
        const std::size_t p = id_text.find_first_not_of(" \t\r\n");
        if (p == std::string::npos) { continue; }
        if (std::strtoul(id_text.c_str() + p, nullptr, 16) != pci_device_id) { continue; }
        m.found = true;
        std::string text;
        if (read_whole(dev + "/mem_info_vram_total", &text)) {
            const std::size_t p = text.find_first_of("0123456789");
            if (p != std::string::npos) {
                m.vram_bytes = std::strtoull(text.c_str() + p, nullptr, 10);
                m.have_vram = true;
            }
        }
        if (read_whole(dev + "/pp_dpm_sclk", &text)) {
            // "2: 2520Mhz " -- the highest level of the whole table, not the active one.
            std::istringstream ss(text);
            std::string line;
            while (std::getline(ss, line)) {
                const std::size_t colon = line.find(':');
                if (colon == std::string::npos) { continue; }
                const std::size_t d = line.find_first_of("0123456789", colon + 1);
                if (d == std::string::npos || line.find("Mhz", d) == std::string::npos) { continue; }
                const unsigned long long v = std::strtoull(line.c_str() + d, nullptr, 10);
                if (!m.have_sclk || v > m.sclk_mhz) { m.sclk_mhz = v; m.have_sclk = true; }
            }
        }
        break;
    }
    return m;
}

// ---- derived sources from an existing run ------------------------------------------------------

struct Against {
    bool have_dir = false;
    std::string run_id, probe;
    std::vector<std::pair<unsigned long long, double>> chase; // (footprint_bytes, ns_per_hop)
    int clock_sms = 0;
    bool have_clock_sm = false;
};

Against read_against(const std::string& dir) {
    Against a;
    std::string manifest;
    read_whole(dir + "/manifest.json", &manifest);
    std::string rid;
    if (json_field_str(manifest, "run_id", &rid)) { a.run_id = rid; }
    else {
        const std::size_t p = dir.find_last_of('/');
        a.run_id = p == std::string::npos ? dir : dir.substr(p + 1);
    }
    json_field_str(manifest, "probe", &a.probe);

    std::ifstream f(dir + "/probe.jsonl");
    if (!f) { return a; }
    a.have_dir = true;
    std::vector<bool> sm_seen(1024, false);
    std::string line;
    while (std::getline(f, line)) {
        std::string rtype;
        if (!json_field_str(line, "type", &rtype) || rtype != "measure") { continue; }
        std::string name;
        json_field_str(line, "name", &name);
        if (name == "chase_hop_latency") {
            std::string kind, pattern;
            json_field_str(line, "kind", &kind);
            json_field_str(line, "pattern", &pattern);
            // The inflection uses the device-local kind and the seq pattern (the spec's
            // documented choice), from summary records only (tags.per_rep, no top-level rep).
            if (pattern != "seq" || (kind != "device" && kind != "vulkan_device_local")) { continue; }
            if (line.find("\"per_rep\"") == std::string::npos) { continue; }
            double v = 0, fp = 0;
            if (json_field_num(line, "rep", &fp) || !json_field_num(line, "value", &v)) { continue; }
            unsigned long long bytes = 0;
            if (json_field_num(line, "footprint_mib", &fp)) {
                bytes = static_cast<unsigned long long>(fp * 1048576.0);
            } else if (json_field_num(line, "footprint_bytes", &fp)) {
                bytes = static_cast<unsigned long long>(fp);
            } else { continue; }
            a.chase.emplace_back(bytes, v);
        } else if (name == "derived_sm_clock_mhz" || name == "clock_trace") {
            double sm = -1;
            if (json_field_num(line, "sm", &sm) && sm >= 0 && sm < 1024) {
                const int i = static_cast<int>(sm);
                if (!sm_seen[i]) {
                    sm_seen[i] = true;
                    ++a.clock_sms;
                    a.have_clock_sm = true;
                }
            }
        }
    }
    return a;
}

// Inflection rule (spec): footprints ascending; the first footprint whose value exceeds 2x
// the smallest footprint's value. The band is (the footprint strictly before it, that
// footprint).
bool inflection_band(const Against& a, unsigned long long* low, unsigned long long* high,
                     double* base_ns, double* at_inflection_ns) {
    if (a.chase.size() < 2) { return false; }
    auto c = a.chase; // sorted copy
    std::sort(c.begin(), c.end());
    const double base = c.front().second;
    for (std::size_t i = 1; i < c.size(); ++i) {
        if (c[i].second > 2.0 * base) {
            *low = c[i - 1].first;
            *high = c[i].first;
            *base_ns = base;
            *at_inflection_ns = c[i].second;
            return true;
        }
    }
    return false;
}

// The feature record for one key's source coverage. effect_observed is null: this documents
// what could not be cross-checked, it is not a fallback verdict.
void feature_sources(const char* key, bool requested, int granted, const Obj& evidence) {
    emit("feature",
         Obj().str("feature", std::string("constants.") + key + ".sources")
             .boolean("requested", requested)
             .boolean("granted", granted != 0)
             .raw("effect_observed", "null")
             .obj("evidence", evidence));
}

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    VkCtx ctx("constants", o.device,
              Obj().integer("device", o.device)
                  .raw("against", o.against.empty() ? "null" : "\"" + json_escape(o.against) + "\""));
    ctx.declare_common();

    const unsigned int pci_id = ctx.props.deviceID;
    char pci_id_hex[8];
    std::snprintf(pci_id_hex, sizeof(pci_id_hex), "0x%x", pci_id);
    note("matching the KFD and drm sysfs sources by PCI device ID " + std::string(pci_id_hex) +
         " (vkGetPhysicalDeviceProperties deviceID)");

    // ---- the derived sources, from an existing run when --against was given --------------------
    const Against against = o.against.empty() ? Against() : read_against(o.against);
    unsigned long long band_low = 0, band_high = 0;
    double base_ns = 0, inflect_ns = 0;
    const bool band = inflection_band(against, &band_low, &band_high, &base_ns, &inflect_ns);
    if (!o.against.empty() && !against.have_dir) {
        note("--against " + o.against +
             ": no probe.jsonl (wrong or missing run dir): no derived sources");
    } else if (o.against.empty()) {
        note("derived source pointer_chase inflection: skipped, --against RUN_DIR not given");
        note("derived source clock per-SM coverage: skipped, --against RUN_DIR not given");
    }

    // ---- constants.device_name (single source: always a feature record) ------------------------
    declared("constants.device_name", "\"" + json_escape(ctx.props.deviceName) + "\"",
             "vkGetPhysicalDeviceProperties");
    feature_sources("device_name", true, 1,
                    Obj().str("why_single",
                              "the KFD node name is a different format (gfx-family) and not a "
                              "comparable source"));

    // ---- constants.memory_bytes ------------------------------------------------------------------
    unsigned long long heaps = 0;
    for (std::uint32_t i = 0; i < ctx.mem_props.memoryHeapCount; ++i) {
        if (ctx.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            heaps += ctx.mem_props.memoryHeaps[i].size;
        }
    }
    declared("constants.memory_bytes", std::to_string(heaps), "Vulkan memory heaps");

    const KfdMatch kfd = kfd_match(pci_id);
    const bool kfd_present = kfd.found && kfd.have_vram;
    if (kfd_present) {
        declared("constants.memory_bytes", std::to_string(kfd.vram_bytes), "KFD sysfs");
    } else if (kfd.found) {
        note("KFD sysfs: no VRAM mem bank on the matching node");
    }

    const DrmMatch drm = drm_match(pci_id);
    const bool drm_present = drm.found && drm.have_vram;
    if (!kfd.found) {
        note("KFD sysfs: no /sys/class/kfd node matches PCI device ID " + std::string(pci_id_hex) +
             ": no KFD memory or CU source");
    }
    if (drm_present) {
        declared("constants.memory_bytes", std::to_string(drm.vram_bytes),
                 "amdgpu sysfs mem_info_vram_total");
    } else if (drm.found) {
        note("amdgpu sysfs: mem_info_vram_total missing on the matching card");
    }
    if (!drm.found) {
        note("amdgpu sysfs: no /sys/class/drm card matches PCI device ID " +
             std::string(pci_id_hex) + ": no VRAM or sclk source");
    }

    {
        // UMA detection: on an integrated device the device-local heap (and the KFD VRAM bank)
        // size the host memory pool, while mem_info_vram_total is the small framebuffer
        // carve-out; on a discrete GPU all three sources size the same memory and must agree.
        const bool uma = (drm_present && heaps > drm.vram_bytes) ||
                         (kfd_present && kfd.vram_bytes > heaps);
        if (uma) {
            note("integrated device: the Vulkan device-local heap (" + std::to_string(heaps) +
                 " B)" +
                 (kfd_present ? " and the KFD VRAM bank (" + std::to_string(kfd.vram_bytes) +
                                   " B)"
                              : "") +
                 " size the host memory pool, while amdgpu mem_info_vram_total (" +
                 std::to_string(drm.vram_bytes) +
                 " B) is the framebuffer carve-out: the sources measure different pools, so no "
                 "agreement check is applied");
            feature_sources("memory_bytes", true, 1,
                            Obj().str("why",
                                      "integrated device: the carve-out and the host memory "
                                      "pool are different quantities, so the sources are not "
                                      "comparable"));
        } else {
            std::vector<std::pair<std::string, unsigned long long>> vals;
            vals.emplace_back("Vulkan memory heaps", heaps);
            if (kfd_present) { vals.emplace_back("KFD sysfs", kfd.vram_bytes); }
            if (drm_present) { vals.emplace_back("amdgpu sysfs mem_info_vram_total", drm.vram_bytes); }
            if (vals.size() >= 2) {
                Obj d;
                bool ok = true;
                for (const auto& sv : vals) {
                    d.raw(sv.first.c_str(), std::to_string(sv.second)); // numeric value in detail
                    if (sv.second != vals.front().second) { ok = false; }
                }
                check("constants.memory_bytes.agreement", ok, d);
            }
            if (vals.size() < 3) { // the spec lists three sources for this key
                std::string missing;
                if (!kfd_present) { missing += "KFD node or VRAM bank missing; "; }
                if (!drm_present) { missing += "drm card or mem_info_vram_total missing"; }
                feature_sources("memory_bytes", true, static_cast<int>(vals.size()),
                                Obj().str("absent", missing));
            }
        }
    }

    // ---- constants.sm_count (single source: always a feature record) ----------------------------
    {
        int granted = 0;
        if (kfd.found && kfd.cu_count > 0) {
            declared("constants.sm_count", std::to_string(kfd.cu_count), "KFD sysfs");
            granted = 1;
        }
        if (against.have_dir && against.have_clock_sm) {
            declared("constants.sm_count", std::to_string(against.clock_sms),
                     "clock:" + against.run_id);
            if (granted == 1) {
                check("constants.sm_count.agreement",
                      kfd.cu_count == static_cast<long long>(against.clock_sms),
                      Obj().integer("KFD sysfs", kfd.cu_count)
                          .uint(("clock:" + against.run_id).c_str(), against.clock_sms));
            }
        } else if (against.have_dir) {
            note("clock per-SM coverage: no per-SM clock measures in " + against.run_id +
                 (against.probe.empty() ? "" : " (a " + against.probe + " run)") +
                 "; pass a clock run's directory to use it as a second sm_count source");
        }
        feature_sources("sm_count", true, granted,
                        Obj().str("derived",
                                  "the optional second source is the distinct-SM count of a "
                                  "clock run's per-SM measures, via --against"));
    }

    // ---- constants.l2_bytes (no direct source: always a feature record) --------------------------
    {
        if (against.have_dir) {
            if (band) {
                note("pointer_chase L2 inflection band from " + against.run_id +
                     " (device-local kind, pattern seq): " + std::to_string(band_low) +
                     " B < L2 < " + std::to_string(band_high) + " B (base " +
                     std::to_string(static_cast<long long>(base_ns)) + " ns/hop, " +
                     std::to_string(static_cast<long long>(inflect_ns)) +
                     " ns/hop at the inflection); data for the future, not a check");
            } else {
                note("pointer_chase inflection: no footprint in " + against.run_id +
                     " exceeded 2x the smallest footprint's value: no band");
            }
        }
        feature_sources("l2_bytes", false, band ? 1 : 0,
                        Obj().str("why",
                                  "no direct L2 source on this stack; the --against "
                                  "pointer_chase inflection is a band, not a value"));
    }

    // ---- constants.clock_max_mhz (single source: always a feature record) ------------------------
    {
        int granted = 0;
        if (drm.found && drm.have_sclk) {
            declared("constants.clock_max_mhz", std::to_string(drm.sclk_mhz), "amdgpu sysfs sclk");
            granted = 1;
        } else if (drm.found) {
            note("amdgpu sysfs: no parseable pp_dpm_sclk table on the matching card");
        }
        feature_sources("clock_max_mhz", true, granted,
                        Obj().str("why_single",
                                  "the pp_dpm_sclk table is the only max-clock source on this "
                                  "stack"));
    }

    return probe_end();
}
