// constants: the device's declared facts, each read from every independent source the machine
// exposes, with agreement required. Spec: probes/constants/spec.md.
//
// Every other probe divides a number by one of these facts (clock, memory size, SM count, L2
// size). This probe reads each fact from every independent source and checks that they agree:
// a disagreement is a finding, and a missing source is an honest record, never a guess. The
// probe does no GPU work: it is a set of reads, so there is no launch to verify.

#include "common/echolock.cuh"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace echolock;

namespace {

struct Options {
    std::string against; // --against RUN_DIR, optional
};

Options parse(int argc, char** argv) {
    Args args(argc, argv, "usage: constants [--against RUN_DIR]");
    Options o;
    for (std::string a; args.next(a);) {
        if (a == "--against") { o.against = args.value(a); }
        else { args.usage_error("unknown argument " + a); }
    }
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

std::string trim(const std::string& s) {
    const std::size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) { return ""; }
    const std::size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// The two name sources disagree about the vendor prefix ("NVIDIA X" vs "X"): strip it before
// comparing.
std::string normalize_name(std::string s) {
    s = trim(s);
    for (const char* v : {"NVIDIA ", "AMD ", "Intel "}) {
        if (s.compare(0, std::strlen(v), v) == 0) { s = trim(s.substr(std::strlen(v))); break; }
    }
    return s;
}

// The "Model:" and "Video Memory:" lines of /proc/driver/nvidia/gpus/*/information.
void parse_nvidia_information(const std::string& text, std::string* model,
                              unsigned long long* vram_bytes) {
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.rfind("Model:", 0) == 0) {
            *model = trim(line.substr(6));
        } else if (line.rfind("Video Memory:", 0) == 0) {
            const std::size_t p = line.find_first_of("0123456789");
            if (p != std::string::npos) {
                *vram_bytes = std::strtoull(line.c_str() + p, nullptr, 10) * 1048576ull;
            }
        }
    }
}

// The first /proc/driver/nvidia/gpus/<pci>/ directory in PCI-address order is taken to be
// CUDA device 0 (the runtime enumerates GPUs by PCI address).
std::string nvidia_information_path() {
    std::vector<std::string> dirs;
    if (DIR* d = ::opendir("/proc/driver/nvidia/gpus")) {
        while (dirent* e = ::readdir(d)) {
            if (e->d_name[0] != '.') { dirs.push_back(e->d_name); }
        }
        ::closedir(d);
    }
    if (dirs.empty()) { return ""; }
    std::sort(dirs.begin(), dirs.end());
    return "/proc/driver/nvidia/gpus/" + dirs[0] + "/information";
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
    probe_start("constants",
                Obj().raw("against", o.against.empty() ? "null" : "\"" + json_escape(o.against) + "\""));
    EL_CUDA(cudaSetDevice(0));
    cudaDeviceProp p{};
    EL_CUDA(cudaGetDeviceProperties(&p, 0));

    // ---- the /proc/driver/nvidia source (one read serves the name and memory keys) -------------
    std::string proc_model;
    unsigned long long proc_vram = 0;
    {
        const std::string path = nvidia_information_path();
        if (path.empty()) {
            note("/proc/driver/nvidia/gpus is not present on this machine: the "
                 "/proc/driver/nvidia information source is unavailable");
        } else {
            std::string text;
            if (!read_whole(path, &text)) {
                note("could not read " + path);
            } else {
                parse_nvidia_information(text, &proc_model, &proc_vram);
                if (proc_model.empty()) {
                    note("no parseable \"Model:\" line in " + path);
                }
            }
        }
    }

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

    // ---- constants.device_name ------------------------------------------------------------------
    declared("constants.device_name", "\"" + json_escape(p.name) + "\"", "cudaGetDeviceProperties");
    if (!proc_model.empty()) {
        declared("constants.device_name", "\"" + json_escape(proc_model) + "\"",
                 "/proc/driver/nvidia information");
        check("constants.device_name.agreement",
              normalize_name(p.name) == normalize_name(proc_model),
              Obj().str("cudaGetDeviceProperties", p.name)
                  .str("/proc/driver/nvidia information", proc_model));
    }
    if (proc_model.empty()) {
        feature_sources("device_name", true, 1,
                        Obj().str("absent",
                                  "/proc/driver/nvidia information (not present or unparseable)"));
    }

    // ---- constants.memory_bytes -----------------------------------------------------------------
    {
        const unsigned long long api_bytes = static_cast<unsigned long long>(p.totalGlobalMem);
        declared("constants.memory_bytes", std::to_string(api_bytes), "cudaGetDeviceProperties");
        if (proc_vram != 0) {
            declared("constants.memory_bytes", std::to_string(proc_vram),
                     "/proc/driver/nvidia information");
            check("constants.memory_bytes.agreement", api_bytes == proc_vram,
                  Obj().uint("cudaGetDeviceProperties", api_bytes)
                      .uint("/proc/driver/nvidia information", proc_vram));
        }
        if (proc_vram == 0) {
            feature_sources("memory_bytes", true, 1,
                            Obj().str("absent",
                                      "/proc/driver/nvidia information (not present or unparseable)"));
        }
    }

    // ---- constants.sm_count (single source: always a feature record) ----------------------------
    {
        const unsigned long long sms = static_cast<unsigned long long>(p.multiProcessorCount);
        declared("constants.sm_count", std::to_string(sms), "cudaGetDeviceProperties");
        if (against.have_dir && against.have_clock_sm) {
            declared("constants.sm_count", std::to_string(against.clock_sms),
                     "clock:" + against.run_id);
            check("constants.sm_count.agreement",
                  sms == static_cast<unsigned long long>(against.clock_sms),
                  Obj().uint("cudaGetDeviceProperties", sms)
                      .uint(("clock:" + against.run_id).c_str(), against.clock_sms));
        } else if (against.have_dir) {
            note("clock per-SM coverage: no per-SM clock measures in " + against.run_id +
                 (against.probe.empty() ? "" : " (a " + against.probe + " run)") +
                 "; pass a clock run's directory to use it as a second sm_count source");
        }
        feature_sources("sm_count", true, 1,
                        Obj().str("derived",
                                  "the optional second source is the distinct-SM count of a "
                                  "clock run's per-SM measures, via --against"));
    }

    // ---- constants.l2_bytes (single source: always a feature record) ----------------------------
    {
        const unsigned long long l2 = static_cast<unsigned long long>(p.l2CacheSize);
        declared("constants.l2_bytes", std::to_string(l2), "cudaGetDeviceProperties");
        if (against.have_dir) {
            if (band) {
                check("constants.l2_bytes.agreement", band_low < l2 && l2 < band_high,
                      Obj().uint("band_low_bytes", band_low).uint("band_high_bytes", band_high)
                          .uint("l2_bytes", l2).str("run", against.run_id).str("pattern", "seq")
                          .num("base_ns_per_hop", base_ns)
                          .num("at_inflection_ns_per_hop", inflect_ns));
            } else {
                note("pointer_chase inflection: no footprint in " + against.run_id +
                     " exceeded 2x the smallest footprint's value: no band to compare");
            }
        }
        feature_sources("l2_bytes", true, 1,
                        Obj().str("derived",
                                  "the pointer_chase inflection via --against is a band, not a "
                                  "value; it bounds this key instead of declaring it"));
    }

    // ---- constants.clock_max_mhz (single source: always a feature record) -----------------------
    {
        // clockRate is a device attribute, not a cudaDeviceProp member (see clock.cu).
        const int khz = runtime_attr(cudaDevAttrClockRate, 0);
        if (khz > 0) {
            declared("constants.clock_max_mhz", std::to_string(khz / 1000),
                     "cudaDeviceGetAttribute(ClockRate)");
        } else {
            note("cudaDeviceGetAttribute(ClockRate) unavailable: no clock_max_mhz source");
        }
        feature_sources("clock_max_mhz", true, khz > 0 ? 1 : 0,
                        Obj().str("why_single",
                                  "no independent max-clock source is exposed on the CUDA "
                                  "stack of this probe"));
    }

    return probe_end();
}
