// Shared plumbing for the Vulkan backend: the same JSONL record contract as common/echolock.cuh
// (v1 records, CLOCK_MONOTONIC t_ns, probe/measure/check/declared/feature/note), without any
// CUDA dependency. Exit codes match too: 0 ok, 1 runtime error, 2 usage, 3 verify failed.
//
// The oracle follows the same rule as the CUDA probes: a result slot nobody wrote must not be
// able to pass. The compute shader folds a per-dispatch nonce into its result, and the host
// folds the same nonce into the expectation before comparing.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace echolock {

constexpr int kExitOk = 0, kExitError = 1, kExitUsage = 2, kExitVerifyFailed = 3;

inline std::int64_t mono_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline std::string json_escape(const std::string& s) {
    std::string out;
    for (const unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

class Obj {
  public:
    Obj& str(const char* k, const std::string& v) { return raw(k, "\"" + json_escape(v) + "\""); }
    Obj& num(const char* k, double v) {
        if (!std::isfinite(v)) { return raw(k, "null"); }
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.10g", v);
        return raw(k, buf);
    }
    Obj& integer(const char* k, long long v) { return raw(k, std::to_string(v)); }
    Obj& uint(const char* k, unsigned long long v) { return raw(k, std::to_string(v)); }
    Obj& boolean(const char* k, bool v) { return raw(k, v ? "true" : "false"); }
    Obj& obj(const char* k, const Obj& v) { return raw(k, v.text()); }
    Obj& raw(const char* k, const std::string& json) {
        body_ += (body_.empty() ? "\"" : ",\"") + json_escape(k) + "\":" + json;
        return *this;
    }
    std::string text() const { return "{" + body_ + "}"; }

  private:
    std::string body_;
};

struct Identity {
    std::string probe = "unknown", backend = "vulkan";
    int failed_checks = 0;
};
inline Identity& identity() {
    static Identity id;
    return id;
}

inline void emit(const char* type, const Obj& fields = Obj()) {
    const std::string f = fields.text();
    std::printf("{\"v\":1,\"t_ns\":%lld,\"src\":\"probe\",\"probe\":\"%s\",\"backend\":\"%s\","
                "\"type\":\"%s\"%s%s\n",
                static_cast<long long>(mono_ns()), json_escape(identity().probe).c_str(),
                json_escape(identity().backend).c_str(), type, f.size() > 2 ? "," : "",
                f.size() > 2 ? f.substr(1).c_str() : "}");
    std::fflush(stdout);
}

inline void probe_start(const std::string& probe, const Obj& params) {
    identity().probe = probe;
    emit("probe_start", Obj().obj("params", params));
}

inline int probe_end() {
    const bool ok = identity().failed_checks == 0;
    emit("probe_end", Obj().str("status", ok ? "ok" : "verify_failed")
                          .integer("failed_checks", identity().failed_checks));
    return ok ? kExitOk : kExitVerifyFailed;
}

[[noreturn]] inline void fail(const std::string& message) {
    emit("error", Obj().str("message", message));
    emit("probe_end", Obj().str("status", "error").str("message", message));
    std::fprintf(stderr, "echolock: %s\n", message.c_str());
    std::exit(kExitError);
}

inline void declared(const std::string& key, const std::string& json_value,
                     const std::string& source) {
    emit("declared", Obj().str("key", key).raw("value", json_value).str("source", source));
}

inline void measure(const std::string& name, double value, const std::string& unit,
                    const Obj& tags, bool verified, long long rep = -1) {
    Obj o;
    o.str("name", name).num("value", value).str("unit", unit).obj("tags", tags).boolean(
        "verified", verified);
    if (rep >= 0) { o.integer("rep", rep); }
    emit("measure", o);
}

inline bool check(const std::string& name, bool pass, const Obj& detail = Obj()) {
    if (!pass) { ++identity().failed_checks; }
    emit("check", Obj().str("name", name).boolean("pass", pass).obj("detail", detail));
    return pass;
}

inline void feature(const std::string& name, bool requested, int granted, bool effect_observed,
                    const Obj& evidence) {
    Obj o;
    o.str("feature", name).boolean("requested", requested);
    if (granted < 0) { o.raw("granted", "null"); } else { o.boolean("granted", granted != 0); }
    o.boolean("effect_observed", effect_observed).obj("evidence", evidence);
    emit("feature", o);
}

inline void note(const std::string& message) { emit("note", Obj().str("message", message)); }

// ---- XOR oracle (mirrors common/echolock.cuh) ---------------------------------------------------

inline unsigned rep_nonce(long long rep) { return 0xA5A5A500u ^ static_cast<unsigned>(rep + 1); }

inline unsigned fold_nonce(unsigned expected, unsigned nonce) { return expected ^ nonce; }

// The 16-byte-word generator the CUDA probes fill with (murmur3 fmix32 on the word index),
// so a buffer filled here is checkable the same way.
inline std::uint32_t pattern_word(std::uint64_t w, std::uint32_t seed = 0x45594C4Bu) {
    std::uint32_t h = static_cast<std::uint32_t>(w) ^ static_cast<std::uint32_t>(w >> 32) ^ seed;
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

// Expected XOR of `n` pattern words, computed in parallel on the host.
inline std::uint32_t expected_xor(std::uint64_t n, std::uint32_t seed = 0x45594C4Bu) {
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    const std::uint64_t per = (n + threads - 1) / threads;
    std::vector<std::uint32_t> part(threads, 0);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            std::uint32_t a = 0;
            for (std::uint64_t i = std::uint64_t(t) * per,
                                end = std::min(n, std::uint64_t(t + 1) * per);
                 i < end; ++i) {
                a ^= pattern_word(i, seed);
            }
            part[t] = a;
        });
    }
    for (auto& th : pool) { th.join(); }
    std::uint32_t r = 0;
    for (const std::uint32_t p : part) { r ^= p; }
    return r;
}

struct Summary {
    double min = 0, p50 = 0, p90 = 0, max = 0, mean = 0;
};
inline Summary summarize(std::vector<double> v) {
    Summary s;
    if (v.empty()) { return s; }
    std::sort(v.begin(), v.end());
    const auto at = [&](double q) { return v[std::min(v.size() - 1, std::size_t(q * v.size()))]; };
    double sum = 0;
    for (const double x : v) { sum += x; }
    s.min = v.front();
    s.p50 = at(0.5);
    s.p90 = at(0.9);
    s.max = v.back();
    s.mean = sum / double(v.size());
    return s;
}

inline Obj summary_obj(const Summary& s) {
    return Obj().num("min", s.min).num("p50", s.p50).num("p90", s.p90).num("max", s.max).num(
        "mean", s.mean);
}

inline unsigned hw_threads() { return std::max(1u, std::thread::hardware_concurrency()); }

// Small argument parser: --key value pairs, like the CUDA probes' Args.
class Args {
  public:
    Args(int argc, char** argv, std::string usage) : usage_(std::move(usage)) {
        for (int i = 1; i < argc; ++i) { args_.emplace_back(argv[i]); }
    }
    bool next(std::string& key) {
        if (pos_ >= args_.size()) { return false; }
        key = args_[pos_++];
        if (key == "--help" || key == "-h") {
            std::fprintf(stdout, "%s\n", usage_.c_str());
            std::exit(kExitOk);
        }
        return true;
    }
    std::string value(const std::string& key) {
        if (pos_ >= args_.size()) { usage_error("missing value after " + key); }
        return args_[pos_++];
    }
    [[noreturn]] void usage_error(const std::string& why) const {
        std::fprintf(stderr, "%s\n%s\n", why.c_str(), usage_.c_str());
        std::exit(kExitUsage);
    }

  private:
    std::vector<std::string> args_;
    std::size_t pos_ = 0;
    std::string usage_;
};

} // namespace echolock
