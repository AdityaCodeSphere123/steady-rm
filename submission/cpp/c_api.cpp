#include "rm_engine.h"

#include <cstring>
#include <mutex>
#include <sstream>

namespace {
rm::Engine* g_engine = nullptr;
std::mutex g_m;
std::string g_err;

rm::Engine& eng() { if (!g_engine) g_engine = new rm::Engine(); return *g_engine; }

std::vector<std::string_view> split(const char* s, int n) {
    std::vector<std::string_view> out;
    if (!s || n <= 0) return out;
    out.reserve(n);
    const char* p = s;
    for (int i = 0; i < n; ++i) {
        const char* e = std::strchr(p, '\n');
        if (!e) { out.emplace_back(p); break; }
        out.emplace_back(p, e - p);
        p = e + 1;
    }
    return out;
}

int write_out(const std::vector<rm::Ranked>& r, int32_t* idx, double* sc) {
    for (size_t i = 0; i < r.size(); ++i) { idx[i] = r[i].idx; sc[i] = r[i].score; }
    return static_cast<int>(r.size());
}

std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o + "\"";
}

int copy_buf(const std::string& s, char* buf, int len) {
    if (static_cast<int>(s.size()) + 1 > len) return -static_cast<int>(s.size() + 1);
    std::memcpy(buf, s.c_str(), s.size() + 1);
    return static_cast<int>(s.size());
}
}  // namespace

extern "C" {

int rm_prepare(const char* path) {
    std::lock_guard<std::mutex> g(g_m);
    g_err.clear();
    try { return eng().prepare(path, g_err) ? 0 : -1; }
    catch (const std::exception& e) { g_err = e.what(); return -1; }
}

const char* rm_last_error() { return g_err.c_str(); }

int rm_set_param(const char* name, double v) {
    std::lock_guard<std::mutex> g(g_m);
    return rm::set_param(eng().P, name, v) ? 0 : -1;
}

int rm_stats(char* buf, int len) {
    std::lock_guard<std::mutex> g(g_m);
    return copy_buf(eng().stats_json(), buf, len);
}

int rm_score(const char* query, const char* cands, int nc, int k, int32_t* idx, double* sc) {
    std::lock_guard<std::mutex> g(g_m);
    try { return write_out(eng().score_candidates(query, split(cands, nc), k), idx, sc); }
    catch (const std::exception& e) { g_err = e.what(); return -1; }
}

int rm_feedback(const char* query, const char* seeds, int ns, const char* cands, int nc, int k,
                int32_t* idx, double* sc) {
    std::lock_guard<std::mutex> g(g_m);
    try { return write_out(eng().feedback(query, split(seeds, ns), split(cands, nc), k), idx, sc); }
    catch (const std::exception& e) { g_err = e.what(); return -1; }
}

int rm_relevance_model(const char* query, const char* seeds, int ns, int variant, char* buf, int len) {
    std::lock_guard<std::mutex> g(g_m);
    auto m = eng().relevance_model(query, split(seeds, ns), variant);
    std::ostringstream o;
    o.precision(12);
    o << "[";
    for (size_t i = 0; i < m.size(); ++i) o << (i ? "," : "") << "[" << jstr(m[i].first) << "," << m[i].second << "]";
    o << "]";
    return copy_buf(o.str(), buf, len);
}

}  // extern "C"
