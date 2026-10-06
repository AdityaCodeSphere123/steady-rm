// Flat C interface for ctypes. Doc-id lists cross the boundary as one
// '\n'-joined byte string; results come back as indices into the caller's
// candidate list plus scores, so no strings are returned for rankings.
#include "rm_engine.h"

#include <cstring>
#include <mutex>
#include <sstream>

namespace {
rm::Engine* g_engine = nullptr;
std::mutex g_m;               // calls are serialised; ctypes drops the GIL
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

int rm_get_param(const char* name, double* v) {
    std::lock_guard<std::mutex> g(g_m);
    return rm::get_param(eng().P, name, *v) ? 0 : -1;
}

int rm_set_param_a(const char* name, double v) {
    std::lock_guard<std::mutex> g(g_m);
    return rm::set_param(eng().PA, name, v) ? 0 : -1;
}

int rm_get_param_a(const char* name, double* v) {
    std::lock_guard<std::mutex> g(g_m);
    return rm::get_param(eng().PA, name, *v) ? 0 : -1;
}

int rm_score_a(const char* query, const char* cands, int nc, int k, int32_t* idx, double* sc) {
    std::lock_guard<std::mutex> g(g_m);
    try { return write_out(eng().score_track_a(query, split(cands, nc), k), idx, sc); }
    catch (const std::exception& e) { g_err = e.what(); return -1; }
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

// JSON with everything one feedback call derived (report / oral defence).
int rm_explain(const char* query, const char* seeds, int ns, const char* cands, int nc, char* buf, int len) {
    std::lock_guard<std::mutex> g(g_m);
    rm::Explain ex;
    auto cv = split(cands, nc);
    if (cv.empty()) cv.emplace_back("");
    eng().feedback(query, split(seeds, ns), cv, 1, &ex);
    std::ostringstream o;
    o.precision(10);
    o << "{\"query_terms\":[";
    for (size_t i = 0; i < ex.qterms.size(); ++i) o << (i ? "," : "") << jstr(ex.qterms[i]);
    o << "],\"seed_weights\":[";
    for (size_t i = 0; i < ex.seed_weights.size(); ++i)
        o << (i ? "," : "") << "[" << jstr(ex.seed_weights[i].first) << "," << ex.seed_weights[i].second << "]";
    o << "],\"model\":[";
    for (size_t i = 0; i < ex.rm.size(); ++i) o << (i ? "," : "") << "[" << jstr(ex.rm[i].first) << "," << ex.rm[i].second << "]";
    o << "],\"lambda\":" << ex.lambda << ",\"coherence\":" << ex.coherence << ",\"clarity\":" << ex.clarity
      << ",\"mu\":" << ex.mu << ",\"qpp\":" << ex.qpp << "}";
    return copy_buf(o.str(), buf, len);
}

// The estimated relevance model alone: variant 1 = RM1, 2 = RM2.
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

int rm_analyze(const char* query, char* buf, int len) {
    std::lock_guard<std::mutex> g(g_m);
    auto t = eng().analyze_query(query);
    std::string o = "[";
    for (size_t i = 0; i < t.size(); ++i) o += (i ? "," : "") + jstr(t[i]);
    return copy_buf(o + "]", buf, len);
}

}  // extern "C"
