#include "rm_engine.h"
#include "analysis.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <atomic>
#include <memory>

namespace rm {

// ---------------------------------------------------------------------------
// Parameters by name
// ---------------------------------------------------------------------------
#define RM_PARAMS(X) \
    X(stemmer) X(doc_stop) X(query_stop) X(qstop_min_len) X(qstop_list) X(smoothing) X(mu) \
    X(mu_factor) X(mu_min) X(mu_max) X(jm_lambda) X(w_od) X(w_uw) X(uw_window) X(prior_w) \
    X(prior_k) X(rm_variant) X(rm2_mix) X(rm2_uniform_prior) X(fb_mu) X(fb_docs) X(fb_terms) \
    X(beta) X(seed_sdm) X(trim_floor) X(anchor_min) X(term_support) X(per_doc_cap) X(min_cf) \
    X(idf_power) X(agree_gamma) X(lambda) X(lambda_min) X(lambda_max) X(coh_w) X(clarity_scale) \
    X(fb_sdm) X(fb_prior_w) X(fb_rm_w) X(fb_mode) X(fb_loo) X(fb_seed_ql) X(fb_scale) X(fb_corr_pow) X(par_min_tokens) X(lead_w) X(lead_len) X(lead_mu) X(lam_qlen_slope) \
    X(lam_qlen_pivot) X(qlen_shrink) X(qpp_w) X(qpp_lo) X(qpp_hi) \
    X(mu_mode) X(mu_loo_scale) X(mu_loo_docs) X(len_prior) X(small_seed_n) X(longq_n) \
    X(bm25_w) X(bm25_k1) X(bm25_b) X(bm25_delta) X(bm25_z) X(pool_lambda) X(pool_docs) \
    X(ranker) X(ax_b) X(ax_k) X(ax_c) X(ax_w) X(qspell)

bool set_param(Params& p, const std::string& n, double v) {
#define X(f) if (n == #f) { p.f = v; return true; }
    RM_PARAMS(X)
#undef X
    return false;
}
bool get_param(const Params& p, const std::string& n, double& v) {
#define X(f) if (n == #f) { v = p.f; return true; }
    RM_PARAMS(X)
#undef X
    return false;
}

// ---------------------------------------------------------------------------
// Tokeniser (same rules as A1's analysis.cpp each_surface): lowercase ASCII
// alnum runs, dotted initials glued (u.s. -> us), length-1 tokens dropped.
// Used identically for documents and queries.
// ---------------------------------------------------------------------------
static uint8_t T_ALNUM[256], T_LOWER[256];
static const bool T_INIT = [] {
    for (int i = 0; i < 256; ++i) {
        bool a = (i >= '0' && i <= '9') || (i >= 'a' && i <= 'z') || (i >= 'A' && i <= 'Z');
        T_ALNUM[i] = a;
        T_LOWER[i] = (i >= 'A' && i <= 'Z') ? static_cast<uint8_t>(i + 32) : static_cast<uint8_t>(i);
    }
    return true;
}();

template <typename Fn>
static void each_surface(std::string_view text, Fn&& emit) {
    (void)T_INIT;
    char tok[256];
    size_t tn = 0;
    bool overflow = false;
    std::string acro;
    auto flush_acro = [&] { if (acro.size() >= 2) emit(std::string_view(acro)); acro.clear(); };
    auto take = [&] {
        if (tn == 0 && !overflow) return;
        if (tn == 1 && tok[0] >= 'a' && tok[0] <= 'z') acro.push_back(tok[0]);
        else { flush_acro(); if (tn > 1) emit(std::string_view(tok, tn)); }
        tn = 0; overflow = false;
    };
    for (unsigned char c : text) {
        if (T_ALNUM[c]) {
            if (tn < sizeof(tok)) tok[tn++] = static_cast<char>(T_LOWER[c]);
            else overflow = true;  // absurdly long run: truncated
        } else {
            take();
            if (c != '.') flush_acro();
        }
    }
    take();
    flush_acro();
}

static std::unordered_set<std::string> stem_set(const std::unordered_set<std::string>& words, uint8_t st) {
    std::unordered_set<std::string> out;
    for (const auto& w : words) { out.insert(w); out.insert(apply_stem(w, st)); }
    return out;
}

// ---------------------------------------------------------------------------
// Scratch (one per worker thread)
// ---------------------------------------------------------------------------
struct Engine::Scratch {
    std::vector<uint32_t> cnt;      // per-term counter, all zero between uses
    std::vector<uint32_t> touched;
    std::vector<std::pair<uint32_t, uint16_t>> hits;  // (position, query slot)
    explicit Scratch(size_t V) : cnt(V, 0) {}
};

struct Engine::Query {
    std::vector<std::string> strs;          // analysed terms, in order (in-vocab only)
    std::vector<uint32_t> terms;            // tids, with duplicates
    std::vector<uint32_t> uniq, uniq_cnt;   // distinct tids and multiplicity
    std::vector<uint16_t> slot_of_term;     // terms[i] -> slot in uniq
    std::vector<std::pair<uint16_t, uint16_t>> pairs;  // adjacent (slot a, slot b)
    std::vector<std::pair<uint64_t, uint64_t>> pair_cf;  // (cf ordered, cf unordered)
    size_t len() const { return terms.size(); }
};

Engine::Engine() {
    unsigned hw = std::thread::hardware_concurrency();
    int w = hw == 0 ? 2 : static_cast<int>(hw);
    if (const char* e = std::getenv("RM_THREADS")) w = std::atoi(e);
    pool_.reset(new ThreadPool(std::min(w, MAX_THREADS)));
}
Engine::~Engine() = default;

// ---------------------------------------------------------------------------
// prepare(): one parallel pass over the corpus
// ---------------------------------------------------------------------------
namespace {
struct Shard {
    StrTable surf{1 << 14};                 // surface form -> local term or DROP
    std::vector<uint32_t> surf_to_local;
    StrTable local{1 << 14};                // stem -> local term id
    std::vector<uint64_t> cf;
    std::vector<uint32_t> df, stamp;
    std::vector<uint32_t> toks;
    std::vector<uint64_t> doc_end;          // cumulative token count per doc
    std::vector<std::string> ids;
    size_t bad_lines = 0;
};
}  // namespace

bool Engine::prepare(const std::string& path_in, std::string& err) {
    auto t0 = std::chrono::steady_clock::now();
    // RM_TIMING=1 prints a per-phase breakdown to stderr.
    const bool timing = std::getenv("RM_TIMING") && std::getenv("RM_TIMING")[0] == '1';
    auto mark = t0;
    auto phase = [&](const char* name) {
        if (!timing) return;
        auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[prepare] %-12s %.3fs\n", name, std::chrono::duration<double>(now - mark).count());
        mark = now;
    };
    std::string path = path_in;
    struct stat st{};
    if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) path += "/corpus.jsonl";
    const double sig = P.stemmer * 10 + P.doc_stop;
    if (ready_ && path == path_ && sig == tok_cfg_sig_) return true;  // idempotent

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "cannot open " + path; return false; }
    if (fstat(fd, &st) != 0) { close(fd); err = "cannot stat " + path; return false; }
    size_t size = static_cast<size_t>(st.st_size);
    const char* data = nullptr;
    void* map = nullptr;
    if (size > 0) {
        map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map == MAP_FAILED) { close(fd); err = "mmap failed"; return false; }
        madvise(map, size, MADV_SEQUENTIAL);
        data = static_cast<const char*>(map);
    }
    close(fd);

    const uint8_t stemmer = static_cast<uint8_t>(P.stemmer);
    const bool doc_stop = P.doc_stop > 0;
    const int W = pool_->workers();
    // Split into ~4 chunks per worker at line boundaries (dynamic balance).
    // Chunk count depends on the file only, never on W: term ids (and so every
    // tie-break) are then identical whether 1 or 4 threads ran prepare().
    (void)W;
    const size_t n_chunks = size == 0 ? 0 : std::min<size_t>(256, std::max<size_t>(16, size >> 22));
    std::vector<size_t> cut(n_chunks + 1, 0);
    for (size_t c = 1; c < n_chunks; ++c) {
        size_t pos = std::max(cut[c - 1], size * c / n_chunks);
        const void* nl = pos < size ? memchr(data + pos, '\n', size - pos) : nullptr;
        cut[c] = nl ? static_cast<size_t>(static_cast<const char*>(nl) - data) + 1 : size;
    }
    if (n_chunks) cut[n_chunks] = size;
    std::vector<Shard> shards(n_chunks);

    pool_->run(n_chunks, [&](size_t c, int) {
        Shard& sh = shards[c];
        JsonDoc doc;
        std::string surf_str;
        uint32_t docno = 0;
        const char* p = data + cut[c];
        const char* end = data + cut[c + 1];
        while (p < end) {
            const char* nl = static_cast<const char*>(memchr(p, '\n', end - p));
            const char* le = nl ? nl : end;
            const char* ls = p;
            p = nl ? nl + 1 : end;
            while (ls < le && (*ls == ' ' || *ls == '\r' || *ls == '\t')) ++ls;
            if (ls == le) continue;
            if (!parse_doc_line(ls, le, doc)) { ++sh.bad_lines; continue; }
            ++docno;
            each_surface(doc.text, [&](std::string_view s) {
                uint64_t h = hash_str(s);
                bool ins = false;
                uint32_t sid = sh.surf.insert_h(s, h, &ins);
                if (ins) {
                    uint32_t lid = StrTable::NONE;
                    surf_str.assign(s);
                    if (!(doc_stop && INDEX_STOPWORDS.count(surf_str))) {
                        const std::string& stem = apply_stem(surf_str, stemmer);
                        bool ins2 = false;
                        lid = sh.local.insert(stem, &ins2);
                        if (ins2) { sh.cf.push_back(0); sh.df.push_back(0); sh.stamp.push_back(0); }
                    }
                    sh.surf_to_local.push_back(lid);
                }
                uint32_t lid = sh.surf_to_local[sid];
                if (lid == StrTable::NONE) return;
                sh.toks.push_back(lid);
                ++sh.cf[lid];
                if (sh.stamp[lid] != docno) { sh.stamp[lid] = docno; ++sh.df[lid]; }
            });
            sh.doc_end.push_back(sh.toks.size());
            sh.ids.emplace_back(doc.id);
        }
        // The surface-form cache is only needed while tokenising; free it now
        // so peak memory is not (number of chunks) x (chunk vocabulary).
        sh.surf = StrTable(16);
        std::vector<uint32_t>().swap(sh.surf_to_local);
        std::vector<uint32_t>().swap(sh.stamp);
    });
    if (map) munmap(map, size);
    phase("tokenise");

    // Merge local vocabularies into the global one (deterministic order).
    vocab_ = StrTable(1 << 16);
    cf_.clear(); df_.clear();
    std::vector<std::vector<uint32_t>> remap(n_chunks);
    size_t N = 0, T = 0;
    for (size_t c = 0; c < n_chunks; ++c) {
        Shard& sh = shards[c];
        remap[c].resize(sh.local.size());
        for (uint32_t l = 0; l < sh.local.size(); ++l) {
            bool ins = false;
            uint32_t g = vocab_.insert_h(sh.local.key(l), sh.local.hash_of(l), &ins);
            if (ins) { cf_.push_back(0); df_.push_back(0); }
            cf_[g] += sh.cf[l];
            df_[g] += sh.df[l];
            remap[c][l] = g;
        }
        N += sh.ids.size();
        T += sh.toks.size();
    }
    std::vector<size_t> doc_base(n_chunks + 1, 0), tok_base(n_chunks + 1, 0);
    for (size_t c = 0; c < n_chunks; ++c) {
        doc_base[c + 1] = doc_base[c] + shards[c].ids.size();
        tok_base[c + 1] = tok_base[c] + shards[c].toks.size();
    }
    toks_.assign(T, 0);
    tok_off_.assign(N + 1, 0);
    pool_->run(n_chunks, [&](size_t c, int) {
        Shard& sh = shards[c];
        const auto& rmp = remap[c];
        uint32_t* out = toks_.data() + tok_base[c];
        for (size_t i = 0; i < sh.toks.size(); ++i) out[i] = rmp[sh.toks[i]];
        for (size_t d = 0; d < sh.doc_end.size(); ++d) tok_off_[doc_base[c] + d + 1] = tok_base[c] + sh.doc_end[d];
        std::vector<uint32_t>().swap(sh.toks);
    });
    phase("merge+remap");
    docids_ = StrTable(N + 16);
    docid_to_doc_.clear();
    doc_key_.assign(N, 0);
    size_t bad = 0;
    for (size_t c = 0; c < n_chunks; ++c) {
        bad += shards[c].bad_lines;
        for (size_t d = 0; d < shards[c].ids.size(); ++d) {
            bool ins = false;
            uint32_t key = docids_.insert(shards[c].ids[d], &ins);
            doc_key_[doc_base[c] + d] = key;
            if (ins) docid_to_doc_.push_back(static_cast<int32_t>(doc_base[c] + d));  // first wins
        }
    }
    shards.clear();
    total_ = T;
    avgdl_ = N ? static_cast<double>(T) / N : 0.0;

    // Term flags for query / expansion-term stopping, on stemmed forms.
    auto long_s = stem_set(STOPWORDS, stemmer), short_s = stem_set(INDEX_STOPWORDS, stemmer);
    term_flags_.assign(vocab_.size(), 0);
    std::string tmp;
    for (uint32_t t = 0; t < vocab_.size(); ++t) {
        tmp.assign(vocab_.key(t));
        if (long_s.count(tmp)) term_flags_[t] |= 1;
        if (short_s.count(tmp)) term_flags_[t] |= 2;
    }

    scratch_.clear();
    for (int w = 0; w < pool_->workers(); ++w) scratch_.emplace_back(new Scratch(vocab_.size()));
    pair_cf_.clear();
    qmark_.assign(vocab_.size(), 0xFFFF);
    phase("docids+flags");
    build_postings();
    phase("postings");
    estimate_mu_loo();
    phase("mu_loo");
    path_ = path;
    tok_cfg_sig_ = sig;
    ready_ = true;
    prepare_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (N == 0 && bad) { err = "no parseable documents"; }
    return true;
}

std::string Engine::stats_json() const {
    std::ostringstream o;
    o << "{\"docs\":" << n_docs() << ",\"vocab\":" << vocab_.size() << ",\"tokens\":" << total_
      << ",\"avgdl\":" << avgdl_ << ",\"mu\":" << mu_eff() << ",\"mu_loo\":" << mu_loo_ << ",\"threads\":" << pool_->workers()
      << ",\"prepare_seconds\":" << prepare_seconds_ << "}";
    return o.str();
}

// Leave-one-out log likelihood of the collection under Dirichlet smoothing:
//   L(mu) = sum_D sum_w c(w,D) log((c(w,D) - 1 + mu p(w|C)) / (|D| - 1 + mu))
// L is maximised where dL/dmu = 0:
//   g(mu) = sum_D sum_w c(w,D) [ p(w|C)/(c-1+mu p) - 1/(|D|-1+mu) ]
// g is found by bisection on log(mu) over an evenly spaced document sample.
void Engine::estimate_mu_loo() {
    mu_loo_ = 0;
    const size_t N = n_docs();
    if (N == 0 || total_ == 0) return;
    const size_t S = std::min<size_t>(N, static_cast<size_t>(std::max(100.0, P.mu_loo_docs)));
    struct Row { float c, p; };
    const int W = pool_->workers();
    std::vector<std::vector<Row>> rows(W);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> docs(W);   // (row end, |D|)
    std::vector<size_t> sample(S);
    for (size_t i = 0; i < S; ++i) sample[i] = i * N / S;
    const double T = static_cast<double>(total_);
    // One task per worker, each taking a contiguous slice of the sample.
    pool_->run(W, [&](size_t w, int wk) {
        Scratch& sc = *scratch_[wk];
        for (size_t i = S * w / W; i < S * (w + 1) / W; ++i) {
            auto b = bag(static_cast<int32_t>(sample[i]), sc);
            uint32_t dl = static_cast<uint32_t>(tok_off_[sample[i] + 1] - tok_off_[sample[i]]);
            if (dl < 2) continue;
            for (auto& [t, c] : b) rows[w].push_back({static_cast<float>(c), static_cast<float>(cf_[t] / T)});
            docs[w].push_back({static_cast<uint32_t>(rows[w].size()), dl});
        }
    });
    auto g = [&](double mu) {
        std::vector<double> part(W, 0.0);
        pool_->run(W, [&](size_t w, int) {
            double acc = 0; uint32_t r = 0;
            for (auto& [end, dl] : docs[w]) {
                const double inv = 1.0 / (dl - 1 + mu);
                for (; r < end; ++r) {
                    const double c = rows[w][r].c, p = rows[w][r].p;
                    acc += c * (p / (c - 1 + mu * p) - inv);
                }
            }
            part[w] = acc;
        });
        double t = 0; for (double x : part) t += x; return t;
    };
    double lo = std::log(1.0), hi = std::log(1e6);
    if (g(std::exp(lo)) <= 0) { mu_loo_ = 1.0; return; }
    if (g(std::exp(hi)) >= 0) { mu_loo_ = 1e6; return; }
    for (int it = 0; it < 50; ++it) {
        double mid = 0.5 * (lo + hi);
        (g(std::exp(mid)) > 0 ? lo : hi) = mid;
    }
    mu_loo_ = std::exp(0.5 * (lo + hi));
}

double Engine::mu_eff() const {
    if (P.mu > 0) return P.mu;
    if (P.mu_mode >= 1 && mu_loo_ > 0)
        return std::min(P.mu_max, std::max(P.mu_min, P.mu_loo_scale * mu_loo_));
    double m = P.mu_factor * avgdl_;
    if (!(m > 0)) return 1000.0;
    return std::min(P.mu_max, std::max(P.mu_min, m));
}

int32_t Engine::doc_index(std::string_view id) const {
    uint32_t i = docids_.find(id);
    return i == StrTable::NONE ? -1 : docid_to_doc_[i];
}

static void edits1(const std::string& w, std::vector<std::string>& out) {
    static const char az[] = "abcdefghijklmnopqrstuvwxyz";
    const size_t n = w.size();
    out.clear();
    out.reserve(n * 54 + 26);
    for (size_t i = 0; i < n; ++i) {   // deletes
        std::string e = w; e.erase(i, 1); out.push_back(std::move(e));
    }
    for (size_t i = 0; i + 1 < n; ++i) {   // transposes
        std::string e = w; std::swap(e[i], e[i + 1]); out.push_back(std::move(e));
    }
    for (size_t i = 0; i < n; ++i) {   // replaces
        for (int c = 0; c < 26; ++c) {
            if (az[c] == w[i]) continue;
            std::string e = w; e[i] = az[c]; out.push_back(std::move(e));
        }
    }
    for (size_t i = 0; i <= n; ++i) {   // inserts
        for (int c = 0; c < 26; ++c) {
            std::string e = w; e.insert(e.begin() + static_cast<std::ptrdiff_t>(i), az[c]);
            out.push_back(std::move(e));
        }
    }
}

uint32_t Engine::resolve_query_term(const std::string& surface, uint8_t stemmer) const {
    const std::string& stem = apply_stem(surface, stemmer);
    uint32_t t = vocab_.find(stem);
    const bool rare = t != StrTable::NONE && df_[t] <= 2;
    if (t != StrTable::NONE && !rare) return t;
    if (P.qspell < 1 || surface.size() < 4) return t;
    auto best_of = [&](const std::vector<std::string>& cands, uint32_t cur) {
        uint32_t best = cur;
        uint64_t best_cf = cur == StrTable::NONE ? 2 : cf_[cur];
        const uint64_t floor_cf = cur == StrTable::NONE ? 2 : std::max<uint64_t>(cf_[cur] * 8, 8);
        for (const auto& e : cands) {
            if (e == surface) continue;
            const std::string& st = apply_stem(e, stemmer);
            uint32_t u = vocab_.find(st);
            if (u == StrTable::NONE || u == cur) continue;
            if (cf_[u] > best_cf && cf_[u] >= floor_cf) { best = u; best_cf = cf_[u]; }
        }
        return best;
    };
    std::vector<std::string> e1;
    edits1(surface, e1);
    uint32_t best = best_of(e1, t);
    if ((best == t || best == StrTable::NONE) && P.qspell >= 2 && surface.size() >= 5) {
        std::vector<std::string> e2, tmp;
        // Dist-2 is large; only expand deletes + transposes of the surface.
        e2.reserve(e1.size() * 8);
        for (size_t i = 0; i < surface.size(); ++i) {
            std::string d = surface; d.erase(i, 1);
            edits1(d, tmp);
            e2.insert(e2.end(), tmp.begin(), tmp.end());
        }
        for (size_t i = 0; i + 1 < surface.size(); ++i) {
            std::string tr = surface; std::swap(tr[i], tr[i + 1]);
            edits1(tr, tmp);
            e2.insert(e2.end(), tmp.begin(), tmp.end());
        }
        best = best_of(e2, best == StrTable::NONE ? t : best);
    }
    return best;
}

// ---------------------------------------------------------------------------
// Query analysis
// ---------------------------------------------------------------------------
Engine::Query Engine::parse_query(std::string_view qtext) {
    const uint8_t stemmer = static_cast<uint8_t>(P.stemmer);
    std::vector<std::string> surf;
    each_surface(qtext, [&](std::string_view s) { surf.emplace_back(s); });
    const auto& stops = P.qstop_list >= 3 ? LUCENE_MINUS_WH
                       : P.qstop_list >= 2 ? MINIMAL_STOPWORDS
                       : P.qstop_list > 0 ? STOPWORDS
                       : INDEX_STOPWORDS;
    bool drop = P.query_stop >= 2 || (P.query_stop >= 1 && surf.size() >= P.qstop_min_len);
    std::vector<std::string> kept;
    if (drop) for (auto& s : surf) if (!stops.count(s)) kept.push_back(s);
    if (kept.empty()) kept = surf;   // all-stopword query: keep everything
    if (P.doc_stop > 0) {
        std::vector<std::string> k2;
        for (auto& s : kept) if (!INDEX_STOPWORDS.count(s)) k2.push_back(s);
        kept.swap(k2);
    }
    Query q;
    int prev = -1;  // slot of previous in-vocab term; OOV breaks adjacency
    for (auto& s : kept) {
        uint32_t t = resolve_query_term(s, stemmer);
        if (t == StrTable::NONE) { prev = -1; continue; }  // OOV: constant for every doc
        auto it = std::find(q.uniq.begin(), q.uniq.end(), t);
        uint16_t slot;
        if (it == q.uniq.end()) { slot = static_cast<uint16_t>(q.uniq.size()); q.uniq.push_back(t); q.uniq_cnt.push_back(0); }
        else slot = static_cast<uint16_t>(it - q.uniq.begin());
        ++q.uniq_cnt[slot];
        q.terms.push_back(t);
        q.slot_of_term.push_back(slot);
        q.strs.push_back(std::string(vocab_.key(t)));
        if (prev >= 0) q.pairs.push_back({static_cast<uint16_t>(prev), slot});
        prev = slot;
        if (q.uniq.size() >= 0xFFF0) break;
    }
    q.pair_cf.assign(q.pairs.size(), {0, 0});
    return q;
}

std::vector<std::string> Engine::analyze_query(std::string_view query) { return parse_query(query).strs; }

// Marks the query's terms in the dense qmark_ array (term -> slot) for the
// lifetime of one call, so doc_features() classifies each token in O(1).
struct Engine::Marks {
    Engine& e; const Query& q;
    Marks(Engine& en, const Query& qq) : e(en), q(qq) {
        for (size_t u = 0; u < q.uniq.size(); ++u) e.qmark_[q.uniq[u]] = static_cast<uint16_t>(u);
    }
    ~Marks() { for (uint32_t t : q.uniq) e.qmark_[t] = 0xFFFF; }
};

// ---------------------------------------------------------------------------
// Proximity counting. hits are (position, slot), sorted by position.
//   ordered   #1(a b): a at i and b at i+1
//   unordered #uwW(a b): an a-occurrence with a b within distance < W
// The same function counts documents and (summed) the whole collection, so
// document and background statistics are defined identically.
// ---------------------------------------------------------------------------
static void count_pairs(const std::vector<std::pair<uint32_t, uint16_t>>& hits,
                        const std::vector<std::pair<uint16_t, uint16_t>>& pairs, uint32_t W,
                        uint64_t* od, uint64_t* uw) {
    const size_t H = hits.size();
    for (size_t p = 0; p < pairs.size(); ++p) {
        const uint16_t a = pairs[p].first, b = pairs[p].second;
        for (size_t i = 0; i < H; ++i) {
            if (hits[i].second != a) continue;
            const uint32_t pos = hits[i].first;
            if (i + 1 < H && hits[i + 1].first == pos + 1 && hits[i + 1].second == b) ++od[p];
            bool found = false;
            for (size_t j = i + 1; j < H && hits[j].first < pos + W; ++j)
                if (hits[j].second == b && hits[j].first != pos) { found = true; break; }
            for (size_t j = i; !found && j-- > 0 && hits[j].first + W > pos;)
                if (hits[j].second == b && hits[j].first != pos) found = true;
            if (found) ++uw[p];
        }
    }
}

// Document lists per term (doc ids only), built in parallel: every
// (term, doc) pair is written once through a per-term atomic cursor. List
// order is not sorted, which is fine: lists are only ever summed over.
void Engine::build_postings() {
    post_off_.clear(); post_.clear();
    const size_t V = vocab_.size(), N = n_docs();
    uint64_t total = 0;
    for (uint32_t d : df_) total += d;
    if (V == 0 || total > (1ull << 28)) return;     // > 1 GB: fall back to corpus scans (8 GB VM)
    post_off_.assign(V + 1, 0);
    for (size_t t = 0; t < V; ++t) post_off_[t + 1] = post_off_[t] + df_[t];
    post_.assign(total, 0);
    std::unique_ptr<std::atomic<uint64_t>[]> cur(new std::atomic<uint64_t>[V]);
    for (size_t t = 0; t < V; ++t) cur[t].store(post_off_[t], std::memory_order_relaxed);
    const int W = pool_->workers();
    std::vector<std::vector<uint32_t>> stamp(W, std::vector<uint32_t>(V, 0));
    const size_t chunks = std::min<size_t>(1024, std::max<size_t>(1, N / 256));
    pool_->run(chunks, [&](size_t c, int w) {
        auto& st = stamp[w];
        for (size_t d = N * c / chunks; d < N * (c + 1) / chunks; ++d) {
            const uint32_t mark = static_cast<uint32_t>(d + 1);
            for (uint64_t i = tok_off_[d]; i < tok_off_[d + 1]; ++i) {
                const uint32_t t = toks_[i];
                if (st[t] == mark) continue;
                st[t] = mark;
                post_[cur[t].fetch_add(1, std::memory_order_relaxed)] = static_cast<uint32_t>(d);
            }
        }
    });
}

// Collection counts of #1(a b) and #uwW(a b) for the query's adjacent pairs.
// Only documents containing the rarer term of a pair can contribute, so each
// pair walks that term's document list (split into parallel tasks) instead
// of the whole corpus. Cached by term pair: pure collection statistics.
void Engine::ensure_pair_stats(Query& q) {
    if (q.pairs.empty() || (P.w_od <= 0 && P.w_uw <= 0)) return;
    const uint32_t W = static_cast<uint32_t>(std::max(2.0, P.uw_window));
    std::lock_guard<std::mutex> g(pair_m_);
    if (pair_window_ != W) { pair_cf_.clear(); pair_window_ = W; }
    std::vector<size_t> missing;
    auto key_of = [&](size_t p) {
        return (static_cast<uint64_t>(q.uniq[q.pairs[p].first]) << 32) | q.uniq[q.pairs[p].second];
    };
    for (size_t p = 0; p < q.pairs.size(); ++p) {
        auto it = pair_cf_.find(key_of(p));
        if (it != pair_cf_.end()) q.pair_cf[p] = it->second;
        else missing.push_back(p);
    }
    if (missing.empty()) return;
    const size_t N = n_docs();
    const bool have_post = !post_off_.empty();
    // Task = (pair index, first doc slot, last doc slot) over the rarer term's
    // list, or over the whole corpus when there is no inverted list.
    struct Task { size_t m; uint64_t b, e; };
    std::vector<Task> tasks;
    const uint64_t step = 4096;
    for (size_t m = 0; m < missing.size(); ++m) {
        const size_t p = missing[m];
        uint32_t a = q.uniq[q.pairs[p].first], b = q.uniq[q.pairs[p].second];
        uint32_t r = df_[a] <= df_[b] ? a : b;
        uint64_t lo = have_post ? post_off_[r] : 0, hi = have_post ? post_off_[r + 1] : N;
        for (uint64_t x = lo; x < hi; x += step) tasks.push_back({m, x, std::min(hi, x + step)});
    }
    std::vector<uint64_t> od(tasks.size(), 0), uw(tasks.size(), 0);
    pool_->run(tasks.size(), [&](size_t ti, int) {
        const Task& tk = tasks[ti];
        const size_t p = missing[tk.m];
        const uint32_t a = q.uniq[q.pairs[p].first], b = q.uniq[q.pairs[p].second];
        const std::vector<std::pair<uint16_t, uint16_t>> pr{{0, static_cast<uint16_t>(a == b ? 0 : 1)}};
        std::vector<std::pair<uint32_t, uint16_t>> hits;
        uint64_t lod = 0, luw = 0;
        for (uint64_t x = tk.b; x < tk.e; ++x) {
            const size_t d = have_post ? post_[x] : x;
            const uint32_t* t = toks_.data() + tok_off_[d];
            const uint32_t n = static_cast<uint32_t>(tok_off_[d + 1] - tok_off_[d]);
            hits.clear();
            for (uint32_t i = 0; i < n; ++i) {
                if (t[i] == a) hits.push_back({i, 0});
                else if (t[i] == b) hits.push_back({i, 1});
            }
            if (hits.size() >= 2) count_pairs(hits, pr, W, &lod, &luw);
        }
        od[ti] = lod; uw[ti] = luw;
    });
    std::vector<std::pair<uint64_t, uint64_t>> sum(missing.size(), {0, 0});
    for (size_t ti = 0; ti < tasks.size(); ++ti) { sum[tasks[ti].m].first += od[ti]; sum[tasks[ti].m].second += uw[ti]; }
    for (size_t m = 0; m < missing.size(); ++m) {
        q.pair_cf[missing[m]] = sum[m];
        pair_cf_[key_of(missing[m])] = sum[m];
    }
}

// ---------------------------------------------------------------------------
// Smoothed document language model
// ---------------------------------------------------------------------------
double Engine::long_query_scale(const Query& q) const {
    if (P.longq_n <= 0 || q.len() <= P.longq_n) return 1.0;
    return P.longq_n / static_cast<double>(q.len());
}

double Engine::log_p(uint32_t tf, uint32_t dlen, double pc) const {
    const double mu = mu_eff();
    const double dir = (static_cast<double>(tf) + mu * pc) / (static_cast<double>(dlen) + mu);
    if (P.smoothing >= 2) {
        // Two-stage: (1-λ) Dirichlet + λ P(w|C). Not the same as JM (smoothing=1).
        const double l = std::min(0.95, std::max(0.0, P.jm_lambda));
        return std::log(std::max(1e-18, (1.0 - l) * dir + l * pc));
    }
    if (P.smoothing >= 1) {
        const double l = P.jm_lambda;
        double ml = dlen ? static_cast<double>(tf) / dlen : 0.0;
        return std::log((1 - l) * ml + l * pc);
    }
    return std::log(dir);
}

Engine::DocFeat Engine::doc_features(const Query& q, int32_t d, Scratch& s) const {
    DocFeat f;
    const size_t U = q.uniq.size();
    std::vector<uint32_t> tfv(2 * U, 0);
    uint32_t* tf = tfv.data();
    uint32_t* tfl = tf + U;   // occurrences inside the lead window
    const uint32_t lead_len = static_cast<uint32_t>(std::max(0.0, P.lead_len));
    uint32_t dlen = 0;
    s.hits.clear();
    if (d >= 0) {
        const uint32_t* t = toks_.data() + tok_off_[d];
        dlen = static_cast<uint32_t>(tok_off_[d + 1] - tok_off_[d]);
        for (uint32_t i = 0; i < dlen; ++i) {
            const uint16_t u = qmark_[t[i]];     // query slot, or 0xFFFF
            if (u == 0xFFFF) continue;
            ++tf[u];
            if (i < lead_len) ++tfl[u];
            s.hits.push_back({i, u});
        }
    }
    const double T = static_cast<double>(total_);
    const double lw = P.lead_w * long_query_scale(q);
    const double ll = std::min<double>(dlen, lead_len), lmu = P.lead_mu;
    if (P.ranker >= 1) {
        // F2EXP: tfn = tf log(1 + c avgdl/|D|), then tfn/(tfn+b) * ((N+1)/df)^k
        const double N = std::max(1.0, static_cast<double>(n_docs()));
        const double avg = avgdl_ > 0 ? avgdl_ : 1.0;
        const double b = P.ax_b > 0 ? P.ax_b : 0.5;
        const double k = P.ax_k > 0 ? P.ax_k : 0.35;
        const double c = P.ax_c > 0 ? P.ax_c : 1.0;
        auto f2 = [&](uint32_t tfn_raw, double dl, uint32_t tid) {
            const double df = std::max(1.0, static_cast<double>(df_[tid]));
            const double tfn = static_cast<double>(tfn_raw) * std::log(1.0 + c * avg / std::max(1.0, dl));
            const double idf = std::pow((N + 1.0) / df, k);
            return (tfn / (tfn + b)) * idf;
        };
        for (size_t u = 0; u < U; ++u) {
            const uint32_t tid = q.uniq[u];
            double s = f2(tf[u], dlen, tid);
            if (lw > 0) s = (1.0 - lw) * s + lw * f2(tfl[u], std::max(1.0, ll), tid);
            f.uni += q.uniq_cnt[u] * s;
            if (tf[u]) f.matched += q.uniq_cnt[u];
        }
    } else {
    for (size_t u = 0; u < U; ++u) {
        const double pc = cf_[q.uniq[u]] / T;
        double lp = log_p(tf[u], dlen, pc);
        if (lw > 0) lp = std::log((1 - lw) * std::exp(lp) + lw * (tfl[u] + lmu * pc) / (ll + lmu));
        f.uni += q.uniq_cnt[u] * lp;
        if (tf[u]) f.matched += q.uniq_cnt[u];
    }
    }
    if (!q.pairs.empty() && (P.w_od > 0 || P.w_uw > 0)) {
        const uint32_t W = static_cast<uint32_t>(std::max(2.0, P.uw_window));
        std::vector<uint64_t> od(q.pairs.size(), 0), uw(q.pairs.size(), 0);
        if (s.hits.size() >= 2) count_pairs(s.hits, q.pairs, W, od.data(), uw.data());
        for (size_t p = 0; p < q.pairs.size(); ++p) {
            double pco = (q.pair_cf[p].first ? q.pair_cf[p].first : 0.5) / T;
            double pcu = (q.pair_cf[p].second ? q.pair_cf[p].second : 0.5) / T;
            f.od += log_p(static_cast<uint32_t>(od[p]), dlen, pco);
            f.uw += log_p(static_cast<uint32_t>(uw[p]), dlen, pcu);
        }
    }
    return f;
}

// Query part of the score. With sdm=false this is sum_q log P(q|D).
double Engine::qpart(const Query&, const DocFeat& f, bool sdm) const {
    return sdm ? f.uni + P.w_od * f.od + P.w_uw * f.uw : f.uni;
}

// Scores every doc in `docs`:
//   model == nullptr:  qpart (SDM)                           [Track A]
//   otherwise:         lam * qpart/|Q| + (1-lam) * sum_w M(w) log P(w|D)
// The RM sum is split into a document-independent background term plus a
// correction over the model terms the document actually contains, so a
// document costs O(|D|) however many expansion terms there are.
std::vector<double> Engine::score_pool(const Query& q, const std::vector<int32_t>& docs,
                                       const std::vector<std::pair<uint32_t, double>>* model,
                                       double lam, bool sdm) {
    const size_t n = docs.size();
    std::vector<double> out(n, 0.0);
    const double T = static_cast<double>(total_), mu = mu_eff();
    const bool jm = P.smoothing >= 1 && P.smoothing < 2;
    const bool two_stage = P.smoothing >= 2;
    const double ts_l = std::min(0.95, std::max(0.0, P.jm_lambda));
    std::vector<double>& mw = mw_;       // dense model weights, all zero between calls
    double msum = 0, base_const = 0;
    // Background part, log P(w|D) of a term absent from D:
    //   Dirichlet   log(mu pc) - log(|D| + mu)
    //   JM          log(l pc)
    //   two-stage   log(pc) + log((1-l) mu / (|D| + mu) + l)
    // (two-stage used the Dirichlet form before, which added a spurious
    // -log(|D|+mu)-like length penalty to the expansion score.)
    if (model) {
        if (mw.size() != vocab_.size()) mw.assign(vocab_.size(), 0.0);
        for (auto& [t, w] : *model) {
            mw[t] = w;
            msum += w;
            double pc = cf_[t] / T;
            base_const += w * (jm ? std::log(P.jm_lambda * pc) : two_stage ? std::log(pc) : std::log(mu * pc));
        }
    }
    const double qn = q.len() ? static_cast<double>(q.len()) : 1.0;
    const double lq_scale = long_query_scale(q);
    auto one = [&](size_t i, Scratch& s) {
        const int32_t d = docs[i];
        DocFeat f = doc_features(q, d, s);
        double qs = qpart(q, f, sdm);
        const double lprior = P.len_prior != 0 && d >= 0
            ? P.len_prior * lq_scale * std::log(1.0 + static_cast<double>(tok_off_[d + 1] - tok_off_[d])) : 0.0;
        if (!model) { out[i] = qs + lprior; return; }
        double rmp = base_const;
        uint32_t dlen = 0;
        if (d >= 0) {
            const uint32_t* t = toks_.data() + tok_off_[d];
            dlen = static_cast<uint32_t>(tok_off_[d + 1] - tok_off_[d]);
            s.touched.clear();
            for (uint32_t k = 0; k < dlen; ++k) { uint32_t w = t[k]; if (mw[w] > 0) { if (s.cnt[w]++ == 0) s.touched.push_back(w); } }
            for (uint32_t w : s.touched) {
                double pc = cf_[w] / T;
                rmp += mw[w] * (log_p(s.cnt[w], dlen, pc) - log_p(0, dlen, pc));
                s.cnt[w] = 0;
            }
        }
        if (two_stage) rmp += msum * std::log((1.0 - ts_l) * mu / (dlen + mu) + ts_l);
        else if (!jm) rmp -= msum * std::log(dlen + mu);
        out[i] = lam * qs / qn + (1 - lam) * rmp + lprior / qn;
    };
    uint64_t work = 0;
    for (int32_t d : docs) if (d >= 0) work += tok_off_[d + 1] - tok_off_[d];
    if (work >= P.par_min_tokens && pool_->workers() > 1 && n > 8) {
        const size_t chunks = std::min<size_t>(n, pool_->workers() * 4);
        pool_->run(chunks, [&](size_t c, int w) {
            for (size_t i = n * c / chunks; i < n * (c + 1) / chunks; ++i) one(i, *scratch_[w]);
        });
    } else {
        for (size_t i = 0; i < n; ++i) one(i, *scratch_[0]);
    }
    if (model) for (auto& x : *model) mw[x.first] = 0.0;
    return out;
}

// sum_w P(w|R) log P(w|D) for each doc, term by term (no decomposition):
// the expansion score of RM3 / the negative cross-entropy of RM1.
std::vector<double> Engine::rm_pool(const std::vector<int32_t>& docs,
                                    const std::vector<std::pair<uint32_t, double>>& model) const {
    const double T = static_cast<double>(total_);
    std::unordered_map<uint32_t, uint32_t> slot;
    for (size_t j = 0; j < model.size(); ++j) slot.emplace(model[j].first, static_cast<uint32_t>(j));
    std::vector<double> out(docs.size(), 0.0);
    std::vector<uint32_t> tf(model.size());
    for (size_t i = 0; i < docs.size(); ++i) {
        const int32_t d = docs[i];
        std::fill(tf.begin(), tf.end(), 0);
        uint32_t dlen = 0;
        if (d >= 0) {
            const uint32_t* t = toks_.data() + tok_off_[d];
            dlen = static_cast<uint32_t>(tok_off_[d + 1] - tok_off_[d]);
            for (uint32_t k = 0; k < dlen; ++k) {
                auto it = slot.find(t[k]);
                if (it != slot.end()) ++tf[it->second];
            }
        }
        double s = 0;
        for (size_t j = 0; j < model.size(); ++j)
            s += model[j].second * log_p(tf[j], dlen, cf_[model[j].first] / T);
        out[i] = s;
    }
    return out;
}

std::vector<double> Engine::bm25_pool(const Query& q, const std::vector<int32_t>& docs) const {
    const size_t n = docs.size(), U = q.uniq.size();
    std::vector<double> out(n, 0.0);
    if (!U) return out;
    const double N = std::max(1.0, static_cast<double>(n_docs()));
    const double avg = avgdl_ > 0 ? avgdl_ : 1.0;
    const double k1 = P.bm25_k1 > 0 ? P.bm25_k1 : 1.2;
    const double b = std::min(1.0, std::max(0.0, P.bm25_b));
    const double delta = std::max(0.0, P.bm25_delta);
    std::vector<double> idf(U);
    for (size_t u = 0; u < U; ++u) {
        const double df = static_cast<double>(df_[q.uniq[u]]);
        idf[u] = std::log(1.0 + (N - df + 0.5) / (df + 0.5));
    }
    std::vector<uint32_t> tf(U, 0);
    for (size_t i = 0; i < n; ++i) {
        const int32_t d = docs[i];
        if (d < 0) continue;
        const uint32_t* t = toks_.data() + tok_off_[d];
        const uint32_t dlen = static_cast<uint32_t>(tok_off_[d + 1] - tok_off_[d]);
        std::fill(tf.begin(), tf.end(), 0);
        for (uint32_t k = 0; k < dlen; ++k) {
            const uint16_t u = qmark_[t[k]];
            if (u != 0xFFFF) ++tf[u];
        }
        const double K = k1 * (1.0 - b + b * static_cast<double>(dlen) / avg);
        double s = 0;
        for (size_t u = 0; u < U; ++u) {
            if (!tf[u]) continue;
            const double f = static_cast<double>(tf[u]);
            s += q.uniq_cnt[u] * idf[u] * ((f * (k1 + 1.0)) / (f + K) + delta);
        }
        out[i] = s;
    }
    return out;
}

std::vector<double> Engine::f2exp_pool(const Query& q, const std::vector<int32_t>& docs) const {
    const size_t n = docs.size(), U = q.uniq.size();
    std::vector<double> out(n, 0.0);
    if (!U) return out;
    const double N = std::max(1.0, static_cast<double>(n_docs()));
    const double avg = avgdl_ > 0 ? avgdl_ : 1.0;
    const double b = P.ax_b > 0 ? P.ax_b : 0.5;
    const double k = P.ax_k > 0 ? P.ax_k : 0.35;
    const double c = P.ax_c > 0 ? P.ax_c : 1.0;
    const double lead_len = std::max(1.0, P.lead_len);
    const double lw = P.lead_w * long_query_scale(q);
    auto f2 = [&](double tfn_raw, double dl, size_t u) {
        const double df = std::max(1.0, static_cast<double>(df_[q.uniq[u]]));
        const double tfn = tfn_raw * std::log(1.0 + c * avg / std::max(1.0, dl));
        return (tfn / (tfn + b)) * std::pow((N + 1.0) / df, k);
    };
    std::vector<uint32_t> tf(U, 0), tfl(U, 0);
    for (size_t i = 0; i < n; ++i) {
        const int32_t d = docs[i];
        if (d < 0) continue;
        const uint32_t* t = toks_.data() + tok_off_[d];
        const uint32_t dlen = static_cast<uint32_t>(tok_off_[d + 1] - tok_off_[d]);
        std::fill(tf.begin(), tf.end(), 0);
        std::fill(tfl.begin(), tfl.end(), 0);
        for (uint32_t p = 0; p < dlen; ++p) {
            const uint16_t u = qmark_[t[p]];
            if (u == 0xFFFF) continue;
            ++tf[u];
            if (p < lead_len) ++tfl[u];
        }
        const double ll = std::min<double>(dlen, lead_len);
        double s = 0;
        for (size_t u = 0; u < U; ++u) {
            double v = f2(static_cast<double>(tf[u]), dlen, u);
            if (lw > 0) v = (1.0 - lw) * v + lw * f2(static_cast<double>(tfl[u]), std::max(1.0, ll), u);
            s += q.uniq_cnt[u] * v;
        }
        out[i] = s;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Ranking helpers
// ---------------------------------------------------------------------------
static void zscore(std::vector<double>& v) {
    if (v.empty()) return;
    double m = 0; for (double x : v) m += x; m /= v.size();
    double var = 0; for (double x : v) var += (x - m) * (x - m); var /= v.size();
    double sd = std::sqrt(var);
    for (double& x : v) x = sd < 1e-12 ? 0.0 : (x - m) / sd;
}

std::vector<double> Engine::fuse_scores(const std::vector<double>& score, double prior_w) const {
    const size_t n = score.size();
    std::vector<double> fused = score;
    if (prior_w > 0 && n > 1) {
        zscore(fused);
        std::vector<double> pr(n);
        for (size_t i = 0; i < n; ++i) pr[i] = 1.0 / (P.prior_k + i);
        zscore(pr);
        for (size_t i = 0; i < n; ++i) fused[i] += prior_w * pr[i];
    }
    return fused;
}

std::vector<Ranked> Engine::fuse_rank(std::vector<double>& score, const std::vector<std::string_view>& ids,
                                      double prior_w, int k) const {
    return top_k(fuse_scores(score, prior_w), ids, k);
}

std::vector<Ranked> Engine::top_k(const std::vector<double>& fused, const std::vector<std::string_view>& ids,
                                  int k) const {
    const size_t n = fused.size();
    std::vector<int32_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = static_cast<int32_t>(i);
    auto cmp = [&](int32_t a, int32_t b) {
        if (fused[a] != fused[b]) return fused[a] > fused[b];
        return ids[a] < ids[b];   // deterministic doc_id tie-break
    };
    size_t kk = std::min<size_t>(n, k < 0 ? 0 : static_cast<size_t>(k));
    std::partial_sort(order.begin(), order.begin() + kk, order.end(), cmp);
    std::vector<Ranked> out;
    for (size_t i = 0; i < kk; ++i) out.push_back({order[i], fused[order[i]]});
    return out;
}

std::vector<Ranked> Engine::pool_order(const std::vector<std::string_view>& ids, int k) const {
    std::vector<Ranked> out;
    for (size_t i = 0; i < ids.size() && static_cast<int>(i) < k; ++i) out.push_back({static_cast<int32_t>(i), -static_cast<double>(i)});
    return out;
}

// ---------------------------------------------------------------------------
// Track A
// ---------------------------------------------------------------------------
std::vector<Ranked> Engine::score_candidates(std::string_view query, const std::vector<std::string_view>& cands, int k) {
    if (cands.empty() || k <= 0) return {};
    Query q = parse_query(query);
    if (q.terms.empty()) return pool_order(cands, k);
    ensure_pair_stats(q);
    Marks mk(*this, q);
    std::vector<int32_t> docs(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) docs[i] = doc_index(cands[i]);
    return top_k(track_a(q, cands, docs), cands, k);
}

// The final Track A score of every candidate (before sorting). Shared by
// score_candidates() and feedback(fb_mode=1), so Track B starts from the
// exact same numbers.
std::vector<double> Engine::track_a(const Query& q, const std::vector<std::string_view>& cands,
                                    const std::vector<int32_t>& docs) {
    std::vector<double> s = score_pool(q, docs, nullptr, 1.0, true);
    if (P.pool_lambda < 1.0 && P.pool_lambda >= 0.0) {
        const size_t nd = std::min(cands.size(), static_cast<size_t>(std::max(1.0, P.pool_docs)));
        std::vector<std::string_view> seeds(cands.begin(), cands.begin() + static_cast<std::ptrdiff_t>(nd));
        auto seed = seed_weights(q, seeds, *scratch_[0]);
        if (!seed.empty()) {
            auto model = estimate(q, seed, static_cast<int>(P.rm_variant), *scratch_[0]);
            if (!model.empty())
                s = score_pool(q, docs, &model, P.pool_lambda, true);
        }
    }
    add_lexical(q, docs, s);
    return fuse_scores(s, P.prior_w);
}

void Engine::add_lexical(const Query& q, const std::vector<int32_t>& docs, std::vector<double>& s) const {
    if ((P.bm25_w <= 0 && P.ax_w <= 0) || docs.size() <= 1) return;
    if (P.bm25_z >= 1.0) {
        zscore(s);
        if (P.bm25_w > 0) {
            auto bm = bm25_pool(q, docs);
            zscore(bm);
            for (size_t i = 0; i < s.size(); ++i) s[i] += P.bm25_w * bm[i];
        }
        if (P.ax_w > 0) {
            auto ax = f2exp_pool(q, docs);
            zscore(ax);
            for (size_t i = 0; i < s.size(); ++i) s[i] += P.ax_w * ax[i];
        }
    } else {
        if (P.bm25_w > 0) {
            auto bm = bm25_pool(q, docs);
            for (size_t i = 0; i < s.size(); ++i) s[i] += P.bm25_w * bm[i];
        }
        if (P.ax_w > 0) {
            auto ax = f2exp_pool(q, docs);
            for (size_t i = 0; i < s.size(); ++i) s[i] += P.ax_w * ax[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Relevance model
// ---------------------------------------------------------------------------
std::vector<std::pair<uint32_t, uint32_t>> Engine::bag(int32_t d, Scratch& s) const {
    std::vector<std::pair<uint32_t, uint32_t>> b;
    if (d < 0) return b;
    const uint32_t* t = toks_.data() + tok_off_[d];
    uint32_t n = static_cast<uint32_t>(tok_off_[d + 1] - tok_off_[d]);
    s.touched.clear();
    for (uint32_t i = 0; i < n; ++i) if (s.cnt[t[i]]++ == 0) s.touched.push_back(t[i]);
    for (uint32_t w : s.touched) { b.push_back({w, s.cnt[w]}); s.cnt[w] = 0; }
    std::sort(b.begin(), b.end());
    return b;
}

bool Engine::expansion_ok(uint32_t t) const {
    return !(term_flags_[t] & 1) && cf_[t] >= P.min_cf;
}

// P(D|Q) over the seed: softmax(beta * mean_q log P(q|D)); beta < 0 gives
// the exact RM1 weight P(Q|D) = exp(sum_q log P(q|D)).
std::vector<Engine::Seed> Engine::seed_weights(const Query& q, const std::vector<std::string_view>& seeds, Scratch& s) {
    std::vector<Seed> out;
    std::unordered_set<int32_t> seen;
    std::vector<int32_t> ds;
    for (auto id : seeds) { int32_t d = doc_index(id); if (d >= 0 && seen.insert(d).second) ds.push_back(d); }
    if (ds.empty()) return out;
    const double qn = static_cast<double>(q.len());
    std::vector<double> m(ds.size());
    std::vector<bool> anch(ds.size());
    size_t n_anch = 0;
    const double saved_ranker = P.ranker;
    if (P.fb_seed_ql > 0) P.ranker = 0;   // P(D|Q) from log P(q|D), not F2EXP
    for (size_t i = 0; i < ds.size(); ++i) {
        DocFeat f = doc_features(q, ds[i], s);
        m[i] = qpart(q, f, P.seed_sdm > 0) / qn;
        anch[i] = f.matched > 0;
        n_anch += anch[i];
    }
    P.ranker = saved_ranker;
    bool use_anchor = P.anchor_min > 0 && n_anch >= P.anchor_min;
    const double beta = P.beta < 0 ? qn : P.beta;
    double best = -1e300;
    for (size_t i = 0; i < ds.size(); ++i) if (!use_anchor || anch[i]) best = std::max(best, m[i]);
    for (size_t i = 0; i < ds.size(); ++i) {
        if (use_anchor && !anch[i]) continue;
        out.push_back({ds[i], std::exp(beta * (m[i] - best))});
    }
    if (P.trim_floor > 0) {
        std::vector<Seed> kept;
        for (auto& x : out) if (x.w >= P.trim_floor) kept.push_back(x);   // max weight is 1
        if (!kept.empty()) out.swap(kept);
    }
    std::stable_sort(out.begin(), out.end(), [](const Seed& a, const Seed& b) {
        return a.w != b.w ? a.w > b.w : a.d < b.d;
    });
    if (out.size() > static_cast<size_t>(std::max(1.0, P.fb_docs))) out.resize(static_cast<size_t>(std::max(1.0, P.fb_docs)));
    double z = 0; for (auto& x : out) z += x.w;
    for (auto& x : out) x.w = z > 0 ? x.w / z : 1.0 / out.size();
    return out;
}

std::vector<std::pair<uint32_t, double>> Engine::finish_model(std::vector<std::pair<uint32_t, double>>& m) const {
    if (P.idf_power != 0) {
        const double N = n_docs();
        for (auto& [t, w] : m) w *= std::pow(std::log((N + 1) / (df_[t] + 0.5)), P.idf_power);
    }
    std::sort(m.begin(), m.end(), [](const auto& a, const auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
    size_t keep = static_cast<size_t>(std::max(1.0, P.fb_terms));
    if (m.size() > keep) m.resize(keep);
    double z = 0; for (auto& x : m) z += x.second;
    if (z <= 0) return {};
    for (auto& x : m) x.second /= z;
    return m;
}

// RM1: P(w|R) = sum_D P(w|D) P(D|Q)
std::vector<std::pair<uint32_t, double>> Engine::est_rm1(const std::vector<Seed>& seed, Scratch& s) {
    std::unordered_map<uint32_t, std::pair<double, int>> acc;
    const double T = static_cast<double>(total_), fm = P.fb_mu;
    for (auto& sd : seed) {
        auto b = bag(sd.d, s);
        double dlen = static_cast<double>(tok_off_[sd.d + 1] - tok_off_[sd.d]);
        if (dlen + fm <= 0) continue;
        for (auto& [t, tf] : b) {
            double p = (tf + fm * cf_[t] / T) / (dlen + fm);
            if (p > P.per_doc_cap) p = P.per_doc_cap;
            auto& a = acc[t];
            a.first += sd.w * p;
            a.second += 1;
        }
    }
    int need = seed.size() >= 3 ? static_cast<int>(P.term_support) : 1;
    std::vector<std::pair<uint32_t, double>> m;
    for (auto& [t, a] : acc) if (a.second >= need && expansion_ok(t)) m.push_back({t, a.first});
    if (m.empty()) for (auto& [t, a] : acc) if (!(term_flags_[t] & 1)) m.push_back({t, a.first});
    return finish_model(m);
}

// RM2 (Lavrenko & Croft 2001, conditional / i.i.d. sampling):
//   P(w|R) ∝ P(w) prod_i sum_D P(q_i|D) P(D|w),  P(D|w) = P(w|D)P(D)/P(w)
// A term only scores well if it co-occurs with EVERY query term somewhere
// in the seed, which is what makes RM2 resistant to off-topic seed docs.
std::vector<std::pair<uint32_t, double>> Engine::est_rm2(const Query& q, const std::vector<Seed>& seed, Scratch& s) {
    const size_t S = seed.size(), Q = q.len();
    if (!S || !Q) return {};
    const double T = static_cast<double>(total_), fm = P.fb_mu;
    std::vector<double> PD(S), dl(S);
    std::vector<std::vector<double>> pq(S, std::vector<double>(Q));
    std::unordered_map<uint32_t, uint32_t> uidx;
    std::vector<uint32_t> terms;
    std::vector<int> support;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> bags(S);
    for (size_t j = 0; j < S; ++j) {
        PD[j] = P.rm2_uniform_prior > 0 ? 1.0 / S : seed[j].w;
        dl[j] = static_cast<double>(tok_off_[seed[j].d + 1] - tok_off_[seed[j].d]);
        bags[j] = bag(seed[j].d, s);
        for (auto& [t, tf] : bags[j]) {
            auto it = uidx.emplace(t, static_cast<uint32_t>(terms.size()));
            if (it.second) { terms.push_back(t); support.push_back(0); }
            ++support[it.first->second];
        }
        for (size_t i = 0; i < Q; ++i) {
            uint32_t tf = 0;
            auto it = std::lower_bound(bags[j].begin(), bags[j].end(), std::make_pair(q.terms[i], 0u));
            if (it != bags[j].end() && it->first == q.terms[i]) tf = it->second;
            pq[j][i] = std::exp(log_p(tf, static_cast<uint32_t>(dl[j]), cf_[q.terms[i]] / T));
        }
    }
    const size_t U = terms.size();
    std::vector<float> tfm(U * S, 0.0f);
    for (size_t j = 0; j < S; ++j) for (auto& [t, tf] : bags[j]) tfm[uidx[t] * S + j] = static_cast<float>(tf);
    int need = S >= 3 ? static_cast<int>(P.term_support) : 1;
    std::vector<std::pair<uint32_t, double>> logv;
    std::vector<double> pwd(S);
    for (size_t u = 0; u < U; ++u) {
        uint32_t t = terms[u];
        if (support[u] < need || !expansion_ok(t)) continue;
        double pc = cf_[t] / T, Pw = 0;
        for (size_t j = 0; j < S; ++j) {
            pwd[j] = dl[j] + fm > 0 ? (tfm[u * S + j] + fm * pc) / (dl[j] + fm) : 0.0;
            Pw += pwd[j] * PD[j];
        }
        if (Pw <= 0) continue;
        double lv = std::log(Pw);
        bool ok = true;
        for (size_t i = 0; i < Q && ok; ++i) {
            double inner = 0;
            for (size_t j = 0; j < S; ++j) inner += pq[j][i] * pwd[j] * PD[j];
            if (inner <= 0) ok = false; else lv += std::log(inner / Pw);
        }
        if (ok) logv.push_back({t, lv});
    }
    if (logv.empty()) return {};
    double mx = -1e300; for (auto& x : logv) mx = std::max(mx, x.second);
    for (auto& x : logv) x.second = std::exp(x.second - mx);
    return finish_model(logv);
}

std::vector<std::pair<uint32_t, double>> Engine::estimate(const Query& q, std::vector<Seed>& seed, int variant, Scratch& s) {
    auto run = [&]() -> std::vector<std::pair<uint32_t, double>> {
        double mix = variant == 1 ? 0.0 : variant == 2 ? 1.0 : P.rm2_mix;
        if (mix <= 0) return est_rm1(seed, s);
        if (mix >= 1) return est_rm2(q, seed, s);
        auto a = est_rm1(seed, s), b = est_rm2(q, seed, s);
        std::unordered_map<uint32_t, double> m;
        for (auto& [t, w] : a) m[t] += (1 - mix) * w;
        for (auto& [t, w] : b) m[t] += mix * w;
        std::vector<std::pair<uint32_t, double>> v(m.begin(), m.end());
        double save = P.idf_power; P.idf_power = 0;          // already applied
        auto r = finish_model(v);
        P.idf_power = save;
        return r;
    };
    auto model = run();
    if (P.agree_gamma > 0 && seed.size() >= 3 && !model.empty()) {
        // Down-weight seed docs whose term distribution disagrees with the
        // consensus model (cosine over the model's terms), then re-estimate.
        std::unordered_map<uint32_t, double> mm(model.begin(), model.end());
        double mn = 0; for (auto& x : model) mn += x.second * x.second;
        mn = std::sqrt(mn);
        double z = 0;
        for (auto& sd : seed) {
            auto b = bag(sd.d, s);
            double dot = 0, dn = 0;
            for (auto& [t, tf] : b) { dn += double(tf) * tf; auto it = mm.find(t); if (it != mm.end()) dot += tf * it->second; }
            double cs = (dn > 0 && mn > 0) ? dot / (std::sqrt(dn) * mn) : 0.0;
            sd.w *= std::pow(std::max(cs, 1e-3), P.agree_gamma);
            z += sd.w;
        }
        for (auto& sd : seed) sd.w = z > 0 ? sd.w / z : 1.0 / seed.size();
        model = run();
    }
    return model;
}

// Mean pairwise cosine of (1+log tf)*idf vectors of the seed docs.
double Engine::coherence(const std::vector<Seed>& seed, Scratch& s) const {
    if (seed.size() < 2) return 0.5;
    const double N = n_docs();
    std::vector<std::vector<std::pair<uint32_t, double>>> v(seed.size());
    std::vector<double> norm(seed.size(), 0);
    for (size_t i = 0; i < seed.size(); ++i) {
        for (auto& [t, tf] : bag(seed[i].d, s)) {
            if (term_flags_[t] & 1) continue;
            double w = (1 + std::log(double(tf))) * std::log(1 + N / df_[t]);
            v[i].push_back({t, w});
            norm[i] += w * w;
        }
        norm[i] = std::sqrt(norm[i]);
    }
    double sum = 0; int pairs = 0;
    for (size_t i = 0; i < v.size(); ++i) for (size_t j = i + 1; j < v.size(); ++j) {
        if (norm[i] <= 0 || norm[j] <= 0) continue;
        double dot = 0; size_t a = 0, b = 0;
        while (a < v[i].size() && b < v[j].size()) {
            if (v[i][a].first < v[j][b].first) ++a;
            else if (v[i][a].first > v[j][b].first) ++b;
            else { dot += v[i][a].second * v[j][b].second; ++a; ++b; }
        }
        sum += dot / (norm[i] * norm[j]); ++pairs;
    }
    return pairs ? std::min(1.0, std::max(0.0, sum / pairs)) : 0.5;
}

// Clarity: KL(P(.|R) || P(.|C)), scaled to [0,1].
double Engine::clarity(const std::vector<std::pair<uint32_t, double>>& m) const {
    double kl = 0;
    for (auto& [t, p] : m) { double pc = double(cf_[t]) / total_; if (p > 0 && pc > 0) kl += p * std::log(p / pc); }
    return std::min(1.0, std::max(0.0, kl / P.clarity_scale));
}

std::vector<std::pair<std::string, double>> Engine::relevance_model(std::string_view query,
                                                                    const std::vector<std::string_view>& seeds,
                                                                    int variant, Explain* ex) {
    std::vector<std::pair<std::string, double>> out;
    Query q = parse_query(query);
    if (q.terms.empty()) return out;
    ensure_pair_stats(q);
    Marks mk(*this, q);
    Scratch& s = *scratch_[0];
    auto seed = seed_weights(q, seeds, s);
    if (seed.empty()) return out;
    auto m = estimate(q, seed, variant, s);
    for (auto& [t, w] : m) out.push_back({std::string(vocab_.key(t)), w});
    if (ex) {
        ex->qterms = q.strs;
        for (auto& sd : seed) ex->seed_weights.push_back({std::string(docids_.key(doc_key_[sd.d])), sd.w});
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tracks B/C
// ---------------------------------------------------------------------------
std::vector<Ranked> Engine::feedback(std::string_view query, const std::vector<std::string_view>& seeds,
                                     const std::vector<std::string_view>& cands, int k, Explain* ex) {
    if (cands.empty() || k <= 0) return {};
    Query q = parse_query(query);
    if (q.terms.empty()) return pool_order(cands, k);
    ensure_pair_stats(q);
    Marks mk(*this, q);
    Scratch& s = *scratch_[0];
    const int variant = static_cast<int>(P.rm_variant);
    auto seed = seed_weights(q, seeds, s);
    if (seed.empty()) return score_candidates(query, cands, k);   // no usable feedback: exactly QL
    auto model = estimate(q, seed, variant, s);
    if (model.empty()) return score_candidates(query, cands, k);

    double lam, coh = 0, clar = 0;
    if (variant != 3) lam = 0.0;                        // pure RM1 / RM2 ranking
    else if (P.lambda >= 1) return score_candidates(query, cands, k);  // Track B == Track A
    else if (P.lambda >= 0) lam = std::min(1.0, P.lambda);
    else {
        coh = coherence(seed, s);
        clar = clarity(model);
        double quality = std::min(1.0, std::max(0.0, P.coh_w * coh + (1 - P.coh_w) * clar));
        lam = P.lambda_max - (P.lambda_max - P.lambda_min) * quality;
    }
    if (variant == 3 && P.lam_qlen_slope != 0)
        lam = std::min(0.95, std::max(0.0, lam + P.lam_qlen_slope * (q.len() - P.lam_qlen_pivot)));
    if (variant == 3 && P.small_seed_n > 0) {
        size_t valid = 0;
        std::unordered_set<int32_t> seen;
        for (auto id : seeds) { int32_t d = doc_index(id); if (d >= 0 && seen.insert(d).second) ++valid; }
        if (valid < P.small_seed_n) lam = 1.0 - (1.0 - lam) * valid / P.small_seed_n;
    }
    if (variant == 3 && P.qlen_shrink > 0 && q.len() > P.qlen_shrink)
        lam = 1.0 - (1.0 - lam) * P.qlen_shrink / q.len();
    std::vector<int32_t> docs(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) docs[i] = doc_index(cands[i]);
    double qpp_g = -1;
    if (variant == 3 && P.qpp_w > 0 && docs.size() >= 20) {
        // Uses only the query and the candidate pool: identical for clean and
        // perturbed seeds, so it cannot tell (or care) which call this is.
        std::vector<double> ql = score_pool(q, docs, nullptr, 1.0, true);
        zscore(ql);
        std::sort(ql.begin(), ql.end(), std::greater<double>());
        double g = 0; for (int i = 0; i < 10; ++i) g += ql[i];
        qpp_g = g / 10;
        double t = std::min(1.0, std::max(0.0, (P.qpp_hi - qpp_g) / std::max(1e-9, P.qpp_hi - P.qpp_lo)));
        lam = std::min(1.0, lam + P.qpp_w * t * (1.0 - lam));
    }
    if (ex) {
        ex->qterms = q.strs;
        for (auto& sd : seed) ex->seed_weights.push_back({std::string(docids_.key(doc_key_[sd.d])), sd.w});
        for (auto& [t, w] : model) ex->rm.push_back({std::string(vocab_.key(t)), w});
        ex->lambda = lam; ex->coherence = coh; ex->clarity = clar; ex->mu = mu_eff();
    }
    if (ex) { ex->lambda = lam; ex->qpp = qpp_g; }
    // Textbook RM3 when the lexical ranker is off.
    if (P.ranker == 0 && P.bm25_w <= 0 && P.lead_w <= 0 && P.ax_w <= 0 && P.len_prior == 0) {
        std::vector<double> base = score_pool(q, docs, &model, lam, P.fb_sdm > 0);
        return fuse_rank(base, cands, 0.0, k);
    }
    if (P.fb_mode >= 1) {
        // RM3 on top of the Track A score. KL scoring with the expanded model
        // P_RM3 = lambda P(w|Q) + (1-lambda) P(w|R) is
        //   lambda * (query part) + (1-lambda) * sum_w P(w|R) log P(w|D);
        // the query part here is the whole Track A score S_A, and the
        // expansion part is put on the same within-pool scale by a z-score.
        std::vector<double> sa = track_a(q, cands, docs);
        std::vector<double> r = rm_pool(docs, model);
        if (P.fb_loo > 0 && seed.size() >= 2) {
            for (size_t j = 0; j < seed.size(); ++j) {
                std::vector<size_t> at;
                for (size_t i = 0; i < docs.size(); ++i) if (docs[i] == seed[j].d) at.push_back(i);
                if (at.empty()) continue;
                std::vector<Seed> rest;
                double z = 0;
                for (size_t m = 0; m < seed.size(); ++m) if (m != j) { rest.push_back(seed[m]); z += seed[m].w; }
                for (auto& x : rest) x.w = z > 0 ? x.w / z : 1.0 / rest.size();
                auto m2 = estimate(q, rest, variant, s);
                if (m2.empty()) continue;
                const double v = rm_pool({seed[j].d}, m2)[0];
                for (size_t i : at) r[i] = v;
            }
        }
        if (P.fb_scale >= 2 && r.size() > 1) {
            // Center, then divide by fb_scale. Do not z-score: a z-score
            // gives a noisy seed the same size as a clean one. The raw
            // log-probability std is ~0.4, so lambda 0.9 with no divisor
            // (4 Oct 13:19) dropped hidden B by 0.0023. Dividing by 8 makes
            // the 0.1 weight the same size as the 13:04 run (B −0.00030).
            double m = 0;
            for (double x : r) m += x;
            m /= static_cast<double>(r.size());
            const double div = P.fb_scale;
            for (double& x : r) x = (x - m) / div;
        } else if (P.fb_scale >= 1 && r.size() > 1) {
            // Seed-independent unit: the pool std of the plain query
            // log-likelihood per query term. A mixed (noisy) seed gives a
            // flatter P(w|R) and so a smaller expansion term; a z-score
            // would blow it back up to one standard deviation.
            const double saved_ranker = P.ranker, saved_lp = P.len_prior;
            P.ranker = 0; P.len_prior = 0;
            std::vector<double> ql = score_pool(q, docs, nullptr, 1.0, false);
            P.ranker = saved_ranker; P.len_prior = saved_lp;
            double m = 0, mq = 0;
            for (size_t i = 0; i < r.size(); ++i) { m += r[i]; mq += ql[i]; }
            m /= r.size(); mq /= r.size();
            double var = 0;
            for (double x : ql) var += (x - mq) * (x - mq);
            const double sd = std::sqrt(var / ql.size()) / q.len();
            for (double& x : r) x = sd > 1e-12 ? (x - m) / sd : 0.0;
        } else {
            zscore(r);
        }
        if (P.fb_corr_pow > 0 && r.size() > 2) {
            double mr = 0, ms = 0;
            for (size_t i = 0; i < r.size(); ++i) { mr += r[i]; ms += sa[i]; }
            mr /= static_cast<double>(r.size());
            ms /= static_cast<double>(sa.size());
            double vr = 0, vs = 0, c = 0;
            for (size_t i = 0; i < r.size(); ++i) {
                double dr = r[i] - mr, ds = sa[i] - ms;
                vr += dr * dr; vs += ds * ds; c += dr * ds;
            }
            double rho = (vr > 0 && vs > 0) ? c / std::sqrt(vr * vs) : 0;
            double w = std::pow(std::max(0.0, rho), P.fb_corr_pow);
            for (double& x : r) x *= w;
        }
        for (size_t i = 0; i < sa.size(); ++i) sa[i] = lam * sa[i] + (1.0 - lam) * r[i];
        return top_k(sa, cands, k);
    }
    // RM3 query expansion, then the Track A ranker. The language-model
    // score is λ·P(Q|D) + (1-λ)·Σ_w P(w|R) log P(w|D). BM25 and the rank
    // prior are the same ones score_candidates() uses. λ is that weight,
    // not a later rescaling.
    std::vector<double> base = score_pool(q, docs, &model, lam, P.fb_sdm > 0);
    add_lexical(q, docs, base);
    const double pw = P.fb_prior_w < 0 ? P.prior_w : P.fb_prior_w;
    return fuse_rank(base, cands, pw, k);
}

}  // namespace rm
