#include "rm_engine.h"
#include "analysis.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>

namespace rm {

// ---------------------------------------------------------------------------
// Parameters by name
// ---------------------------------------------------------------------------
#define RM_PARAMS(X) \
    X(stemmer) X(query_stop) X(smoothing) X(mu) X(jm_lambda) X(spud_mu) X(spud_bg) \
    X(w_od) X(w_uw) X(uw_window) X(tr_w) X(tr_k) X(tr_mindf) X(lead_w) X(lead_len) X(lead_mu) \
    X(fb_mu) X(fb_docs) X(fb_terms) X(beta) X(support_frac) X(idf_power) X(fb_qterms) \
    X(fb_cov) X(fb_anchor_exp) X(lambda) X(fb_mode) X(fb_scale) X(fb_sdm)

bool set_param(Params& p, const std::string& n, double v) {
#define X(f) if (n == #f) { p.f = v; return true; }
    RM_PARAMS(X)
#undef X
    return false;
}

// Candidate scoring runs on the thread pool above this many pool tokens.
constexpr uint64_t PAR_MIN_TOKENS = 250000;

// ---------------------------------------------------------------------------
// Tokeniser: lowercase ASCII alnum runs, dotted initials glued (u.s. -> us),
// other length-1 tokens dropped. Used identically for documents and queries.
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
        else {
            flush_acro();
            if (tn > 1) emit(std::string_view(tok, tn));
        }
        tn = 0; overflow = false;
    };
    for (char ch : text) {
        const unsigned char c = static_cast<unsigned char>(ch);
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

// ---------------------------------------------------------------------------
// Per-call state
// ---------------------------------------------------------------------------
struct Engine::Scratch {
    std::vector<uint32_t> cnt;      // per-term counter, all zero between uses
    std::vector<uint32_t> touched;
    std::vector<std::pair<uint32_t, uint16_t>> hits;  // (position, query slot)
    explicit Scratch(size_t V) : cnt(V, 0) {}
};

struct Engine::Query {
    std::vector<uint32_t> terms;            // tids, with duplicates (in-vocab only)
    std::vector<uint32_t> uniq, uniq_cnt;   // distinct tids and multiplicity
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
    StrTable surf{1 << 14};                 // surface form -> local term
    std::vector<uint32_t> surf_to_local;
    StrTable local{1 << 14};                // stem -> local term id
    std::vector<uint64_t> cf;
    std::vector<uint32_t> df, stamp;
    std::vector<uint32_t> toks;
    std::vector<uint64_t> doc_end;          // cumulative token count per doc
    std::vector<std::string> ids;
};
}  // namespace

bool Engine::prepare(const std::string& path_in, std::string& err) {
    auto t0 = std::chrono::steady_clock::now();
    std::string path = path_in;
    struct stat st{};
    if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) path += "/corpus.jsonl";
    if (ready_ && path == path_ && P.stemmer == stemmer_used_) return true;  // idempotent

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
    // Chunk count depends on the file only, never on the thread count, so
    // term ids (and every tie-break) do not depend on how many threads ran.
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
            if (!parse_doc_line(ls, le, doc)) continue;
            ++docno;
            each_surface(doc.text, [&](std::string_view s) {
                uint64_t h = hash_str(s);
                bool ins = false;
                uint32_t sid = sh.surf.insert_h(s, h, &ins);
                if (ins) {
                    surf_str.assign(s);
                    bool ins2 = false;
                    uint32_t lid = sh.local.insert(apply_stem(surf_str, stemmer), &ins2);
                    if (ins2) { sh.cf.push_back(0); sh.df.push_back(0); sh.stamp.push_back(0); }
                    sh.surf_to_local.push_back(lid);
                }
                uint32_t lid = sh.surf_to_local[sid];
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
    docids_ = StrTable(N + 16);
    docid_to_doc_.clear();
    for (size_t c = 0; c < n_chunks; ++c) {
        for (size_t d = 0; d < shards[c].ids.size(); ++d) {
            bool ins = false;
            docids_.insert(shards[c].ids[d], &ins);
            if (ins) docid_to_doc_.push_back(static_cast<int32_t>(doc_base[c] + d));  // first wins
        }
    }
    shards.clear();
    total_ = T;
    avgdl_ = N ? static_cast<double>(T) / N : 0.0;

    // Expansion-term stoplist, on stemmed forms.
    std::unordered_set<std::string> stops;
    for (const auto& w : STOPWORDS) { stops.insert(w); stops.insert(apply_stem(w, stemmer)); }
    stop_.assign(vocab_.size(), 0);
    std::string tmp;
    for (uint32_t t = 0; t < vocab_.size(); ++t) {
        tmp.assign(vocab_.key(t));
        stop_[t] = stops.count(tmp) > 0;
    }

    scratch_.clear();
    for (int w = 0; w < pool_->workers(); ++w) scratch_.emplace_back(new Scratch(vocab_.size()));
    pair_cf_.clear();
    trans_.clear();
    trmark_.clear();
    qmark_.assign(vocab_.size(), 0xFFFF);
    build_postings();
    {
        dv_.assign(N, 0);
        df_total_ = 0;
        for (uint32_t x : df_) df_total_ += x;
        const int W = pool_->workers();
        std::vector<std::vector<uint32_t>> stamp(W, std::vector<uint32_t>(vocab_.size(), 0));
        const size_t chunks = std::min<size_t>(1024, std::max<size_t>(1, N / 256));
        pool_->run(chunks, [&](size_t c, int w) {
            auto& st = stamp[w];
            for (size_t d = N * c / chunks; d < N * (c + 1) / chunks; ++d) {
                const uint32_t mark = static_cast<uint32_t>(d + 1);
                uint32_t u = 0;
                for (uint64_t i = tok_off_[d]; i < tok_off_[d + 1]; ++i)
                    if (st[toks_[i]] != mark) { st[toks_[i]] = mark; ++u; }
                dv_[d] = u;
            }
        });
    }
    path_ = path;
    stemmer_used_ = P.stemmer;
    ready_ = true;
    prepare_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

std::string Engine::stats_json() const {
    std::ostringstream o;
    o << "{\"docs\":" << n_docs() << ",\"vocab\":" << vocab_.size() << ",\"tokens\":" << total_
      << ",\"avgdl\":" << avgdl_ << ",\"mu\":" << P.mu << ",\"threads\":" << pool_->workers()
      << ",\"prepare_seconds\":" << prepare_seconds_ << "}";
    return o.str();
}

int32_t Engine::doc_index(std::string_view id) const {
    uint32_t i = docids_.find(id);
    return i == StrTable::NONE ? -1 : docid_to_doc_[i];
}

// ---------------------------------------------------------------------------
// Query analysis
// ---------------------------------------------------------------------------
Engine::Query Engine::parse_query(std::string_view qtext) {
    const uint8_t stemmer = static_cast<uint8_t>(P.stemmer);
    std::vector<std::string> surf;
    each_surface(qtext, [&](std::string_view s) { surf.emplace_back(s); });
    std::vector<std::string> kept;
    if (P.query_stop > 0) for (auto& s : surf) if (!INDEX_STOPWORDS.count(s)) kept.push_back(s);
    if (kept.empty()) kept = surf;   // all-stopword query: keep everything
    Query q;
    int prev = -1;  // slot of previous in-vocab term; OOV breaks adjacency
    for (auto& s : kept) {
        uint32_t t = vocab_.find(apply_stem(s, stemmer));
        if (t == StrTable::NONE) { prev = -1; continue; }  // OOV: constant for every doc
        auto it = std::find(q.uniq.begin(), q.uniq.end(), t);
        uint16_t slot;
        if (it == q.uniq.end()) { slot = static_cast<uint16_t>(q.uniq.size()); q.uniq.push_back(t); q.uniq_cnt.push_back(0); }
        else slot = static_cast<uint16_t>(it - q.uniq.begin());
        ++q.uniq_cnt[slot];
        q.terms.push_back(t);
        if (prev >= 0) q.pairs.push_back({static_cast<uint16_t>(prev), slot});
        prev = slot;
        if (q.uniq.size() >= 0xFFF0) break;
    }
    q.pair_cf.assign(q.pairs.size(), {0, 0});
    return q;
}

// Marks the query's terms in the dense qmark_ array (term -> slot) for the
// lifetime of one call, so doc_features() classifies each token in O(1).
struct Engine::Marks {
    Engine& e; const Query& q;
    std::vector<uint32_t> tv;
    Marks(Engine& en, const Query& qq) : e(en), q(qq) {
        for (size_t u = 0; u < q.uniq.size(); ++u) e.qmark_[q.uniq[u]] = static_cast<uint16_t>(u);
        if (e.marks_depth_++ > 0) return;   // nested call for the same query
        e.trlist_.clear();
        if (e.P.tr_w <= 0) return;
        if (e.trmark_.size() != e.vocab_.size()) e.trmark_.assign(e.vocab_.size(), -1);
        for (size_t u = 0; u < q.uniq.size(); ++u)
            for (auto& [v, t] : e.translations(q.uniq[u])) {
                if (e.trmark_[v] < 0) {
                    e.trmark_[v] = static_cast<int32_t>(e.trlist_.size());
                    e.trlist_.emplace_back();
                    tv.push_back(v);
                }
                e.trlist_[e.trmark_[v]].push_back({static_cast<uint16_t>(u), t});
            }
    }
    ~Marks() {
        for (uint32_t t : q.uniq) e.qmark_[t] = 0xFFFF;
        if (--e.marks_depth_ > 0) return;
        for (uint32_t v : tv) e.trmark_[v] = -1;
        e.trlist_.clear();
    }
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
// (term, doc) pair is written once through a per-term atomic cursor. Lists
// are not sorted, which is fine: they are only ever summed over.
void Engine::build_postings() {
    post_off_.clear(); post_.clear();
    const size_t V = vocab_.size(), N = n_docs();
    uint64_t total = 0;
    for (uint32_t d : df_) total += d;
    if (V == 0 || total > (1ull << 28)) return;     // > 1 GB: fall back to corpus scans
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
// pair walks that term's document list (split into parallel tasks). Cached
// by term pair: pure collection statistics.
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

// t(w|v) for the translation model. Co-occurrence is counted over at most
// 30000 of w's documents, taken at a fixed stride, so it is deterministic.
const std::vector<std::pair<uint32_t, double>>& Engine::translations(uint32_t w) {
    const uint32_t K = static_cast<uint32_t>(std::max(1.0, P.tr_k));
    const uint32_t mindf = static_cast<uint32_t>(std::max(1.0, P.tr_mindf));
    const uint64_t key = (static_cast<uint64_t>(w) << 32) | (static_cast<uint64_t>(K & 0xFFFF) << 16) | (mindf & 0xFFFF);
    auto it = trans_.find(key);
    if (it != trans_.end()) return it->second;
    const size_t V = vocab_.size();
    std::vector<uint32_t> co(V, 0), stamp(V, 0), seen;
    const uint64_t lo = post_off_[w], n = post_off_[w + 1] - lo;
    const uint64_t cap = 30000, stride = n > cap ? (n + cap - 1) / cap : 1;
    uint32_t mark = 0, sampled = 0;
    for (uint64_t x = 0; x < n; x += stride) {
        const uint32_t d = post_[lo + x];
        ++mark; ++sampled;
        for (uint64_t i = tok_off_[d]; i < tok_off_[d + 1]; ++i) {
            const uint32_t t = toks_[i];
            if (stamp[t] == mark) continue;
            stamp[t] = mark;
            if (co[t]++ == 0) seen.push_back(t);
        }
    }
    std::vector<std::pair<uint32_t, double>> cand;
    const double scale = sampled ? static_cast<double>(n) / sampled : 0.0;
    for (uint32_t v : seen) {
        if (v == w || stop_[v] || df_[v] < mindf) continue;
        cand.push_back({v, co[v] * scale / std::sqrt(static_cast<double>(df_[v]) * df_[w])});
    }
    std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    if (cand.size() > K) cand.resize(K);
    double z = 0;
    for (auto& x : cand) z += x.second;
    for (auto& x : cand) x.second /= z > 0 ? z : 1.0;
    return trans_[key] = std::move(cand);
}

// ---------------------------------------------------------------------------
// Smoothed document language model
// ---------------------------------------------------------------------------
double Engine::pbg(uint32_t t) const {
    if (P.smoothing >= 3 && P.spud_bg > 0 && df_total_ > 0) return df_[t] / df_total_;
    return static_cast<double>(cf_[t]) / static_cast<double>(total_);
}

double Engine::log_p_dir(uint32_t tf, uint32_t dlen, double pc) const {
    const double mu = P.mu;
    return std::log((static_cast<double>(tf) + mu * pc) / (static_cast<double>(dlen) + mu));
}

double Engine::log_p(uint32_t tf, uint32_t dlen, double pc, uint32_t dv) const {
    if (P.smoothing >= 3) {
        const double om = P.spud_mu;
        const double ml = dlen ? static_cast<double>(tf) / dlen : 0.0;
        return std::log((dv * ml + om * pc) / (dv + om));
    }
    const double mu = P.mu;
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
    // sum_v t(w|v) tf(v,D) per query slot, for the translation model.
    std::vector<double> trc;
    if (!trlist_.empty() && d >= 0) {
        trc.assign(U, 0.0);
        const uint32_t* t = toks_.data() + tok_off_[d];
        for (uint32_t i = 0; i < dlen; ++i) {
            const int32_t m = trmark_[t[i]];
            if (m >= 0) for (auto& [u, w] : trlist_[m]) trc[u] += w;
        }
    }
    const double T = static_cast<double>(total_);
    const double lw = P.lead_w;
    const double ll = std::min<double>(dlen, lead_len);
    const double lmu = P.lead_mu;
    const uint32_t dvd = dv(d);
    for (size_t u = 0; u < U; ++u) {
        const double pc = pbg(q.uniq[u]);
        double lp = log_p(tf[u], dlen, pc, dvd);
        if (!trc.empty()) lp = std::log((1 - P.tr_w) * std::exp(lp) + P.tr_w * trc[u] / std::max<uint32_t>(dlen, 1));
        if (lw > 0) lp = std::log((1 - lw) * std::exp(lp) + lw * (static_cast<double>(tfl[u]) + lmu * pc) / (ll + lmu));
        f.uni += q.uniq_cnt[u] * lp;
        if (tf[u]) f.matched += q.uniq_cnt[u];
    }
    if (!q.pairs.empty() && (P.w_od > 0 || P.w_uw > 0)) {
        const uint32_t W = static_cast<uint32_t>(std::max(2.0, P.uw_window));
        std::vector<uint64_t> od(q.pairs.size(), 0), uw(q.pairs.size(), 0);
        if (s.hits.size() >= 2) count_pairs(s.hits, q.pairs, W, od.data(), uw.data());
        for (size_t p = 0; p < q.pairs.size(); ++p) {
            double pco = (q.pair_cf[p].first ? q.pair_cf[p].first : 0.5) / T;
            double pcu = (q.pair_cf[p].second ? q.pair_cf[p].second : 0.5) / T;
            // SPUD needs distinct-term counts, which pairs do not have: they keep Dirichlet.
            if (P.smoothing >= 3) {
                f.od += log_p_dir(static_cast<uint32_t>(od[p]), dlen, pco);
                f.uw += log_p_dir(static_cast<uint32_t>(uw[p]), dlen, pcu);
            } else {
                f.od += log_p(static_cast<uint32_t>(od[p]), dlen, pco, 0);
                f.uw += log_p(static_cast<uint32_t>(uw[p]), dlen, pcu, 0);
            }
        }
    }
    return f;
}

// Query part of the score. With sdm=false this is sum_q log P(q|D).
double Engine::qpart(const DocFeat& f, bool sdm) const {
    return sdm ? f.uni + P.w_od * f.od + P.w_uw * f.uw : f.uni;
}

// Scores every doc in `docs`:
//   model == nullptr:  qpart                                 [Track A]
//   otherwise:         lam * qpart/|Q| + (1-lam) * sum_w M(w) log P(w|D)
// The RM sum is split into a document-independent background term plus a
// correction over the model terms the document actually contains, so a
// document costs O(|D|) however many expansion terms there are.
std::vector<double> Engine::score_pool(const Query& q, const std::vector<int32_t>& docs,
                                       const Model* model, double lam, bool sdm) {
    const size_t n = docs.size();
    std::vector<double> out(n, 0.0);
    const double mu = P.mu;
    const bool jm = P.smoothing >= 1 && P.smoothing < 2;
    const bool spud = P.smoothing >= 3;
    const bool two_stage = P.smoothing >= 2 && !spud;
    const double om = P.spud_mu;
    const double ts_l = std::min(0.95, std::max(0.0, P.jm_lambda));
    std::vector<double>& mw = mw_;       // dense model weights, all zero between calls
    double msum = 0, base_const = 0;
    // Background part, log P(w|D) of a term absent from D:
    //   Dirichlet   log(mu pc) - log(|D| + mu)
    //   JM          log(l pc)
    //   two-stage   log(pc) + log((1-l) mu / (|D| + mu) + l)
    //   SPUD        log(om pc) - log(|D_v| + om)
    if (model) {
        if (mw.size() != vocab_.size()) mw.assign(vocab_.size(), 0.0);
        for (auto& [t, w] : *model) {
            mw[t] = w;
            msum += w;
            double pc = pbg(t);
            base_const += w * (jm ? std::log(P.jm_lambda * pc) : two_stage ? std::log(pc)
                               : spud ? std::log(om * pc) : std::log(mu * pc));
        }
    }
    const double qn = q.len() ? static_cast<double>(q.len()) : 1.0;
    auto one = [&](size_t i, Scratch& s) {
        const int32_t d = docs[i];
        DocFeat f = doc_features(q, d, s);
        double qs = qpart(f, sdm);
        if (!model) { out[i] = qs; return; }
        double rmp = base_const;
        uint32_t dlen = 0;
        if (d >= 0) {
            const uint32_t* t = toks_.data() + tok_off_[d];
            dlen = static_cast<uint32_t>(tok_off_[d + 1] - tok_off_[d]);
            s.touched.clear();
            for (uint32_t k = 0; k < dlen; ++k) { uint32_t w = t[k]; if (mw[w] > 0) { if (s.cnt[w]++ == 0) s.touched.push_back(w); } }
            const uint32_t dvd = dv(d);
            for (uint32_t w : s.touched) {
                double pc = pbg(w);
                rmp += mw[w] * (log_p(s.cnt[w], dlen, pc, dvd) - log_p(0, dlen, pc, dvd));
                s.cnt[w] = 0;
            }
        }
        if (spud) rmp -= msum * std::log(dv(d) + om);
        else if (two_stage) rmp += msum * std::log((1.0 - ts_l) * mu / (dlen + mu) + ts_l);
        else if (!jm) rmp -= msum * std::log(dlen + mu);
        out[i] = lam * qs / qn + (1 - lam) * rmp;
    };
    uint64_t work = 0;
    for (int32_t d : docs) if (d >= 0) work += tok_off_[d + 1] - tok_off_[d];
    if (work >= PAR_MIN_TOKENS && pool_->workers() > 1 && n > 8) {
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

// sum_w P(w|R) log P(w|D) for each doc, term by term: the expansion score of
// RM3 / the negative cross-entropy of RM1.
std::vector<double> Engine::rm_pool(const std::vector<int32_t>& docs, const Model& model) const {
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
            s += model[j].second * log_p(tf[j], dlen, pbg(model[j].first), dv(d));
        out[i] = s;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Ranking helpers
// ---------------------------------------------------------------------------
std::vector<Ranked> Engine::top_k(const std::vector<double>& score, const std::vector<std::string_view>& ids,
                                  int k) const {
    const size_t n = score.size();
    std::vector<int32_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = static_cast<int32_t>(i);
    auto cmp = [&](int32_t a, int32_t b) {
        if (score[a] != score[b]) return score[a] > score[b];
        return ids[a] < ids[b];   // deterministic doc_id tie-break
    };
    size_t kk = std::min<size_t>(n, k < 0 ? 0 : static_cast<size_t>(k));
    std::partial_sort(order.begin(), order.begin() + kk, order.end(), cmp);
    std::vector<Ranked> out;
    for (size_t i = 0; i < kk; ++i) out.push_back({order[i], score[order[i]]});
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
    return top_k(score_pool(q, docs, nullptr, 1.0, true), cands, k);
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

// P(D|Q) over the seed: softmax(beta * mean_q log P(q|D)); beta < 0 gives
// the exact RM1 weight P(Q|D) = exp(sum_q log P(q|D)). Unknown ids and
// duplicates are skipped; the top fb_docs documents are kept.
std::vector<Engine::Seed> Engine::seed_weights(const Query& q, const std::vector<std::string_view>& seeds, Scratch& s) {
    std::vector<Seed> out;
    std::unordered_set<int32_t> seen;
    std::vector<int32_t> ds;
    for (auto id : seeds) { int32_t d = doc_index(id); if (d >= 0 && seen.insert(d).second) ds.push_back(d); }
    if (ds.empty()) return out;
    const double qn = static_cast<double>(q.len());
    std::vector<double> m(ds.size());
    std::vector<double> cov(ds.size(), 0.0);
    for (size_t i = 0; i < ds.size(); ++i) {
        DocFeat f = doc_features(q, ds[i], s);
        m[i] = qpart(f, false) / qn;
        cov[i] = qn > 0 ? static_cast<double>(f.matched) / qn : 0.0;
    }
    const double beta = P.beta < 0 ? qn : P.beta;
    auto kept = [&](size_t i) { return P.fb_cov <= 0 || cov[i] + 1e-12 >= P.fb_cov; };
    double best = -1e300;
    for (size_t i = 0; i < ds.size(); ++i) if (kept(i)) best = std::max(best, m[i]);
    if (best < -1e200) return {};
    for (size_t i = 0; i < ds.size(); ++i) {
        if (!kept(i)) continue;
        double w = std::exp(beta * (m[i] - best));
        if (P.fb_anchor_exp > 0) w *= std::pow(std::max(cov[i], 0.0), P.fb_anchor_exp);
        if (w > 0) out.push_back({ds[i], w});
    }
    std::stable_sort(out.begin(), out.end(), [](const Seed& a, const Seed& b) {
        return a.w != b.w ? a.w > b.w : a.d < b.d;
    });
    const size_t cap = static_cast<size_t>(std::max(1.0, P.fb_docs));
    if (out.size() > cap) out.resize(cap);
    double z = 0; for (auto& x : out) z += x.w;
    for (auto& x : out) x.w = z > 0 ? x.w / z : 1.0 / out.size();
    return out;
}

int Engine::support_needed(size_t n_seed) const {
    if (P.support_frac <= 0) return 1;
    return std::max(1, static_cast<int>(std::ceil(P.support_frac * static_cast<double>(n_seed) - 1e-9)));
}

// Keeps the fb_terms best terms (plus `must`) and renormalises. Terms are
// ranked by P(w|R) * idf^idf_power but keep their raw probability.
Engine::Model Engine::finish_model(Model& m, const std::vector<uint32_t>& must) const {
    auto by_score = [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    };
    std::unordered_map<uint32_t, double> raw;
    const bool use_idf = P.idf_power != 0;
    const bool sel = use_idf || !must.empty();
    if (sel) {
        const double N = n_docs();
        for (auto& [t, w] : m) {
            raw[t] = w;
            if (use_idf) {
                double idf = std::log((N + 1.0) / (static_cast<double>(df_[t]) + 0.5));
                w *= std::pow(std::max(idf, 0.0), P.idf_power);
            }
        }
    }
    const size_t cap = static_cast<size_t>(std::max(1.0, P.fb_terms));
    if (!must.empty()) {
        std::unordered_set<uint32_t> musts(must.begin(), must.end());
        Model keep, rest;
        for (auto& kv : m) (musts.count(kv.first) ? keep : rest).push_back(kv);
        std::sort(rest.begin(), rest.end(), by_score);
        if (rest.size() > cap) rest.resize(cap);
        m.clear();
        m.insert(m.end(), keep.begin(), keep.end());
        m.insert(m.end(), rest.begin(), rest.end());
    } else {
        std::sort(m.begin(), m.end(), by_score);
        if (m.size() > cap) m.resize(cap);
    }
    if (sel) for (auto& [t, w] : m) w = raw[t];
    double z = 0; for (auto& x : m) z += x.second;
    if (z <= 0) return {};
    for (auto& x : m) x.second /= z;
    return m;
}

// RM1: P(w|R) = sum_D P(w|D) P(D|Q)
Engine::Model Engine::est_rm1(const Query& q, const std::vector<Seed>& seed, Scratch& s) {
    std::unordered_map<uint32_t, std::pair<double, int>> acc;
    const double T = static_cast<double>(total_), fm = P.fb_mu;
    for (auto& sd : seed) {
        auto b = bag(sd.d, s);
        double dlen = static_cast<double>(tok_off_[sd.d + 1] - tok_off_[sd.d]);
        if (dlen + fm <= 0) continue;
        for (auto& [t, tf] : b) {
            double p = (tf + fm * cf_[t] / T) / (dlen + fm);
            auto& a = acc[t];
            a.first += sd.w * p;
            a.second += 1;
        }
    }
    const int need = support_needed(seed.size());
    Model m;
    for (auto& [t, a] : acc) if (a.second >= need && expansion_ok(t)) m.push_back({t, a.first});
    std::vector<uint32_t> must;
    if (P.fb_qterms > 0) {
        for (uint32_t t : q.uniq) {
            if (stop_[t]) continue;
            auto it = acc.find(t);
            if (it == acc.end() || it->second.first <= 0) continue;
            bool present = false;
            for (auto& kv : m) if (kv.first == t) { present = true; break; }
            if (!present) m.push_back({t, it->second.first});
            must.push_back(t);
        }
    }
    return finish_model(m, must);
}

// RM2 (Lavrenko & Croft 2001, conditional / i.i.d. sampling):
//   P(w|R) ∝ P(w) prod_i sum_D P(q_i|D) P(D|w),  P(D|w) = P(w|D)P(D)/P(w)
// with a uniform P(D) over the seed.
Engine::Model Engine::est_rm2(const Query& q, const std::vector<Seed>& seed, Scratch& s) {
    const size_t S = seed.size(), Q = q.len();
    if (!S || !Q) return {};
    const double T = static_cast<double>(total_), fm = P.fb_mu;
    const double PD = 1.0 / S;
    std::vector<double> dl(S);
    std::vector<std::vector<double>> pq(S, std::vector<double>(Q));
    std::unordered_map<uint32_t, uint32_t> uidx;
    std::vector<uint32_t> terms;
    std::vector<int> support;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> bags(S);
    for (size_t j = 0; j < S; ++j) {
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
            pq[j][i] = std::exp(log_p(tf, static_cast<uint32_t>(dl[j]), pbg(q.terms[i]), dv(seed[j].d)));
        }
    }
    const size_t U = terms.size();
    std::vector<float> tfm(U * S, 0.0f);
    for (size_t j = 0; j < S; ++j) for (auto& [t, tf] : bags[j]) tfm[uidx[t] * S + j] = static_cast<float>(tf);
    const int need = support_needed(S);
    Model logv;
    std::vector<double> pwd(S);
    for (size_t u = 0; u < U; ++u) {
        uint32_t t = terms[u];
        if (support[u] < need || !expansion_ok(t)) continue;
        double pc = cf_[t] / T, Pw = 0;
        for (size_t j = 0; j < S; ++j) {
            pwd[j] = dl[j] + fm > 0 ? (tfm[u * S + j] + fm * pc) / (dl[j] + fm) : 0.0;
            Pw += pwd[j] * PD;
        }
        if (Pw <= 0) continue;
        double lv = std::log(Pw);
        bool ok = true;
        for (size_t i = 0; i < Q && ok; ++i) {
            double inner = 0;
            for (size_t j = 0; j < S; ++j) inner += pq[j][i] * pwd[j] * PD;
            if (inner <= 0) ok = false; else lv += std::log(inner / Pw);
        }
        if (ok) logv.push_back({t, lv});
    }
    if (logv.empty()) return {};
    double mx = -1e300; for (auto& x : logv) mx = std::max(mx, x.second);
    for (auto& x : logv) x.second = std::exp(x.second - mx);
    return finish_model(logv, {});
}

std::vector<std::pair<std::string, double>> Engine::relevance_model(std::string_view query,
                                                                    const std::vector<std::string_view>& seeds,
                                                                    int variant) {
    std::vector<std::pair<std::string, double>> out;
    Query q = parse_query(query);
    if (q.terms.empty()) return out;
    ensure_pair_stats(q);
    Marks mk(*this, q);
    Scratch& s = *scratch_[0];
    auto seed = seed_weights(q, seeds, s);
    if (seed.empty()) return out;
    auto m = variant == 2 ? est_rm2(q, seed, s) : est_rm1(q, seed, s);
    for (auto& [t, w] : m) out.push_back({std::string(vocab_.key(t)), w});
    return out;
}

// ---------------------------------------------------------------------------
// Tracks B/C
// ---------------------------------------------------------------------------
std::vector<Ranked> Engine::feedback(std::string_view query, const std::vector<std::string_view>& seeds,
                                     const std::vector<std::string_view>& cands, int k) {
    if (cands.empty() || k <= 0) return {};
    Query q = parse_query(query);
    if (q.terms.empty()) return pool_order(cands, k);
    if (P.lambda >= 1) return score_candidates(query, cands, k);   // Track B == Track A
    ensure_pair_stats(q);
    Marks mk(*this, q);
    Scratch& s = *scratch_[0];
    std::vector<int32_t> docs(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) docs[i] = doc_index(cands[i]);
    auto seed = seed_weights(q, seeds, s);
    if (seed.empty()) return score_candidates(query, cands, k);   // no usable feedback
    auto model = est_rm1(q, seed, s);
    if (model.empty()) return score_candidates(query, cands, k);
    const double lam = std::max(0.0, P.lambda);

    if (P.fb_mode >= 1) {
        // RM3 on top of the Track A score: the query side is the whole
        // score_candidates() score S_A, the expansion side is centred over
        // the pool and divided by fb_scale to put it on S_A's scale.
        std::vector<double> sa = score_pool(q, docs, nullptr, 1.0, true);
        std::vector<double> r = rm_pool(docs, model);
        if (r.size() > 1) {
            double m = 0;
            for (double x : r) m += x;
            m /= static_cast<double>(r.size());
            for (double& x : r) x = (x - m) / P.fb_scale;
        }
        const double wl = 1.0 - lam;
        for (size_t i = 0; i < sa.size(); ++i) sa[i] = (1.0 - wl) * sa[i] + wl * r[i];
        return top_k(sa, cands, k);
    }
    // RM3 inside the language model: lambda * P(Q|D)/|Q| + (1-lambda) * sum_w P(w|R) log P(w|D).
    return top_k(score_pool(q, docs, &model, lam, P.fb_sdm > 0), cands, k);
}

}  // namespace rm
