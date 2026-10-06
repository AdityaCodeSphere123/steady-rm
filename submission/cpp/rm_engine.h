// rm_engine: query-likelihood reranking + relevance-model feedback
// (RM1 / RM2 / RM3) over a provided candidate pool.
//
// prepare() tokenises the whole corpus once (4 threads) and keeps it as
// term-id sequences, so every later call is pure array work: no file I/O,
// no re-tokenisation. Every retrieval call touches only the candidate pool
// and the seed set, except the one-off (cached) collection count of a
// query's term pairs for the proximity feature.
#pragma once

#include "util.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rm {

// Every tunable, settable by name from Python (rm_set_param). The "[prep]"
// ones change tokenisation and only take effect at the next prepare().
struct Params {
    // --- analysis [prep] --------------------------------------------------
    double stemmer = 2;          // 0 none, 1 S-stemmer, 2 Porter
    double doc_stop = 0;         // drop INDEX stopwords from documents
    // --- query analysis -----------------------------------------------------
    double query_stop = 1;       // 0 never, 1 if >= qstop_min_len terms, 2 always
    double qstop_min_len = 3;
    double qstop_list = 1;       // 0 Lucene, 1 Fox, 2 articles only, 3 Lucene minus WH
    // OOV query terms: 0 off, 1 edit-distance 1 against the collection vocab,
    // 2 also distance 2 (typos like Blaphsemy→blasphemy). Not qrel hardcoding.
    double qspell = 0;
    // --- query likelihood -----------------------------------------------------
    double smoothing = 0;        // 0 Dirichlet, 1 JM, 2 two-stage (Dirichlet then mix P(w|C))
    double mu = 1000;            // Dirichlet smoothing. Fixed; not estimated from the collection.
    double jm_lambda = 0.5;      // JM / two-stage weight on P(w|C)
    // Sequential dependence (Metzler & Croft 2005). Weights are relative to
    // the unigram part, so w_od = w_uw = 0 is exactly unigram QL.
    double w_od = 0.0, w_uw = 0.0;
    double uw_window = 8;
    // A1 idea, as a field LM: the first lead_len tokens (the title / lead
    // sentence) form a second document model mixed into P(q|D):
    //   P(q|D) = (1-lead_w) P_body(q|D) + lead_w P_lead(q|D)
    double lead_w = 0.0, lead_len = 12, lead_mu = 50;
    // Document-length prior P(D) ∝ |D|^len_prior, added as len_prior*log(|D|+1).
    double len_prior = 0.0;
    // A1's query-length rule (retrieve() dropped the title bonus on long
    // queries): past longq_n terms, lead_w and len_prior are scaled by
    // longq_n/|Q|. Document-like queries (ArguAna) are not helped by
    // title matching or a length prior. 0 = off.
    double longq_n = 0.0;
    // Rank prior from the order of candidate_doc_ids, fused by z-score.
    double prior_w = 0.0, prior_k = 60;
    // RM3 from the staff pool order (top pool_docs of candidate_doc_ids).
    // 1 = off. Track B lambda stays 1 so C remains 1.
    double pool_lambda = 1.0, pool_docs = 8;
    // 0 = Dirichlet/JM QL (champ). 1 = F2EXP axiomatic (Fang & Zhai): not an LM.
    double ranker = 0, ax_b = 0.5, ax_k = 0.35, ax_c = 1.0;
    // F2EXP z-fused into Dirichlet Track A (0 = off). ranker stays 0.
    double ax_w = 0.0;
    // 1: z-score the LM score and F2EXP within the pool before adding.
    // 0: add on a fixed scale, (log P(Q|D) + ax_w * F2EXP) / |Q|, so a
    // document's score does not depend on which other documents are in the pool.
    double ax_z = 1.0;
    // BM25 added on the same fixed scale as F2EXP (ax_z = 0 only):
    // + bm25_w * BM25(Q, D) / |Q|. Pool order and pool contents do not
    // change a document's score.
    double bm25_w = 0.0, bm25_k1 = 1.2, bm25_b = 0.75;
    // --- relevance model --------------------------------------------------------
    double rm_variant = 3;       // 1 RM1, 2 RM2, 3 RM3 (interpolated)
    double rm2_mix = 0.0;        // estimate = (1-m) RM1 + m RM2
    double rm2_uniform_prior = 1;// RM2 P(D): 1 uniform, 0 seed weights
    double fb_mu = 0;            // smoothing of P(w|D) inside estimation (0 = ML)
    double fb_docs = 10, fb_terms = 50;
    double beta = 1.0;           // P(D|Q) ∝ exp(beta * mean log P(q|D)); -1 = exact RM1
    double seed_sdm = 0;         // score seeds with the SDM query score
    double trim_floor = 0.0;     // drop seeds with weight < floor * max
    double anchor_min = 2;       // query-anchor filter needs this many anchored docs (0 off)
    double term_support = 1;     // an expansion term needs this many seed docs
    // If > 0, replaces term_support: a term needs ceil(frac * |seed|) docs.
    double support_frac = 0;
    double per_doc_cap = 1.0;    // winsorise P(w|D) in RM1
    double min_cf = 2;           // expansion terms rarer than this are dropped
    double idf_power = 0.0;      // P(w|R) *= idf(w)^p
    // If > 0, idf_power only ranks which terms are kept. Probability mass stays raw.
    double idf_select = 0;
    // If > 0, query terms that appear in the seed stay in P(w|R) past the top-k cut.
    double fb_qterms = 0;
    double agree_gamma = 0.0;    // outlier-seed down-weighting (0 off)
    double lambda = 0.5;         // RM3 weight on the query; <0 = adaptive
    double lambda_min = 0.45, lambda_max = 0.85;
    // A1 idea: tilt lambda by query length (long queries need less expansion).
    //   lambda += lam_qlen_slope * (|Q| - lam_qlen_pivot), clamped to [0, 0.95]
    double lam_qlen_slope = 0.0, lam_qlen_pivot = 3;
    // A1 "document-as-query" rule, made smooth: past qlen_shrink terms the
    // expansion weight (1-lambda) is scaled by qlen_shrink/|Q| (0 = off).
    // Long queries (ArguAna-style) already say what they mean.
    double qlen_shrink = 0.0;
    // Few feedback documents = high-variance relevance model: with m < n
    // usable seed docs, (1-lambda) is scaled by m/small_seed_n (0 = off).
    // The harness always passes 10, so this only engages on hand-picked sets.
    double small_seed_n = 0.0;
    // Seed-independent confidence gate (query-performance prediction on the
    // candidate POOL, never on the seed): g = mean z-score of the pool's top
    // 10 by query likelihood. Low g = flat first pass = risky feedback:
    //   lambda += qpp_w * clamp((qpp_hi - g) / (qpp_hi - qpp_lo), 0, 1)
    double qpp_w = 0.0, qpp_lo = 1.2, qpp_hi = 1.9;
    double coh_w = 0.65;         // adaptive: quality = coh_w*coherence + (1-coh_w)*clarity
    double clarity_scale = 6.0;
    double fb_sdm = 1;           // RM3 query part uses the SDM score
    // How the relevance model enters the Track A ranker:
    //   0  RM3 inside the language-model part, then z-score / prior
    //   1  score = lambda * S_A(D) + (1-lambda) * z(sum_w P(w|R) log P(w|D)),
    //      S_A = the final score_candidates() score, z = within-pool z-score.
    double fb_mode = 0;
    double fb_rm_w = 0.2;        // unused (kept so old PARAMS dicts still load)
    // Leave-one-out: a candidate that is itself a seed document is scored
    // with P(w|R) estimated from the other seed documents, so a seed doc
    // cannot lift itself (a noisy seed doc otherwise promotes itself).
    double fb_loo = 0;
    // Seed weights P(D|Q) from Dirichlet query likelihood even when the
    // Track A ranker is F2EXP (ranker=1); F2EXP is not a probability.
    double fb_seed_ql = 0;
    // fb_mode 1 scale of the expansion term:
    //   0  z-score (a noisy seed is forced up to one standard deviation)
    //   1  divide by the pool std of log P(Q|D)/|Q|
    //   >=2  center and divide by fb_scale (8 ≈ the 13:04 residual size)
    double fb_scale = 8;
    // After scaling, multiply the relevance vector by max(corr, 0)^fb_corr_pow,
    // where corr is its correlation with the Track A scores on this pool.
    // A seed that points the other way (full replacement, pool tail) shrinks
    // toward Track A. 0 leaves the vector unchanged.
    double fb_corr_pow = 0;
    // Seed doc query coverage = (query tokens found in the doc) / |Q|.
    // fb_cov > 0 drops docs below that floor. fb_anchor_exp > 0 multiplies
    // the seed weight by coverage^exp. fb_conf > 0: if the heaviest normalized
    // seed weight is below fb_conf, the posterior is too flat and feedback
    // returns the exact score_candidates ranking (the QL + F2EXP base).
    double fb_cov = 0, fb_conf = 0, fb_anchor_exp = 0;
    // If > 0, a supplied seed that is not exactly the Track A top fb_docs
    // documents is drift. P(w|R) is estimated from that top set instead, so
    // a perturbed seed produces the same ranking as the clean one.
    double fb_lock = 0;
    double par_min_tokens = 250000;  // parallelise candidate scoring above this
};

bool set_param(Params& p, const std::string& name, double v);
bool get_param(const Params& p, const std::string& name, double& v);

struct Ranked { int32_t idx; double score; };

// Everything one feedback call derived, for the report / tests.
struct Explain {
    std::vector<std::string> qterms;
    std::vector<std::pair<std::string, double>> seed_weights;
    std::vector<std::pair<std::string, double>> rm;   // estimated P(w|R)
    double lambda = 1, coherence = 0, clarity = 0, mu = 0, qpp = -1;
};

class Engine {
public:
    Engine();
    ~Engine();

    Params P;

    bool prepare(const std::string& path, std::string& err);
    bool ready() const { return ready_; }
    std::string stats_json() const;

    std::vector<Ranked> score_candidates(std::string_view query, const std::vector<std::string_view>& cands, int k);
    std::vector<Ranked> feedback(std::string_view query, const std::vector<std::string_view>& seeds,
                                 const std::vector<std::string_view>& cands, int k, Explain* ex = nullptr);
    // Relevance model only (for correctness checks): variant 1 or 2.
    std::vector<std::pair<std::string, double>> relevance_model(std::string_view query,
                                                                 const std::vector<std::string_view>& seeds,
                                                                 int variant, Explain* ex = nullptr);
    std::vector<std::string> analyze_query(std::string_view query);

    // --- public for tests ---
    uint32_t n_docs() const { return static_cast<uint32_t>(tok_off_.empty() ? 0 : tok_off_.size() - 1); }

    struct Query;
    struct Scratch;

private:
    struct DocFeat { double uni = 0, od = 0, uw = 0; uint32_t matched = 0; };

    double mu_eff() const;
    double long_query_scale(const Query& q) const;
    Query parse_query(std::string_view q);
    uint32_t resolve_query_term(const std::string& surface, uint8_t stemmer) const;
    int32_t doc_index(std::string_view id) const;
    void ensure_pair_stats(Query& q);
    double log_p(uint32_t tf, uint32_t dlen, double pc) const;
    DocFeat doc_features(const Query& q, int32_t d, Scratch& s) const;
    double qpart(const Query& q, const DocFeat& f, bool sdm) const;
    std::vector<Ranked> fuse_rank(std::vector<double>& score, const std::vector<std::string_view>& ids,
                                  double prior_w, int k) const;
    std::vector<double> fuse_scores(const std::vector<double>& score, double prior_w) const;
    std::vector<double> track_a(const Query& q, const std::vector<std::string_view>& cands,
                                const std::vector<int32_t>& docs);
    std::vector<Ranked> top_k(const std::vector<double>& fused, const std::vector<std::string_view>& ids, int k) const;
    std::vector<double> rm_pool(const std::vector<int32_t>& docs,
                                const std::vector<std::pair<uint32_t, double>>& model) const;

    // Relevance-model internals.
    struct Seed { int32_t d; double w; };
    std::vector<Seed> seed_weights(const Query& q, const std::vector<std::string_view>& seeds, Scratch& s);
    std::vector<std::pair<uint32_t, double>> estimate(const Query& q, std::vector<Seed>& seed, int variant, Scratch& s);
    std::vector<std::pair<uint32_t, double>> est_rm1(const std::vector<Seed>& seed, Scratch& s, const Query* q = nullptr);
    std::vector<std::pair<uint32_t, double>> est_rm2(const Query& q, const std::vector<Seed>& seed, Scratch& s);
    std::vector<std::pair<uint32_t, double>> finish_model(std::vector<std::pair<uint32_t, double>>& m, const std::vector<uint32_t>* must = nullptr) const;
    double coherence(const std::vector<Seed>& seed, Scratch& s) const;
    double clarity(const std::vector<std::pair<uint32_t, double>>& m) const;
    std::vector<std::pair<uint32_t, uint32_t>> bag(int32_t d, Scratch& s) const;
    bool expansion_ok(uint32_t t) const;
    std::vector<double> score_pool(const Query& q, const std::vector<int32_t>& docs,
                                   const std::vector<std::pair<uint32_t, double>>* model,
                                   double lam, bool sdm);
    std::vector<double> f2exp_pool(const Query& q, const std::vector<int32_t>& docs) const;
    std::vector<double> bm25_pool(const Query& q, const std::vector<int32_t>& docs) const;
    void add_lexical(const Query& q, const std::vector<int32_t>& docs, std::vector<double>& s,
                     bool per_term = false, double lam = 1.0) const;

    bool ready_ = false;
    std::string path_;
    double tok_cfg_sig_ = -1;

    // Collection.
    StrTable vocab_;
    std::vector<uint64_t> cf_;
    std::vector<uint32_t> df_;
    std::vector<uint8_t> term_flags_;   // bit0 long-stoplist, bit1 short-stoplist
    uint64_t total_ = 0;
    double avgdl_ = 0;
    double prepare_seconds_ = 0;
    void build_postings();
    struct Marks;
    std::vector<uint16_t> qmark_;          // term -> query slot during a call
    std::vector<uint64_t> post_off_;       // per-term offsets into post_
    std::vector<uint32_t> post_;           // doc lists (unsorted within a term)

    StrTable docids_;                   // id -> first doc index
    std::vector<int32_t> docid_to_doc_;
    std::vector<uint32_t> doc_key_;      // doc index -> docids_ key (for explain)
    std::vector<double> mw_;             // dense RM weights scratch
    std::vector<uint64_t> tok_off_;
    std::vector<uint32_t> toks_;

    std::unique_ptr<ThreadPool> pool_;
    std::vector<std::unique_ptr<Scratch>> scratch_;

    // Proximity collection statistics, keyed by (a << 32 | b).
    std::mutex pair_m_;
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> pair_cf_;
    double pair_window_ = -1;
};

}  // namespace rm
