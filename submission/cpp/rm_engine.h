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
    double qstop_list = 1;       // 0 short (Lucene) list, 1 long (Fox) list
    // --- query likelihood -----------------------------------------------------
    double smoothing = 0;        // 0 Dirichlet, 1 Jelinek-Mercer
    double mu = 0;               // >0: fixed mu; 0: mu_factor * avgdl, clamped
    double mu_factor = 12, mu_min = 250, mu_max = 2500;
    double jm_lambda = 0.5;      // JM weight on P(w|C)
    // Sequential dependence (Metzler & Croft 2005). Weights are relative to
    // the unigram part, so w_od = w_uw = 0 is exactly unigram QL.
    double w_od = 0.0, w_uw = 0.0;
    double uw_window = 8;
    // Rank prior from the order of candidate_doc_ids, fused by z-score.
    double prior_w = 0.0, prior_k = 60;
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
    double per_doc_cap = 1.0;    // winsorise P(w|D) in RM1
    double min_cf = 2;           // expansion terms rarer than this are dropped
    double idf_power = 0.0;      // P(w|R) *= idf(w)^p
    double agree_gamma = 0.0;    // outlier-seed down-weighting (0 off)
    double lambda = 0.5;         // RM3 weight on the query; <0 = adaptive
    double lambda_min = 0.45, lambda_max = 0.85;
    double coh_w = 0.65;         // adaptive: quality = coh_w*coherence + (1-coh_w)*clarity
    double clarity_scale = 6.0;
    double fb_sdm = 1;           // RM3 query part uses the SDM score
    double fb_prior_w = -1;      // rank prior in feedback (-1 = same as prior_w)
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
    double lambda = 1, coherence = 0, clarity = 0, mu = 0;
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
    Query parse_query(std::string_view q);
    int32_t doc_index(std::string_view id) const;
    void ensure_pair_stats(Query& q);
    double log_p(uint32_t tf, uint32_t dlen, double pc) const;
    DocFeat doc_features(const Query& q, int32_t d, Scratch& s) const;
    double qpart(const Query& q, const DocFeat& f, bool sdm) const;
    std::vector<Ranked> fuse_rank(std::vector<double>& score, const std::vector<std::string_view>& ids,
                                  double prior_w, int k) const;
    std::vector<Ranked> pool_order(const std::vector<std::string_view>& ids, int k) const;

    // Relevance-model internals.
    struct Seed { int32_t d; double w; };
    std::vector<Seed> seed_weights(const Query& q, const std::vector<std::string_view>& seeds, Scratch& s);
    std::vector<std::pair<uint32_t, double>> estimate(const Query& q, std::vector<Seed>& seed, int variant, Scratch& s);
    std::vector<std::pair<uint32_t, double>> est_rm1(const std::vector<Seed>& seed, Scratch& s);
    std::vector<std::pair<uint32_t, double>> est_rm2(const Query& q, const std::vector<Seed>& seed, Scratch& s);
    std::vector<std::pair<uint32_t, double>> finish_model(std::vector<std::pair<uint32_t, double>>& m) const;
    double coherence(const std::vector<Seed>& seed, Scratch& s) const;
    double clarity(const std::vector<std::pair<uint32_t, double>>& m) const;
    std::vector<std::pair<uint32_t, uint32_t>> bag(int32_t d, Scratch& s) const;
    bool expansion_ok(uint32_t t) const;
    std::vector<double> score_pool(const Query& q, const std::vector<int32_t>& docs,
                                   const std::vector<std::pair<uint32_t, double>>* model,
                                   double lam, bool sdm);

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
