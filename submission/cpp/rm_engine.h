// rm_engine: query-likelihood reranking and relevance-model feedback over a
// provided candidate pool.
//
// prepare() tokenises the corpus once (4 threads) into term-id sequences, so
// every later call is array work over the candidate pool and the seed set.
// The one exception is the cached collection count of a query's term pairs
// for the proximity features.
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

// Every tunable, settable by name from Python (rm_set_param). stemmer only
// takes effect at the next prepare().
struct Params {
    // --- analysis ---------------------------------------------------------
    double stemmer = 2;          // 0 none, otherwise Porter
    double query_stop = 1;       // drop Lucene stopwords from the query
    // --- document language model -----------------------------------------
    double smoothing = 0;        // 0 Dirichlet, 1 JM, 2 two-stage, 3 SPUD
    double mu = 1000;            // Dirichlet; also smooths proximity pairs under SPUD
    double jm_lambda = 0.5;      // JM / two-stage weight on P(w|C)
    // SPUD (Cummins, Paik & Lv 2015), the Polya-urn document model:
    //   P(w|D) = (|D_v| tf/|D| + spud_mu P(w|C)) / (|D_v| + spud_mu)
    // |D_v| = distinct terms in D. spud_bg 1: P(w|C) = df_w / sum df;
    // 0: cf_w / |C|.
    double spud_mu = 300, spud_bg = 1;
    // Sequential dependence (Metzler & Croft 2005), weights relative to the
    // unigram part: w_od = w_uw = 0 is plain query likelihood.
    double w_od = 0.0, w_uw = 0.0;
    double uw_window = 8;
    // Lead field: the first lead_len tokens form a second document model,
    //   P(q|D) = (1-lead_w) P_body(q|D) + lead_w P_lead(q|D)
    double lead_w = 0.0, lead_len = 12, lead_mu = 50;
    // --- relevance model --------------------------------------------------
    double fb_mu = 0;            // Dirichlet smoothing of P(w|D) in estimation (0 = ML)
    double fb_docs = 10, fb_terms = 50;
    double beta = 1.0;           // P(D|Q) ∝ exp(beta * mean log P(q|D)); -1 = exact RM1
    // An expansion term must occur in ceil(support_frac * |seed|) seed docs (at least 1).
    double support_frac = 0;
    // Expansion terms are ranked by P(w|R) * idf(w)^idf_power; the kept
    // terms keep their raw P(w|R).
    double idf_power = 0.0;
    double fb_qterms = 0;        // query terms found in the seed survive the fb_terms cut
    // Seed docs covering less than fb_cov of the query tokens are dropped;
    // the rest are weighted by coverage^fb_anchor_exp.
    double fb_cov = 0, fb_anchor_exp = 0;
    // --- feedback ranking -------------------------------------------------
    double lambda = 0.5;         // weight on the query side; 1 = Track A exactly
    // 0  RM3: lambda * P(Q|D)/|Q| + (1-lambda) * sum_w P(w|R) log P(w|D)
    // 1  (1-w) S_A(D) + w (E(D) - mean E) / fb_scale, w = 1-lambda, where
    //    S_A is the score_candidates() score and E(D) = sum_w P(w|R) log P(w|D)
    double fb_mode = 0;
    double fb_scale = 8;
    double fb_sdm = 1;           // fb_mode 0: the query side includes the proximity features
};

bool set_param(Params& p, const std::string& name, double v);

struct Ranked { int32_t idx; double score; };

class Engine {
public:
    Engine();
    ~Engine();

    Params P;

    bool prepare(const std::string& path, std::string& err);
    std::string stats_json() const;

    std::vector<Ranked> score_candidates(std::string_view query, const std::vector<std::string_view>& cands, int k);
    std::vector<Ranked> feedback(std::string_view query, const std::vector<std::string_view>& seeds,const std::vector<std::string_view>& cands, int k);
    // The estimated relevance model alone: variant 1 = RM1, 2 = RM2.
    std::vector<std::pair<std::string, double>> relevance_model(std::string_view query,const std::vector<std::string_view>& seeds,int variant);

    uint32_t n_docs() const { return static_cast<uint32_t>(tok_off_.empty() ? 0 : tok_off_.size() - 1); }

    struct Query;
    struct Scratch;

private:
    struct DocFeat { double uni = 0, od = 0, uw = 0; uint32_t matched = 0; };
    struct Seed { int32_t d; double w; };
    using Model = std::vector<std::pair<uint32_t, double>>;

    Query parse_query(std::string_view q);
    int32_t doc_index(std::string_view id) const;
    void ensure_pair_stats(Query& q);
    double log_p(uint32_t tf, uint32_t dlen, double pc, uint32_t dv) const;
    double log_p_dir(uint32_t tf, uint32_t dlen, double pc) const;
    double pbg(uint32_t t) const;   // P(w|C) of the active document model
    uint32_t dv(int32_t d) const { return d >= 0 && static_cast<size_t>(d) < dv_.size() ? dv_[d] : 0; }
    DocFeat doc_features(const Query& q, int32_t d, Scratch& s) const;
    double qpart(const DocFeat& f, bool sdm) const;
    std::vector<Ranked> top_k(const std::vector<double>& score, const std::vector<std::string_view>& ids, int k) const;
    std::vector<Ranked> pool_order(const std::vector<std::string_view>& ids, int k) const;
    std::vector<double> score_pool(const Query& q, const std::vector<int32_t>& docs, const Model* model,
                                   double lam, bool sdm);
    std::vector<double> rm_pool(const std::vector<int32_t>& docs, const Model& model) const;

    std::vector<Seed> seed_weights(const Query& q, const std::vector<std::string_view>& seeds, Scratch& s);
    Model est_rm1(const Query& q, const std::vector<Seed>& seed, Scratch& s);
    Model est_rm2(const Query& q, const std::vector<Seed>& seed, Scratch& s);
    Model finish_model(Model& m, const std::vector<uint32_t>& must) const;
    int support_needed(size_t n_seed) const;
    std::vector<std::pair<uint32_t, uint32_t>> bag(int32_t d, Scratch& s) const;
    bool expansion_ok(uint32_t t) const { return !stop_[t]; }
    void build_postings();

    bool ready_ = false;
    std::string path_;
    double stemmer_used_ = -1;

    // Collection.
    StrTable vocab_;
    std::vector<uint64_t> cf_;
    std::vector<uint32_t> df_;
    std::vector<uint32_t> dv_;          // distinct terms per document
    double df_total_ = 0;               // sum of df_ (= sum of dv_)
    std::vector<uint8_t> stop_;         // term is on the expansion stoplist
    uint64_t total_ = 0;
    double avgdl_ = 0;
    double prepare_seconds_ = 0;
    struct Marks;
    std::vector<uint16_t> qmark_;       // term -> query slot during a call
    std::vector<uint64_t> post_off_;    // per-term offsets into post_
    std::vector<uint32_t> post_;        // doc lists (unsorted within a term)

    StrTable docids_;                   // id -> first doc index
    std::vector<int32_t> docid_to_doc_;
    std::vector<double> mw_;            // dense RM weights scratch
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
