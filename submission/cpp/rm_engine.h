// Query-likelihood reranking and relevance-model feedback.
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

// Set by name from Python. stemmer applies on the next prepare().
struct Params {
    double stemmer = 2;          // 0 none, otherwise Porter
    double query_stop = 1;
    double smoothing = 0;        // 0 Dirichlet, 1 JM, 2 two-stage, 3 SPUD
    double mu = 1000;
    double jm_lambda = 0.5;
    double spud_mu = 300, spud_bg = 1;   // spud_bg 1 uses df, 0 uses cf
    double w_od = 0.0, w_uw = 0.0, uw_window = 8;
    double lead_w = 0.0, lead_len = 12, lead_mu = 50;
    double fb_mu = 0;            // 0 is maximum likelihood
    double fb_docs = 10, fb_terms = 50;
    double beta = 1.0;           // below 0, P(D|Q) = P(Q|D)
    double support_frac = 0;
    double idf_power = 0.0;
    double fb_qterms = 0;
    double fb_cov = 0, fb_anchor_exp = 0;
    double rm_model = 3;         // 1 RM1, 2 RM2, 3 RM3
    double lambda = 0.5;         // 1 returns score_candidates()
    double fb_mode = 0;          // 0 inside the LM, 1 added to the Track A score
    double fb_scale = 8;
    double fb_sdm = 1;
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
    double pbg(uint32_t t) const;
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

    StrTable vocab_;
    std::vector<uint64_t> cf_;
    std::vector<uint32_t> df_;
    std::vector<uint32_t> dv_;
    double df_total_ = 0;
    std::vector<uint8_t> stop_;
    uint64_t total_ = 0;
    double avgdl_ = 0;
    double prepare_seconds_ = 0;
    struct Marks;
    std::vector<uint16_t> qmark_;
    std::vector<uint64_t> post_off_;
    std::vector<uint32_t> post_;

    StrTable docids_;
    std::vector<int32_t> docid_to_doc_;
    std::vector<double> mw_;
    std::vector<uint64_t> tok_off_;
    std::vector<uint32_t> toks_;

    std::unique_ptr<ThreadPool> pool_;
    std::vector<std::unique_ptr<Scratch>> scratch_;

    std::mutex pair_m_;
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> pair_cf_;
    double pair_window_ = -1;
};

}  // namespace rm
