"""
tools/make_adversarial.py -- build submission/adversarial_set.json.

Bonus track (docs/GRADING.md): our doc lists are fed as the seed to every
other team's relevance_model_feedback(); the attack score is the nDCG@10
drop vs. their clean run on that query. We cannot run their code, so the
search attacks three simulated defenders spanning the likely range of the
class, all emulated by our own engine with different settings:

  naive     RM3, uniform P(D), 50 terms, lambda 0.5, no defences
  textbook  RM3 with exact RM1 weights P(D|Q) = P(Q|D)
  ours      our full submission (a strong, defended opponent)

Attack docs are NON-relevant pool documents (qrels <= 0 or unjudged). A
poisoned seed drawn from the pool is the strongest attack: the relevance
model then promotes exactly those documents into the top 10. Documents are
chosen greedily to maximise the mean drop over the three defenders, among
the non-relevant pool docs with the highest query likelihood (high P(Q|D)
is what gets a document past likelihood weighting and query anchoring).
Finally the --n queries with the largest mean drop are kept.

  python -m tools.make_adversarial --root data/full --n 5 --out submission/adversarial_set.json
"""
import argparse, json, math, os

from submission import feedback as fb
from tools.multi_sweep import find
from tools.sweep import ndcg10, read_candidates, read_qrels, read_queries

NAIVE = dict(beta=0, anchor_min=0, term_support=1, agree_gamma=0, fb_terms=50, fb_docs=10,
             w_od=0, w_uw=0, prior_w=0, lead_w=0, qlen_shrink=0, min_cf=1,
             rm2_mix=0, per_doc_cap=1, fb_mu=0, idf_power=0, len_prior=0, qpp_w=0, small_seed_n=0, longq_n=0, rm_variant=3)
NAIVE["lambda"] = 0.5
TEXTBOOK = dict(NAIVE, beta=-1)
DEFENDERS = {"naive": NAIVE, "textbook": TEXTBOOK, "ours": {}}


def use(cfg):
    fb.set_params(**fb.PARAMS)
    fb.set_params(**cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="collection dir (corpus/queries/qrels/candidates)")
    ap.add_argument("--n", type=int, default=5, help="queries to keep")
    ap.add_argument("--max-docs", type=int, default=10)
    ap.add_argument("--pool-cands", type=int, default=25)
    ap.add_argument("--out", default="submission/adversarial_set.json")
    a = ap.parse_args()
    corpus = find(a.root, ["corpus.jsonl"])
    queries = read_queries(find(a.root, ["queries_dev.tsv", "queries*.tsv", "*.tsv"]))
    qrels = read_qrels(find(a.root, ["qrels_dev.txt", "qrels*.txt", "qrels*.json"]))
    cands = read_candidates(find(a.root, ["candidates_dev.jsonl", "candidates*.jsonl"]))
    fb.prepare(corpus)

    results = []
    for qid, text in queries:
        pool, rel = cands.get(qid), qrels.get(qid)
        if not pool or not rel:
            continue
        clean, dfn = {}, {}
        for name, cfg in DEFENDERS.items():
            use(cfg)
            seed = [d for d, _ in fb.score_candidates(text, pool, 10)]
            clean[name] = ndcg10([d for d, _ in fb.relevance_model_feedback(text, seed, pool, 10)], rel)
        use({})
        ranked = [d for d, _ in fb.score_candidates(text, pool, len(pool))]
        bad = [d for d in ranked if rel.get(d, 0) <= 0][: a.pool_cands]
        if not bad:
            continue

        def drops(seed):
            out = {}
            for name, cfg in DEFENDERS.items():
                use(cfg)
                got = ndcg10([d for d, _ in fb.relevance_model_feedback(text, seed, pool, 10)], rel)
                out[name] = clean[name] - got
            return out

        chosen, best, best_seed, best_d = [], -1e9, [], {}
        while len(chosen) < a.max_docs:
            step = None
            for d in bad:
                if d in chosen:
                    continue
                dd = drops(chosen + [d])
                m = sum(dd.values()) / len(dd)
                if step is None or m > step[0]:
                    step = (m, d, dd)
            if step is None:
                break
            chosen.append(step[1])
            if step[0] > best:
                best, best_seed, best_d = step[0], list(chosen), step[2]
        results.append((best, qid, best_seed, best_d))
        print(f"{qid}: mean drop {best:.3f}  " + "  ".join(f"{k}={v:.3f}" for k, v in best_d.items())
              + f"  ({len(best_seed)} docs)", flush=True)

    results.sort(reverse=True)
    keep = results[: a.n]
    out = {qid: seed for _, qid, seed, _ in keep}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    if keep:
        avg = {k: sum(r[3][k] for r in keep) / len(keep) for k in DEFENDERS}
        print(f"\nwrote {a.out}: {len(out)} queries; average drop on kept queries: "
              + "  ".join(f"{k}={v:.3f}" for k, v in avg.items()))
        print("('ours' = how much this attack would hurt us: our defence against a strong attacker)")
    use({})


if __name__ == "__main__":
    main()
