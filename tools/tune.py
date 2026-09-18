"""
tools/tune.py -- pick parameters without fitting the ~50 dev topics.
NOT part of the submission.

The problem this solves
-----------------------
There is nothing to train in submission/feedback.py -- no weights, no
gradients. What there is: about eight hyperparameters, and a released dev
topic set small enough that picking them by "best dev nDCG" mostly fits
the topics rather than the collection. The held-out topics are different
ones, so that fit does not transfer.

Two guards, both implemented here:

  1. K-FOLD CROSS-VALIDATION OVER TOPICS. Split the dev topics into folds,
     choose parameters on K-1 folds, score on the held-out fold. The
     reported number is the mean held-out score, which is an honest
     estimate of what the graded topics will do. A setting whose in-fold
     and held-out numbers diverge is fitting noise.
  2. MULTIPLE COLLECTIONS AND MULTIPLE CANDIDATE POOLS. Pass --collection
     more than once. The search then optimises the WORST collection's
     score, not the mean: on an unseen topic set you care about the floor,
     not the average. Use tools/fetch_datasets.py --degrade to add a
     weakened candidate pool as one of the collections.

The objective
-------------
The real leaderboard percentile-ranks each track against the class, which
no single team can compute. The local surrogate is

    OBJ = 0.30*A_ndcg + 0.25*B_ndcg + 0.35*(retention * B_ndcg)

i.e. Track C's ratio re-multiplied by clean nDCG, so the objective cannot
be gamed by tanking B to make the ratio look good. --pareto skips the
scalarisation and prints the B-versus-retention frontier instead, which is
the honest picture and the thing to put in the report: there is no single
best point, there is a trade-off curve, and you choose where to sit on it.

Usage
-----
    python -m tools.tune --collection data/full --folds 5 --search coordinate
    python -m tools.tune --collection data/full --collection data/nfcorpus \
        --collection data/fiqa --search coordinate
    python -m tools.tune --collection data/full --pareto
"""
import argparse
import importlib
import itertools
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

from harness.metrics import evaluate_run
from harness.noise_injection import perturb_pseudo_relevant_set, stable_seed
from harness.candidates_io import read_candidates
from harness.trec_io import read_qrels, read_queries

import submission.feedback as fb

PRF_DEPTH = 10
NOISE_LEVELS = [0.25, 0.5, 0.75]

# Search space. Ordered by expected payoff -- coordinate descent walks it
# in this order and one pass is usually enough.
GRID: Dict[str, List] = {
    "MU_AVGDL_FACTOR": [6.0, 9.0, 12.0, 16.0, 22.0],
    "LAMBDA_MIN": [0.30, 0.40, 0.45, 0.55],
    "LAMBDA_MAX": [0.70, 0.80, 0.85, 0.92],
    "MAX_EXPANSION_TERMS": [20, 40, 60, 100],
    "MAX_FEEDBACK_DOCS": [5, 8, 10],
    "WEIGHT_BETA": [0.8, 1.6, 2.5, 4.0],
    "FEEDBACK_MU": [100.0, 400.0, 1000.0],
    "RANK_PRIOR_WEIGHT": [0.0, 0.15, 0.30, 0.50],
    "TRIM_FLOOR": [0.05, 0.15, 0.30],
}


class Collection:
    def __init__(self, path: str, candidates: str = None) -> None:
        self.path = path
        self.name = os.path.basename(path.rstrip("/"))
        self.corpus = os.path.join(path, "corpus.jsonl")
        self.queries = read_queries(os.path.join(path, "queries_dev.tsv"))
        self.qrels = read_qrels(os.path.join(path, "qrels_dev.txt"))
        cand_path = candidates or os.path.join(path, "candidates_dev.jsonl")
        self.candidates = read_candidates(cand_path)
        self.cand_name = os.path.basename(cand_path)
        self.qids = [q for q, _t in self.queries if q in self.candidates and q in self.qrels]


def _apply(params: Dict) -> None:
    for key, value in params.items():
        setattr(fb, key, value)


def evaluate(coll: Collection, qids: Sequence[str]) -> Dict[str, float]:
    """Track A / B / retention for one collection restricted to `qids`."""
    fb.prepare(coll.corpus)
    text_by_qid = dict(coll.queries)
    qrels = {q: coll.qrels[q] for q in qids}

    ql_run, clean_run = {}, {}
    noisy_runs = {lv: {} for lv in NOISE_LEVELS}

    for qid in qids:
        text = text_by_qid[qid]
        pool = [d for d, _s in coll.candidates[qid]]
        ql_run[qid] = fb.score_candidates(text, pool, 10)
        seed = [d for d, _s in fb.score_candidates(text, pool, PRF_DEPTH)]
        clean_run[qid] = fb.relevance_model_feedback(text, seed, pool, 10)
        rng = random.Random(stable_seed(qid, base_seed=0))
        for lv in NOISE_LEVELS:
            perturbed = perturb_pseudo_relevant_set(seed, pool, lv, rng)
            noisy_runs[lv][qid] = fb.relevance_model_feedback(text, perturbed, pool, 10)

    a = evaluate_run(ql_run, qrels)["aggregate"]
    b = evaluate_run(clean_run, qrels)["aggregate"]["ndcg@10"]
    noisy = [evaluate_run(r, qrels)["aggregate"]["ndcg@10"] for r in noisy_runs.values()]
    mean_noisy = sum(noisy) / len(noisy) if noisy else 0.0
    retention = min(1.0, mean_noisy / b) if b > 0 else 0.0
    return {
        "A_ndcg": a["ndcg@10"], "A_map": a["map@10"],
        "B_ndcg": b, "retention": retention, "mean_noisy": mean_noisy,
    }


def objective(m: Dict[str, float]) -> float:
    """Leaderboard-shaped surrogate. Track C is re-multiplied by B so the
    objective cannot be improved by making B worse."""
    return 0.30 * m["A_ndcg"] + 0.25 * m["B_ndcg"] + 0.35 * (m["retention"] * m["B_ndcg"])


def folds_of(qids: List[str], k: int, seed: int = 0) -> List[List[str]]:
    shuffled = list(qids)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::k] for i in range(k)]


def cross_validated(colls: List[Collection], params: Dict, k: int) -> Dict:
    """Mean held-out objective per collection, plus the worst of them."""
    _apply(params)
    per_coll = {}
    for coll in colls:
        scores, mets = [], []
        for fold in folds_of(coll.qids, k):
            if not fold:
                continue
            m = evaluate(coll, fold)
            scores.append(objective(m))
            mets.append(m)
        n = max(1, len(mets))
        per_coll[f"{coll.name}/{coll.cand_name}"] = {
            "obj": sum(scores) / max(1, len(scores)),
            "A_ndcg": sum(m["A_ndcg"] for m in mets) / n,
            "B_ndcg": sum(m["B_ndcg"] for m in mets) / n,
            "retention": sum(m["retention"] for m in mets) / n,
        }
    objs = [v["obj"] for v in per_coll.values()]
    return {"per_collection": per_coll, "worst": min(objs), "mean": sum(objs) / len(objs)}


def current_params() -> Dict:
    return {key: getattr(fb, key) for key in GRID}


def coordinate_search(colls: List[Collection], k: int, passes: int) -> Tuple[Dict, Dict]:
    best = current_params()
    best_res = cross_validated(colls, best, k)
    print(f"start           worst={best_res['worst']:.4f}  mean={best_res['mean']:.4f}  {best}")
    for p in range(passes):
        improved = False
        for key, values in GRID.items():
            for value in values:
                if value == best[key]:
                    continue
                trial = dict(best, **{key: value})
                if trial["LAMBDA_MIN"] >= trial["LAMBDA_MAX"]:
                    continue
                res = cross_validated(colls, trial, k)
                if res["worst"] > best_res["worst"] + 1e-6:
                    best, best_res, improved = trial, res, True
                    print(f"  {key}={value!r:>8}  worst={res['worst']:.4f}  mean={res['mean']:.4f}")
        if not improved:
            print(f"pass {p + 1}: converged")
            break
    return best, best_res


def pareto(colls: List[Collection], k: int) -> None:
    """B versus retention across fixed lambda -- the trade-off curve."""
    print(f"{'lambda':>8}  {'A':>7}  {'B(clean)':>9}  {'retention':>9}  {'obj':>7}")
    for lam in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        fb.LAMBDA_FIXED = lam
        res = cross_validated(colls, current_params(), k)
        agg = list(res["per_collection"].values())
        a = sum(v["A_ndcg"] for v in agg) / len(agg)
        b = sum(v["B_ndcg"] for v in agg) / len(agg)
        r = sum(v["retention"] for v in agg) / len(agg)
        print(f"{lam:>8.1f}  {a:>7.4f}  {b:>9.4f}  {r:>9.4f}  {res['worst']:>7.4f}")
    fb.LAMBDA_FIXED = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", action="append", required=True,
                    help="directory with corpus.jsonl/queries_dev.tsv/qrels_dev.txt; repeatable")
    ap.add_argument("--candidates", action="append", default=None,
                    help="explicit candidates file per --collection, same order")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--search", choices=["none", "coordinate"], default="none")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--out", default="tuning_result.json")
    args = ap.parse_args()

    cands = args.candidates or [None] * len(args.collection)
    colls = [Collection(p, c) for p, c in zip(args.collection, cands)]
    for coll in colls:
        print(f"{coll.name}: {len(coll.qids)} scorable queries, pool {coll.cand_name}")
    print()

    if args.pareto:
        pareto(colls, args.folds)
        return

    if args.search == "coordinate":
        best, res = coordinate_search(colls, args.folds, args.passes)
    else:
        best = current_params()
        res = cross_validated(colls, best, args.folds)

    print("\n== held-out (cross-validated) ==")
    for name, v in res["per_collection"].items():
        print(f"  {name:<34} obj={v['obj']:.4f}  A={v['A_ndcg']:.4f}  "
              f"B={v['B_ndcg']:.4f}  ret={v['retention']:.4f}")
    print(f"  worst={res['worst']:.4f}  mean={res['mean']:.4f}")
    print("\n== parameters to paste into submission/feedback.py ==")
    for key, value in best.items():
        print(f"{key} = {value!r}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"params": best, "result": res}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
