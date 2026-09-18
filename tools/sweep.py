"""
tools/sweep.py -- local experiment driver. NOT part of the submission;
the grading harness never imports this. Delete it before submitting if
your course requires a clean tree.

Three things the report (Section 8) asks for that run_harness.py does not
give you directly:

  1. An ablation table -- each robustness device in submission/feedback.py
     turned off one at a time, so you can report what each one buys.
  2. A noise sweep deeper than the public practice levels, out to 100%
     contamination.
  3. A HARDER noise model than the public one. harness/noise_injection.py
     draws contamination uniformly from the candidate pool, and says in
     as many words that topically *adjacent* contamination is worse.
     `--noise-model adjacent` implements that: it swaps in the
     highest-scoring non-seed pool documents instead of random ones, so
     the contamination arrives already looking plausible. Treat the
     public recipe as the floor and this as the ceiling; the undisclosed
     grading recipe is somewhere between.

Usage:
    python -m tools.sweep --ablations
    python -m tools.sweep --noise-model adjacent
    python -m tools.sweep --grid lambda
"""
import argparse
import importlib
import random
from typing import Dict, List

from harness.candidates_io import read_candidates
from harness.metrics import evaluate_run
from harness.noise_injection import perturb_pseudo_relevant_set, stable_seed
from harness.trec_io import read_qrels, read_queries

import submission.feedback as fb

PRF_DEPTH = 10
DEEP_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

TOY = dict(
    corpus="data/toy/corpus.jsonl",
    queries="data/toy/queries_dev.tsv",
    qrels="data/toy/qrels_dev.txt",
    candidates="data/toy/candidates_dev.jsonl",
)


def adjacent_perturb(seed_ids, pool_ids, fraction, query, rng):
    """Topically adjacent contamination: replace `fraction` of the seed
    with the strongest non-seed candidates under the submission's own
    first-pass model, rather than with uniform-random pool documents.

    This is the attack a smart opponent runs in the bonus round-robin,
    and the worst realistic case for any term-distribution method: the
    contamination already scores well against the query, so per-document
    query-likelihood weighting cannot reject it on its own.
    """
    n = len(seed_ids)
    n_replace = round(n * fraction)
    if n_replace == 0:
        return list(seed_ids)
    seen = set(seed_ids)
    outside = [d for d in pool_ids if d not in seen]
    if not outside:
        return list(seed_ids)
    ranked = fb.score_candidates(query, outside, len(outside))
    noise = [d for d, _s in ranked[:n_replace]]
    out = list(seed_ids)
    positions = rng.sample(range(n), min(n_replace, n))
    for pos, doc in zip(positions, noise):
        out[pos] = doc
    return out


def evaluate(paths: Dict[str, str], levels: List[float], noise_model: str) -> Dict:
    queries = read_queries(paths["queries"])
    qrels = read_qrels(paths["qrels"])
    cands = read_candidates(paths["candidates"])

    fb.prepare(paths["corpus"])

    ql_run, runs = {}, {lv: {} for lv in levels}
    for qid, text in queries:
        if qid not in cands:
            continue
        pool = [d for d, _s in cands[qid]]
        ql_run[qid] = fb.score_candidates(text, pool, 10)
        clean_seed = [d for d, _s in fb.score_candidates(text, pool, PRF_DEPTH)]
        rng = random.Random(stable_seed(qid, base_seed=0))
        for lv in levels:
            if noise_model == "adjacent":
                seed = adjacent_perturb(clean_seed, pool, lv, text, rng)
            else:
                seed = perturb_pseudo_relevant_set(clean_seed, pool, lv, rng)
            runs[lv][qid] = fb.relevance_model_feedback(text, seed, pool, 10)

    ql = evaluate_run(ql_run, qrels)["aggregate"]
    by_level = {lv: evaluate_run(r, qrels)["aggregate"]["ndcg@10"] for lv, r in runs.items()}
    clean = by_level.get(0.0, 0.0)
    noisy = [v for lv, v in by_level.items() if lv > 0.0]
    retention = (sum(noisy) / len(noisy) / clean) if (clean > 0 and noisy) else 0.0
    return {
        "trackA_ndcg": ql["ndcg@10"],
        "trackA_map": ql["map@10"],
        "clean_ndcg": clean,
        "by_level": by_level,
        "retention": min(1.0, retention),
    }


def _reload():
    importlib.reload(fb)


def show(label: str, res: Dict, levels: List[float]) -> None:
    cells = "  ".join(f"{lv:.2f}:{res['by_level'][lv]:.4f}" for lv in levels)
    print(
        f"{label:<26} A={res['trackA_ndcg']:.4f}/{res['trackA_map']:.4f}  "
        f"B={res['clean_ndcg']:.4f}  ret={res['retention']:.4f}   {cells}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=TOY["corpus"])
    ap.add_argument("--queries", default=TOY["queries"])
    ap.add_argument("--qrels", default=TOY["qrels"])
    ap.add_argument("--candidates", default=TOY["candidates"])
    ap.add_argument("--noise-model", choices=["uniform", "adjacent"], default="uniform")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--grid", choices=["lambda", "mu", "beta", "prior"], default=None)
    args = ap.parse_args()

    paths = dict(
        corpus=args.corpus, queries=args.queries,
        qrels=args.qrels, candidates=args.candidates,
    )
    levels = DEEP_LEVELS
    print(f"noise model: {args.noise_model}    levels: {levels}\n")
    print(f"{'config':<26} {'trackA ndcg/map':<18} {'B(clean)':<9} {'retention':<10} per-level nDCG@10")
    print("-" * 118)

    base = evaluate(paths, levels, args.noise_model)
    show("full model", base, levels)

    if args.ablations:
        switches = [
            ("USE_SOFT_WEIGHTS", "-soft P(D|Q) weights"),
            ("USE_TRIMMING", "-weight trimming"),
            ("USE_QUERY_ANCHOR", "-query anchor"),
            ("USE_TERM_CONSENSUS", "-term consensus"),
            ("USE_ADAPTIVE_LAMBDA", "-adaptive lambda"),
            ("USE_STEMMER", "-stemmer"),
        ]
        for attr, label in switches:
            _reload()
            setattr(fb, attr, False)
            show(label, evaluate(paths, levels, args.noise_model), levels)
        _reload()
        fb.RANK_PRIOR_WEIGHT = 0.0
        show("-rank prior", evaluate(paths, levels, args.noise_model), levels)
        _reload()
        for attr, _ in switches:
            setattr(fb, attr, False)
        fb.LAMBDA_FIXED = 0.5
        show("plain RM3 (lam=0.5)", evaluate(paths, levels, args.noise_model), levels)
        _reload()
        fb.LAMBDA_FIXED = 1.0
        show("no expansion (lam=1)", evaluate(paths, levels, args.noise_model), levels)
        _reload()

    if args.grid == "lambda":
        for lam in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            _reload()
            fb.LAMBDA_FIXED = lam
            show(f"lambda={lam:.1f}", evaluate(paths, levels, args.noise_model), levels)
        _reload()
    elif args.grid == "mu":
        for mu in [200, 500, 1000, 1500, 2000, 3000]:
            _reload()
            fb.MU_MIN = fb.MU_MAX = float(mu)
            show(f"mu={mu}", evaluate(paths, levels, args.noise_model), levels)
        _reload()
    elif args.grid == "beta":
        for beta in [0.0, 0.5, 1.0, 1.6, 2.5, 4.0]:
            _reload()
            fb.WEIGHT_BETA = beta
            show(f"beta={beta}", evaluate(paths, levels, args.noise_model), levels)
        _reload()
    elif args.grid == "prior":
        for w in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
            _reload()
            fb.RANK_PRIOR_WEIGHT = w
            show(f"rank prior w={w}", evaluate(paths, levels, args.noise_model), levels)
        _reload()


if __name__ == "__main__":
    main()
