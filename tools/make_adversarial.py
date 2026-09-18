"""
tools/make_adversarial.py -- generate submission/adversarial_set.json for
the optional bonus round-robin (docs/GRADING.md). NOT part of the graded
submission interface; the harness never imports this.

What the bonus track actually rewards
-------------------------------------
Attack score = the average nDCG@10 drop your poisoned seed induces in
every OTHER team's relevance_model_feedback(), relative to that team's own
clean score on the same query. You never see the defenders. So the search
below optimises against SURROGATE defenders -- several plausible RM3
implementations of the kind classmates are likely to submit -- and keeps
the attack that hurts them on average, rather than the one that happens to
destroy any single configuration.

What makes a poisoned seed strong
---------------------------------
A seed document has to survive every filter a defender might apply before
it can do damage, so the search implicitly selects for all of these at
once:

  * non-relevant under the dev qrels (otherwise promoting it helps);
  * high query likelihood, so per-document P(D|Q) weighting and any
    query-anchored filter still let it vote;
  * mutually coherent with the rest of the poisoned set, so a
    term-consensus filter sees the shared off-topic terms attested by
    several documents rather than one;
  * carrying terms that are concentrated in OTHER non-relevant pool
    documents, so the drift actually promotes them into the top 10.

Greedy forward selection finds sets with those properties without having
to hand-encode any of them: it just measures the damage.

Usage:
    python -m tools.make_adversarial                    # toy set
    python -m tools.make_adversarial --corpus data/full/corpus.jsonl ...
"""
import argparse
import json
import os
from typing import Dict, List

from harness.candidates_io import read_candidates
from harness.metrics import ndcg_at_k
from harness.trec_io import read_qrels, read_queries

import submission.feedback as fb

PRF_DEPTH = 10


def _surrogates():
    """Plausible defender configurations to optimise against.

    Deliberately a spread: an unfiltered textbook RM3 at two interpolation
    weights (what most submissions will be), plus our own hardened model
    (so the attack is not simply one our own defence would shrug off, which
    would be evidence it is narrow).
    """
    return [
        dict(name="rm3-lam0.3", LAMBDA_FIXED=0.3, USE_SOFT_WEIGHTS=False,
             USE_TRIMMING=False, USE_QUERY_ANCHOR=False, USE_TERM_CONSENSUS=False,
             USE_ADAPTIVE_LAMBDA=False),
        dict(name="rm3-lam0.5", LAMBDA_FIXED=0.5, USE_SOFT_WEIGHTS=False,
             USE_TRIMMING=False, USE_QUERY_ANCHOR=False, USE_TERM_CONSENSUS=False,
             USE_ADAPTIVE_LAMBDA=False),
        dict(name="rm3-weighted-lam0.4", LAMBDA_FIXED=0.4, USE_SOFT_WEIGHTS=True,
             USE_TRIMMING=False, USE_QUERY_ANCHOR=False, USE_TERM_CONSENSUS=False,
             USE_ADAPTIVE_LAMBDA=False),
        dict(name="hardened", LAMBDA_FIXED=None),
    ]


_DEFAULTS = {}


def _snapshot():
    global _DEFAULTS
    _DEFAULTS = {
        key: getattr(fb, key)
        for key in (
            "LAMBDA_FIXED", "USE_SOFT_WEIGHTS", "USE_TRIMMING",
            "USE_QUERY_ANCHOR", "USE_TERM_CONSENSUS", "USE_ADAPTIVE_LAMBDA",
        )
    }


def _configure(cfg: Dict):
    for key, value in _DEFAULTS.items():
        setattr(fb, key, value)
    for key, value in cfg.items():
        if key != "name":
            setattr(fb, key, value)


def _ndcg(query, seed, pool, qrels_q):
    ranked = [d for d, _s in fb.relevance_model_feedback(query, seed, pool, 10)]
    return ndcg_at_k(ranked, qrels_q, k=10)


def mean_drop(query, seed, pool, qrels_q, clean_by_surrogate) -> float:
    """Average nDCG@10 drop this seed induces across the surrogates."""
    drops = []
    for cfg in _surrogates():
        _configure(cfg)
        drops.append(clean_by_surrogate[cfg["name"]] - _ndcg(query, seed, pool, qrels_q))
    return sum(drops) / len(drops)


def attack_query(query, pool, qrels_q, max_size: int, candidates_considered: int):
    """Greedy forward selection of a poisoned seed set."""
    _configure({"name": "hardened", "LAMBDA_FIXED": None})
    clean_seed = [d for d, _s in fb.score_candidates(query, pool, PRF_DEPTH)]

    clean_by_surrogate = {}
    for cfg in _surrogates():
        _configure(cfg)
        clean_by_surrogate[cfg["name"]] = _ndcg(query, clean_seed, pool, qrels_q)

    # Only non-relevant documents are worth poisoning with; order them by
    # query likelihood so the strongest impostors are tried first.
    _configure({"name": "hardened", "LAMBDA_FIXED": None})
    ranked_pool = [d for d, _s in fb.score_candidates(query, pool, len(pool))]
    pool_nonrel = [d for d in ranked_pool if qrels_q.get(d, 0) == 0][:candidates_considered]
    if not pool_nonrel:
        return [], 0.0

    chosen: List[str] = []
    best_total = 0.0
    while len(chosen) < max_size:
        best_doc, best_score = None, best_total
        for doc in pool_nonrel:
            if doc in chosen:
                continue
            score = mean_drop(query, chosen + [doc], pool, qrels_q, clean_by_surrogate)
            if score > best_score + 1e-9:
                best_doc, best_score = doc, score
        if best_doc is None:
            break
        chosen.append(best_doc)
        best_total = best_score
    return chosen, best_total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/toy/corpus.jsonl")
    ap.add_argument("--queries", default="data/toy/queries_dev.tsv")
    ap.add_argument("--qrels", default="data/toy/qrels_dev.txt")
    ap.add_argument("--candidates", default="data/toy/candidates_dev.jsonl")
    ap.add_argument("--out", default="submission/adversarial_set.json")
    ap.add_argument("--max-size", type=int, default=5, help="docs per poisoned seed")
    ap.add_argument("--top-queries", type=int, default=5, help="how many queries to attack")
    ap.add_argument("--consider", type=int, default=40,
                    help="how many non-relevant pool docs to search over per query")
    args = ap.parse_args()

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    cands = read_candidates(args.candidates)

    fb.prepare(args.corpus)
    _snapshot()

    results = []
    for qid, text in queries:
        if qid not in cands or qid not in qrels:
            continue
        pool = [d for d, _s in cands[qid]]
        seed, drop = attack_query(text, pool, qrels[qid], args.max_size, args.consider)
        if seed and drop > 0:
            results.append((drop, qid, seed))
            print(f"{qid:<6} mean surrogate drop {drop:+.4f}  seed={seed}")
        else:
            print(f"{qid:<6} no profitable attack found")

    _configure({"name": "hardened", "LAMBDA_FIXED": None})
    results.sort(reverse=True)
    payload = {qid: seed for _drop, qid, seed in results[: args.top_queries]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {len(payload)} attacked queries to {args.out}")


if __name__ == "__main__":
    main()
