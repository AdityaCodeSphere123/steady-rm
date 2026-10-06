"""Per-query comparison of Track A configs on data/full (dev pool and weak
pool), with a paired bootstrap and split-half check.

    python -m tools.robust_check --a mu=550 --b smoothing=3 spud_mu=500 lead_w=0.25
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import ndcg_at_k
from submission import feedback as fb
from tools.sweep import read_candidates, read_qrels, read_queries

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "full")


def kv(items):
    return {k: float(v) for k, v in (x.split("=", 1) for x in items)}


def per_query(params, qs, qr, cands):
    fb.set_params(**{**fb.PARAMS, **params})
    return [ndcg_at_k([d for d, _ in fb.score_candidates(t, cands[q], 10)], qr[q], 10) for q, t in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="*", default=[])
    ap.add_argument("--b", nargs="*", default=[])
    a = ap.parse_args()
    qs = read_queries(os.path.join(D, "queries_dev.tsv"))
    qr = read_qrels(os.path.join(D, "qrels_dev.txt"))
    fb.prepare(os.path.join(D, "corpus.jsonl"))
    for pool in ("candidates_dev.jsonl", "candidates_weak.jsonl"):
        cands = read_candidates(os.path.join(D, pool))
        qq = [(q, t) for q, t in qs if q in cands and q in qr]
        x, y = per_query(kv(a.a), qq, qr, cands), per_query(kv(a.b), qq, qr, cands)
        n = len(qq)
        diff = [j - i for i, j in zip(x, y)]
        rng = random.Random(0)
        boot = sorted(sum(rng.choice(diff) for _ in range(n)) / n for _ in range(4000))
        h = n // 2
        print(f"{pool}: A {sum(x)/n:.4f}  B {sum(y)/n:.4f}  d {sum(diff)/n:+.4f}  "
              f"wins {sum(d > 1e-9 for d in diff)} losses {sum(d < -1e-9 for d in diff)}  "
              f"95% CI [{boot[100]:+.4f}, {boot[3900]:+.4f}]  halves {sum(diff[:h])/h:+.4f} {sum(diff[h:])/(n-h):+.4f}")


if __name__ == "__main__":
    main()
