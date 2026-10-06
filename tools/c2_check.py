"""Hard-noise (Track C2) stability check.

A fraction of the clean seed is replaced, at random positions, by the
best-ranked judged non-relevant pool docs outside the seed. Reports gain
(B - A), mean per-run capped retention min(1, noisy/clean), and the share of
noisy runs whose top 10 differs from the clean one.

    python -m tools.c2_check --sets full scifact --grid beta=20,50
"""
import argparse
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import noise_injection
from harness.metrics import ndcg_at_k
from submission import feedback as fb
from tools.track_a_sweep import load, parse_kv

FRACS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--grid", nargs="*", default=[])
    ap.add_argument("--max-q", type=int, default=300)
    a = ap.parse_args()
    base = {**fb.PARAMS, **{k: v[0] for k, v in parse_kv(a.set).items()}}
    grid = parse_kv(a.grid)
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys])) or [()]
    print(f"{'set':6s} {'config':40s} | gain / capped C2 / changed-top10")
    for s in a.sets:
        corpus, qs, qr, cands = load(s)
        qs = qs[:a.max_q]
        fb.set_params(**base)
        fb.prepare(corpus)
        for c in combos:
            fb.set_params(**{**base, **dict(zip(keys, c))})
            A = B = ret = 0.0
            runs = changed = 0
            for q, t in qs:
                pool, rel = cands[q], qr[q]
                ranked = [d for d, _ in fb.score_candidates(t, pool, len(pool))]
                seed = ranked[:10]
                A += ndcg_at_k(seed, rel, 10)
                clean = [d for d, _ in fb.relevance_model_feedback(t, seed, pool, 10)]
                cb = ndcg_at_k(clean, rel, 10)
                B += cb
                bad = [d for d in ranked[10:] if rel.get(d, 0) <= 0]
                rng = random.Random(noise_injection.stable_seed(q, 11))
                for f in FRACS:
                    n = min(len(bad), int(round(f * len(seed))))
                    noisy = list(seed)
                    for p, d in zip(rng.sample(range(len(seed)), n), bad[:n]):
                        noisy[p] = d
                    r = [d for d, _ in fb.relevance_model_feedback(t, noisy, pool, 10)]
                    nb = ndcg_at_k(r, rel, 10)
                    ret += min(1.0, nb / cb) if cb > 0 else 1.0
                    changed += r != clean
                    runs += 1
            n = len(qs)
            tag = " ".join(f"{k}={v:g}" for k, v in zip(keys, c)) or "base"
            print(f"{s[:6]:6s} {tag:40s} | A {A/n:.4f} gain {(B-A)/n:+.4f}  C2 {ret/runs:.5f}  changed {changed/runs:.3f}", flush=True)
    fb.set_params(**fb.PARAMS)


if __name__ == "__main__":
    main()
