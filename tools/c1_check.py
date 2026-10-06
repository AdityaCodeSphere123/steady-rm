"""Random-noise (Track C1) stability check.

For every query and config, the clean Track B top 10 is compared with the
top 10 under random seed noise at many fractions, both in-pool swaps and
outside-corpus docs appended to the pool. Reports gain (B - A), mean
per-run capped retention min(1, noisy/clean), and the share of noisy runs
whose top 10 differs from the clean one.

    python -m tools.c1_check --sets scifact antique full --grid beta=12,40
"""
import argparse
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import noise_injection
from harness.metrics import ndcg_at_k
from submission import feedback as fb
from tools.track_a_sweep import load, parse_kv

FRACS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


def noisy_runs(qid, seed, pool, corpus_ids):
    pset = set(pool)
    out = []
    for base in (0, 1):
        rng = random.Random(noise_injection.stable_seed(qid, base))
        for f in FRACS:
            out.append((noise_injection.perturb_pseudo_relevant_set(seed, pool, f, rng), pool))
        rng = random.Random(noise_injection.stable_seed(qid, 10 + base))
        outside = [d for d in rng.sample(corpus_ids, min(len(corpus_ids), 60)) if d not in pset]
        for f in FRACS:
            s = list(seed)
            n = min(int(round(f * len(s))), len(outside))
            for p, d in zip(rng.sample(range(len(s)), n), rng.sample(outside, n)):
                s[p] = d
            out.append((s, pool + [d for d in s if d not in pset]))
    return out


def evaluate(data, ids):
    _, qs, qr, cands = data
    A = B = ret = 0.0
    changed = total = 0
    for q, t in qs:
        pool, rel = cands[q], qr[q]
        seed = [d for d, _ in fb.score_candidates(t, pool, 10)]
        A += ndcg_at_k(seed, rel, 10)
        clean = [d for d, _ in fb.relevance_model_feedback(t, seed, pool, 10)]
        cb = ndcg_at_k(clean, rel, 10)
        B += cb
        r = 0.0
        runs = noisy_runs(q, seed, pool, ids)
        for s, p in runs:
            top = [d for d, _ in fb.relevance_model_feedback(t, s, p, 10)]
            changed += top != clean
            total += 1
            r += min(1.0, ndcg_at_k(top, rel, 10) / cb) if cb > 0 else 1.0
        ret += r / len(runs)
    n = len(qs)
    return A / n, B / n, ret / n, changed / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["scifact", "antique", "full"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--grid", nargs="*", default=[])
    a = ap.parse_args()
    base = {**fb.PARAMS, **{k: v[0] for k, v in parse_kv(a.set).items()}}
    grid = parse_kv(a.grid)
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys])) or [()]
    print(f"{'config':40s} | per set gain / capped C1 / changed-top10")
    for s in a.sets:
        data = load(s)
        fb.set_params(**base)
        fb.prepare(data[0])
        ids = [json.loads(l)["doc_id"] for l in open(data[0]) if l.strip()]
        for c in combos:
            fb.set_params(**{**base, **dict(zip(keys, c))})
            A, B, r, ch = evaluate(data, ids)
            tag = " ".join(f"{k}={v:g}" for k, v in zip(keys, c)) or "base"
            print(f"{s[:6]:6s} {tag:33s} | A {A:.4f} gain {B - A:+.4f}  C1 {r:.5f}  changed {ch:.3f}", flush=True)
    fb.set_params(**fb.PARAMS)


if __name__ == "__main__":
    main()
