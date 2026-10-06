"""Track B gain and Track C retention proxies over several collections.

Per set and config:
  gain  = clean Track B nDCG - Track A nDCG          (must be > 0 everywhere)
  ret1  = mean nDCG over random pool swaps (25/50/75%, two draws) / clean B
  ret2  = mean nDCG over hard swaps (30/50%: best-ranked non-relevant pool
          docs outside the seed, from qrels) / clean B
Retention is not capped at 1, so values above 1 are visible.

    python -m tools.robust_tune --sets scifact antique full --grid lambda=0.9,0.95 beta=12,50
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


def conditions(qid, seed, pool, ranked_all, rel):
    rnd = []
    for base in (0, 1):
        rng = random.Random(noise_injection.stable_seed(qid, base))
        for f in (0.25, 0.5, 0.75):
            rnd.append(noise_injection.perturb_pseudo_relevant_set(seed, pool, f, rng))
    hard = []
    bad = [d for d in ranked_all if d not in set(seed) and rel.get(d, 0) <= 0]
    rng = random.Random(noise_injection.stable_seed(qid, 7))
    for f in (0.3, 0.5):
        s = list(seed)
        n = int(round(f * len(s)))
        for p, d in zip(rng.sample(range(len(s)), n), bad[:n]):
            s[p] = d
        hard.append(s)
    return rnd, hard


def evaluate(data):
    _, qs, qr, cands = data
    A = B = R1 = R2 = 0.0
    for q, t in qs:
        pool, rel = cands[q], qr[q]
        full = [d for d, _ in fb.score_candidates(t, pool, len(pool))]
        seed = full[:10]
        A += ndcg_at_k(seed, rel, 10)
        run = lambda s: ndcg_at_k([d for d, _ in fb.relevance_model_feedback(t, s, pool, 10)], rel, 10)
        B += run(seed)
        rnd, hard = conditions(q, seed, pool, full, rel)
        R1 += sum(run(s) for s in rnd) / len(rnd)
        R2 += sum(run(s) for s in hard) / len(hard)
    n = len(qs)
    return A / n, B / n, R1 / n, R2 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["scifact", "antique", "full", "nfcorpus", "vaswani"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--grid", nargs="*", default=[])
    a = ap.parse_args()
    base = {**fb.PARAMS, **{k: v[0] for k, v in parse_kv(a.set).items()}}
    grid = parse_kv(a.grid)
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys])) or [()]
    res = {}
    for s in a.sets:
        data = load(s)
        fb.set_params(**base)
        fb.prepare(data[0])
        for c in combos:
            fb.set_params(**{**base, **dict(zip(keys, c))})
            res.setdefault(c, {})[s] = evaluate(data)
    print(f"{'config':40s} {'minGain':>8s} {'meanGain':>8s} {'ret1':>7s} {'ret2':>7s} | per set gain/ret1/ret2")
    rows = []
    for c, per in res.items():
        g = [per[s][1] - per[s][0] for s in a.sets]
        r1 = [per[s][2] / per[s][1] for s in a.sets]
        r2 = [per[s][3] / per[s][1] for s in a.sets]
        tag = " ".join(f"{k}={v:g}" for k, v in zip(keys, c)) or "base"
        detail = " ".join(f"{s[:4]} {gg:+.4f}/{x:.4f}/{y:.4f}" for s, gg, x, y in zip(a.sets, g, r1, r2))
        rows.append((min(g), sum(g) / len(g), sum(r1) / len(r1), sum(r2) / len(r2), tag, detail))
    for mg, ag, r1, r2, tag, detail in sorted(rows, key=lambda r: -(r[2] + r[3])):
        print(f"{tag:40s} {mg:+8.4f} {ag:+8.4f} {r1:7.4f} {r2:7.4f} | {detail}", flush=True)
    fb.set_params(**fb.PARAMS)


if __name__ == "__main__":
    main()
