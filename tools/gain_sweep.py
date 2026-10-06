"""Track B - Track A gain only, over many collections (no noise runs).

Per set prints Track A once and the gain of each config; a summary ranks
configs by how many sets have positive gain, then by mean gain.

    python -m tools.gain_sweep --sets scifact fiqa --max-q 300 --grid lambda=0.9,0.96
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import ndcg_at_k
from submission import feedback as fb
from tools.track_a_sweep import load, parse_kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--grid", nargs="*", default=[])
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--depth", type=int, default=10, help="seed size")
    ap.add_argument("--pool-seed", action="store_true", help="seed from the first-pass pool order, not Track A")
    a = ap.parse_args()
    base = {**fb.PARAMS, **{k: v[0] for k, v in parse_kv(a.set).items()}}
    grid = parse_kv(a.grid)
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys])) or [()]
    tags = [" ".join(f"{k}={v:g}" for k, v in zip(keys, c)) or "base" for c in combos]
    gains = {t: {} for t in tags}
    for s in a.sets:
        corpus, qs, qr, cands = load(s)
        if a.max_q:
            qs = qs[:a.max_q]
        if not qs:
            print(f"{s}: no judged queries, skipped", flush=True)
            continue
        fb.set_params(**base)
        fb.prepare(corpus)
        cell = []
        for c, tag in zip(combos, tags):
            fb.set_params(**{**base, **dict(zip(keys, c))})
            A = B = 0.0
            for q, t in qs:
                top = [d for d, _ in fb.score_candidates(t, cands[q], max(10, a.depth))]
                A += ndcg_at_k(top[:10], qr[q], 10)
                seed = cands[q][:a.depth] if a.pool_seed else top[:a.depth]
                B += ndcg_at_k([d for d, _ in fb.relevance_model_feedback(t, seed, cands[q], 10)], qr[q], 10)
            A, B = A / len(qs), B / len(qs)
            gains[tag][s] = B - A
            cell.append(f"[{tag}] A {A:.4f} {B - A:+.4f}")
        print(f"{s:10s} n={len(qs):4d} | " + "  ".join(cell), flush=True)
    print()
    rows = []
    for t in tags:
        g = list(gains[t].values())
        rows.append((sum(x > 0 for x in g), sum(g) / len(g), min(g), t, g))
    for pos, mean, mn, t, g in sorted(rows, key=lambda r: (-r[0], -r[1])):
        print(f"{t:40s} pos {pos}/{len(g)}  mean {mean:+.5f}  min {mn:+.4f}")
    fb.set_params(**fb.PARAMS)


if __name__ == "__main__":
    main()
