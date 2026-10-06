"""Track A (and optionally clean Track B) of several configs on several sets,
with a paired bootstrap 95% CI of each config against the first one.

    python -m tools.compare_sets --sets scifact antique full \
        --cfg "base:" --cfg "tuned:spud_mu=1000 lead_w=0.05" [--fb]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import ndcg_at_k
from submission import feedback as fb
from tools.track_a_sweep import load


def per_query(data, fbk):
    _, qs, qr, cands = data
    a, b = [], []
    for q, t in qs:
        r = [d for d, _ in fb.score_candidates(t, cands[q], 10)]
        a.append(ndcg_at_k(r, qr[q], 10))
        if fbk:
            b.append(ndcg_at_k([d for d, _ in fb.relevance_model_feedback(t, r, cands[q], 10)], qr[q], 10))
    return a, b


def ci(x, y):
    d = [j - i for i, j in zip(x, y)]
    rng = random.Random(0)
    n = len(d)
    bs = sorted(sum(rng.choice(d) for _ in range(n)) / n for _ in range(2000))
    return sum(d) / n, bs[50], bs[1950]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--cfg", action="append", required=True, help='"name:k=v k=v"')
    ap.add_argument("--fb", action="store_true")
    a = ap.parse_args()
    cfgs = []
    for c in a.cfg:
        name, _, kv = c.partition(":")
        cfgs.append((name, {k: float(v) for k, v in (x.split("=") for x in kv.split())}))
    for s in a.sets:
        data = load(s)
        res = {}
        for name, p in cfgs:
            fb.set_params(**{**fb.PARAMS, **p})
            fb.prepare(data[0])
            res[name] = per_query(data, a.fb)
        ref = cfgs[0][0]
        for name, _ in cfgs:
            A, B = res[name]
            m, lo, hi = ci(res[ref][0], A)
            line = f"{s:9s} {name:10s} A {sum(A)/len(A):.4f} ({m:+.4f} [{lo:+.4f},{hi:+.4f}])"
            if a.fb:
                line += f"  B {sum(B)/len(B):.4f}  B-A {sum(B)/len(B) - sum(A)/len(A):+.4f}"
            print(line, flush=True)
    fb.set_params(**fb.PARAMS)


if __name__ == "__main__":
    main()
