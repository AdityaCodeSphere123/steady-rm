"""Coordinate ascent on a weighted set of collections (default 2*scifact +
1*antique), Track A or Track B, starting from submission/feedback.PARAMS.
Each parameter's values are scored on every set with one prepare() per set.

    python -m tools.sa_tune --track a --rounds 2
    python -m tools.sa_tune --track b --sets scifact:2 antique:1 full:0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import noise_injection
from harness.metrics import average_precision, ndcg_at_k
from submission import feedback as fb
from tools.track_a_sweep import load

GRID_A = [
    ("spud_mu", [200, 300, 400, 500, 600, 800, 1000, 1400]),
    ("lead_w", [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]),
    ("lead_len", [6, 8, 10, 12, 16, 20, 30]),
    ("lead_mu", [10, 20, 30, 60, 120]),
    ("w_od", [0, 0.02, 0.05, 0.1, 0.15]),
    ("w_uw", [0, 0.02, 0.05, 0.1, 0.15]),
    ("uw_window", [4, 8, 12, 16]),
    ("mu", [300, 550, 1000, 2000]),
    ("len_prior", [-0.1, -0.05, 0, 0.05, 0.1, 0.2]),
    ("query_stop", [0, 1, 2]),
    ("qstop_list", [0, 1, 2, 3]),
    ("spud_bg", [0, 1]),
]
GRID_B = [
    ("lambda", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    ("beta", [1, 3, 5, 8, 12]),
    ("fb_terms", [5, 10, 15, 25, 40]),
    ("fb_docs", [3, 5, 7, 10]),
    ("fb_mu", [0, 100, 250, 500, 1000]),
    ("idf_power", [0, 0.25, 0.5, 1.0]),
    ("support_frac", [0, 0.1, 0.2, 0.3]),
    ("fb_cov", [0, 0.25, 0.5]),
    ("fb_qterms", [0, 1]),
    ("fb_sdm", [0, 1]),
]


def score(data, track):
    """(Track A nDCG, MAP, clean Track B nDCG, Track B nDCG at 50% practice noise)."""
    _, qs, qr, cands = data
    a = m = b = bn = 0.0
    for q, t in qs:
        r = [d for d, _ in fb.score_candidates(t, cands[q], 10)]
        a += ndcg_at_k(r, qr[q], 10)
        m += average_precision(r, qr[q])
        if track == "b":
            b += ndcg_at_k([d for d, _ in fb.relevance_model_feedback(t, r, cands[q], 10)], qr[q], 10)
            noisy = dict(noise_injection.sweep(r, cands[q], seed=noise_injection.stable_seed(q, 0)))[0.5]
            bn += ndcg_at_k([d for d, _ in fb.relevance_model_feedback(t, noisy, cands[q], 10)], qr[q], 10)
    n = len(qs)
    return a / n, m / n, b / n, bn / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["a", "b"], default="a")
    ap.add_argument("--sets", nargs="+", default=["scifact:2", "antique:1", "full:0"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these parameters")
    ap.add_argument("--set", nargs="*", default=[], help="start point overrides k=v")
    a = ap.parse_args()
    sets = [(s.split(":")[0], float(s.split(":")[1]) if ":" in s else 1.0) for s in a.sets]
    datas = {s: load(s) for s, _ in sets}
    grid = [g for g in (GRID_A if a.track == "a" else GRID_B) if a.only is None or g[0] in a.only]
    cur = {**fb.PARAMS, **{k: float(v) for k, v in (x.split("=", 1) for x in a.set)}}
    wsum = sum(w for _, w in sets) or 1.0

    def run(cands_for_param):
        out = {}
        for s, w in sets:
            for key, p in cands_for_param:
                fb.set_params(**p)
                fb.prepare(datas[s][0])
                out.setdefault(key, {})[s] = score(datas[s], a.track)
        return out

    def obj(per):
        if a.track == "a":
            return sum(w * per[s][0] for s, w in sets) / wsum
        return sum(w * 0.5 * (per[s][2] + per[s][3]) for s, w in sets) / wsum

    def show(tag, per):
        cols = " ".join(f"{s} {per[s][0]:.4f}/{per[s][2]:.4f}/{per[s][3]:.4f}" for s, _ in sets)
        print(f"  {tag:28s} obj {obj(per):.5f} | A/B {cols}", flush=True)

    base = run([("base", cur)])["base"]
    show("start", base)
    best = obj(base)
    for r in range(a.rounds):
        changed = False
        for name, vals in grid:
            vals = sorted(set(vals) | {cur[name]} if name in cur else set(vals))
            res = run([(v, {**cur, name: v}) for v in vals])
            print(f"# r{r} {name}", flush=True)
            for v in vals:
                show(f"{name}={v:g}", res[v])
            v_best = max(vals, key=lambda v: obj(res[v]))
            if obj(res[v_best]) > best + 1e-5:
                print(f"  -> {name}: {cur.get(name)} -> {v_best:g}  obj {best:.5f} -> {obj(res[v_best]):.5f}", flush=True)
                cur[name] = v_best
                best = obj(res[v_best])
                changed = True
        if not changed:
            break
    keys = [g[0] for g in grid]
    print("FINAL " + " ".join(f'"{k}": {cur[k]:g},' for k in keys))
    fb.set_params(**fb.PARAMS)


if __name__ == "__main__":
    main()
