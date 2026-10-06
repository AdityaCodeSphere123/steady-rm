"""Track A sweep: prepare each collection once, then score parameter sets.

    python -m tools.track_a_sweep --sets full scifact --grid smoothing=0,2 jm_lambda=0.1,0.3
    python -m tools.track_a_sweep --sets full --set mu=550 --fb

Each line: per-set Track A nDCG@10 / MAP@10 (and clean Track B nDCG@10 with --fb).
"""
import argparse
import itertools
import os
import random
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import average_precision, ndcg_at_k
from submission import feedback as fb
from tools.sweep import read_candidates, read_qrels, read_queries

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def map10(ranked, qrels_q):
    return average_precision(ranked[:10], qrels_q)


def load(name):
    half = None
    if "@" in name:
        name, half = name.split("@")
    weak = name.endswith("_weak")
    d = os.path.join(ROOT, name[:-5] if weak else name)
    qs = read_queries(os.path.join(d, "queries_dev.tsv"))
    qr = read_qrels(os.path.join(d, "qrels_dev.txt"))
    cands = read_candidates(os.path.join(d, "candidates_weak.jsonl" if weak else "candidates_dev.jsonl"))
    qs = [(q, t) for q, t in qs if q in cands and q in qr]
    if half:
        qs = [x for i, x in enumerate(qs) if (i % 2 == 0) == (half == "even")]
    return os.path.join(d, "corpus.jsonl"), qs, qr, cands


def evaluate(data, with_fb=False, ret=False):
    """Track A nDCG/MAP, clean Track B nDCG, and with ret: the share of
    queries whose Track B top 10 moves under pool and seed perturbations,
    and mean noisy/clean Track B nDCG."""
    _, qs, qr, cands = data
    n = m = b = 0.0
    moved = shuffled = 0
    noisy = 0.0
    qids = [q for q, _ in qs]
    for qi, (q, t) in enumerate(qs):
        pool = cands[q]
        ranked = [d for d, _ in fb.score_candidates(t, pool, 10)]
        n += ndcg_at_k(ranked, qr[q], 10)
        m += map10(ranked, qr[q])
        if with_fb or ret:
            fr = [d for d, _ in fb.relevance_model_feedback(t, ranked, pool, 10)]
            b += ndcg_at_k(fr, qr[q], 10)
        if ret:
            rng = random.Random(zlib.crc32(q.encode()))
            other = cands[qids[(qi + 1) % len(qids)]]
            conds = []
            p1 = list(pool); rng.shuffle(p1)
            conds.append((p1, rng.sample(pool, min(10, len(pool)))))
            extra = [d for d in other if d not in set(pool)][:5]
            p2 = pool + extra; rng.shuffle(p2)
            conds.append((p2, ranked[:5] + extra))
            p3 = pool[: max(10, len(pool) - 5)]
            conds.append((p3, list(reversed(ranked))))
            changed = False
            for ci, (pp, seed) in enumerate(conds):
                nr = [d for d, _ in fb.relevance_model_feedback(t, seed, pp, 10)]
                changed |= nr != fr
                if ci == 0:
                    shuffled += nr != fr
                noisy += ndcg_at_k(nr, qr[q], 10) / len(conds)
            moved += changed
    k = len(qs)
    out = (n / k, m / k, (b / k if (with_fb or ret) else None))
    if ret:
        out += (moved / k, (noisy / k) / (b / k) if b > 0 else 1.0, shuffled)
    return out


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = [float(x) for x in v.split(",")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["full"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--grid", nargs="*", default=[])
    ap.add_argument("--fb", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ret", action="store_true", help="also run the pool/seed perturbation check")
    ap.add_argument("--ref", default=None, help="tag of the reference combo (default: first)")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    fixed = {k: v[0] for k, v in parse_kv(a.set).items()}
    grid = parse_kv(a.grid)
    base = dict(fb.PARAMS)
    base.update(fixed)
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys])) if keys else [()]
    res = {}
    for s in a.sets:
        data = load(s)
        fb.set_params(**base)
        t0 = time.time()
        fb.prepare(data[0])
        print(f"# {s}: {len(data[1])} queries, prepare {time.time() - t0:.1f}s", flush=True)
        for combo in combos:
            p = dict(base)
            p.update(dict(zip(keys, combo)))
            fb.set_params(**p)
            r = evaluate(data, a.fb, a.ret)
            n, m, b = r[:3]
            tag = " ".join(f"{k}={v:g}" for k, v in zip(keys, combo)) or "base"
            res.setdefault(tag, {})[s] = (n, m)
            if a.verbose or a.ret:
                extra = f" moved {r[3]:.3f} ratio {r[4]:.5f} shuffle-moved {r[5]}" if a.ret else ""
                print(f"{s:10s} {tag:50s} {n:.5f} {m:.5f}" + (f" {b:.5f}" if b is not None else "") + extra,
                      flush=True)
    base_tag = a.ref or next(iter(res))
    rows = []
    for tag, per in res.items():
        dn = [per[s][0] - res[base_tag][s][0] for s in a.sets]
        dm = [per[s][1] - res[base_tag][s][1] for s in a.sets]
        rows.append((sum(dn) / len(dn), sum(dm) / len(dm), min(dn), tag, dn))
    rows.sort(reverse=True)
    print(f"\nreference: {base_tag}   sets: {' '.join(a.sets)}")
    for mn, mm, worst, tag, dn in rows[: a.top]:
        print(f"{tag:55s} dN {mn:+.5f} dMAP {mm:+.5f} worst {worst:+.5f} | " + " ".join(f"{x:+.4f}" for x in dn))


if __name__ == "__main__":
    main()
