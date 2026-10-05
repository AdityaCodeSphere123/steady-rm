"""How often Track B's top 10 moves when the candidate pool itself changes.

Conditions per query (seed is always a noisy one; fb_lock ignores it):
  rand+10   10 random corpus documents added, pool shuffled
  other+10  10 documents from other queries' pools added
  drop5     5 random pool documents outside the clean Track B top 10 removed
  seedin    the 30%-swapped seed's outside documents added to the pool

    python -m tools.pool_noise configs.json --sets scifact antique
"""
import argparse
import json
import random
import sys
import zlib

from harness.metrics import ndcg_at_k
from submission import feedback as fb
from tools.track_a_sweep import load

COND = ["rand+10", "other+10", "drop5", "seedin"]


def corpus_ids(path, n=20000, seed=7):
    ids = []
    with open(path) as fh:
        for line in fh:
            i = line.find('"doc_id"')
            if i < 0:
                continue
            j = line.index('"', line.index(":", i) + 1) + 1
            ids.append(line[j:line.index('"', j)])
    rng = random.Random(seed)
    return rng.sample(ids, min(n, len(ids)))


def run(data, rand_ids):
    _, qs, qr, cands = data
    qids = [q for q, _ in qs]
    moved = {c: 0 for c in COND}
    dn = {c: 0.0 for c in COND}
    clean = 0.0
    for qi, (q, t) in enumerate(qs):
        pool = cands[q]
        seed0 = [d for d, _ in fb.score_candidates(t, pool, 10)]
        fr = [d for d, _ in fb.relevance_model_feedback(t, seed0, pool, 10)]
        c0 = ndcg_at_k(fr, qr[q], 10)
        clean += c0
        rng = random.Random(zlib.crc32(q.encode()))
        ps = set(pool)
        conds = {}
        p = pool + [d for d in rng.sample(rand_ids, 10) if d not in ps]
        rng.shuffle(p)
        conds["rand+10"] = p
        other = [d for k in rng.sample(qids, min(5, len(qids))) for d in cands[k][:20] if d not in ps]
        p = pool + other[:10]
        rng.shuffle(p)
        conds["other+10"] = p
        tail = [d for d in pool if d not in set(fr)]
        drop = set(rng.sample(tail, min(5, len(tail))))
        conds["drop5"] = [d for d in pool if d not in drop]
        outside = [d for d in rng.sample(rand_ids, 3) if d not in ps]
        conds["seedin"] = pool + outside
        for c, pp in conds.items():
            seed = seed0[:7] + [d for d in pp if d not in ps][:3]
            nr = [d for d, _ in fb.relevance_model_feedback(t, seed, pp, 10)]
            moved[c] += nr != fr
            dn[c] += ndcg_at_k(nr, qr[q], 10) - c0
    k = len(qs)
    return clean / k, moved, {c: v / k for c, v in dn.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs")
    ap.add_argument("--sets", nargs="+", default=["scifact", "antique"])
    a = ap.parse_args()
    configs = json.load(open(a.configs))
    base = dict({"ax_z": 1, "fb_prior_w": -1}, **fb.PARAMS)
    for s in a.sets:
        data = load(s)
        fb.set_params(**base)
        fb.prepare(data[0])
        rand_ids = corpus_ids(data[0])
        print(f"# {s} ({len(data[1])} queries)   moved queries per condition, mean nDCG change")
        for name, c in configs.items():
            fb.set_params(**dict(base, **c))
            b, moved, dn = run(data, rand_ids)
            print(f"{s:9s} {name:22s} B {b:.4f}  " +
                  "  ".join(f"{k} {moved[k]:3d} ({dn[k]:+.5f})" for k in COND), flush=True)


if __name__ == "__main__":
    sys.exit(main())
