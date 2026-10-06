"""Add a few documents of a given kind to every pool and count how often one
of them enters the Track B top 10 (and the nDCG change).

    python -m tools.inject_long configs.json scifact,antique
"""
import json
import random
import sys
import zlib

from harness.metrics import ndcg_at_k
from submission import feedback as fb
from tools.track_a_sweep import load

configs = json.load(open(sys.argv[1]))
SETS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["scifact", "antique"]
base = dict({"ax_z": 1}, **fb.PARAMS)


def corpus(path):
    out = []
    with open(path) as fh:
        for line in fh:
            o = json.loads(line)
            did = o.get("doc_id") or o.get("_id") or o.get("id")
            out.append((did, len(((o.get("title") or "") + " " + (o.get("text") or "")).split())))
    return out


for s in SETS:
    d = load(s)
    _, qs, qr, cands = d
    fb.set_params(**base)
    fb.prepare(d[0])
    docs = corpus(d[0])
    by_len = sorted(docs, key=lambda x: -x[1])
    kinds = {
        "longest": [x for x, _ in by_len[:200]],
        "shortest": [x for x, _ in by_len[-2000:]],
        "random": [x for x, _ in random.Random(1).sample(docs, 2000)],
    }
    for name, c in configs.items():
        fb.set_params(**dict(base, **c))
        row = []
        for kind, ids in kinds.items():
            hit = 0
            dn = 0.0
            for q, t in qs:
                pool = cands[q]
                seed = [x for x, _ in fb.score_candidates(t, pool, 10)]
                ref = [x for x, _ in fb.relevance_model_feedback(t, seed, pool, 10)]
                rng = random.Random(zlib.crc32(q.encode()))
                ps = set(pool)
                add = [x for x in rng.sample(ids, 10) if x not in ps]
                pp = pool + add
                out = [x for x, _ in fb.relevance_model_feedback(t, seed, pp, 10)]
                hit += any(x in set(add) for x in out)
                dn += ndcg_at_k(out, qr[q], 10) - ndcg_at_k(ref, qr[q], 10)
            row.append(f"{kind} {hit:3d} ({dn / len(qs):+.5f})")
        print(f"{s:9s} {name:14s} " + "  ".join(row), flush=True)
