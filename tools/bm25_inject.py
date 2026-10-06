"""Add the BM25 documents ranked just outside each candidate pool ("plausible
but wrong" noise from a different reference ranking) and measure how the
Track B top 10 and nDCG move, per config.

    python -m tools.bm25_inject configs.json scifact,antique [n_add]
"""
import json
import math
import random
import re
import sys
import zlib
from collections import Counter, defaultdict

from harness.metrics import ndcg_at_k
from submission import feedback as fb
from tools.track_a_sweep import load

configs = json.load(open(sys.argv[1]))
SETS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["scifact", "antique"]
N_ADD = int(sys.argv[3]) if len(sys.argv) > 3 else 10
base = dict({"ax_z": 1}, **fb.PARAMS)
TOK = re.compile(r"[a-z0-9]+")
STOP = set("a an the of and or in on to for with is are was were be by as at from that this it its which".split())


def index(path):
    ids, lens, post = [], [], defaultdict(list)
    with open(path) as fh:
        for line in fh:
            o = json.loads(line)
            ids.append(o.get("doc_id") or o.get("_id") or o.get("id"))
            toks = TOK.findall(((o.get("title") or "") + " " + (o.get("text") or "")).lower())
            lens.append(len(toks))
            for t, c in Counter(toks).items():
                post[t].append((len(ids) - 1, c))
    return ids, lens, post


def bm25(q, ids, lens, post, avg, n=400, k1=0.9, b=0.4):
    N = len(ids)
    sc = defaultdict(float)
    for t in set(TOK.findall(q.lower())) - STOP:
        pl = post.get(t)
        if not pl:
            continue
        idf = math.log(1 + (N - len(pl) + 0.5) / (len(pl) + 0.5))
        for d, tf in pl:
            sc[d] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lens[d] / avg))
    return [ids[d] for d, _ in sorted(sc.items(), key=lambda x: -x[1])[:n]]


for s in SETS:
    d = load(s)
    _, qs, qr, cands = d
    ids, lens, post = index(d[0])
    avg = sum(lens) / len(lens)
    near = {}
    for q, t in qs:
        ps = set(cands[q])
        near[q] = [x for x in bm25(t, ids, lens, post, avg) if x not in ps]
    fb.set_params(**base)
    fb.prepare(d[0])
    for name, c in configs.items():
        fb.set_params(**dict(base, **c))
        moved = 0
        dn = 0.0
        dn_rand = 0.0
        moved_rand = 0
        clean_sum = 0.0
        for q, t in qs:
            pool = cands[q]
            seed = [x for x, _ in fb.score_candidates(t, pool, 10)]
            ref = [x for x, _ in fb.relevance_model_feedback(t, seed, pool, 10)]
            c0 = ndcg_at_k(ref, qr[q], 10)
            clean_sum += c0
            add = near[q][:N_ADD]
            out = [x for x, _ in fb.relevance_model_feedback(t, seed, pool + add, 10)]
            moved += out != ref
            dn += ndcg_at_k(out, qr[q], 10) - c0
            rng = random.Random(zlib.crc32(q.encode()))
            add2 = rng.sample(near[q][:100], min(N_ADD, len(near[q][:100])))
            p2 = pool + add2
            rng.shuffle(p2)
            out2 = [x for x, _ in fb.relevance_model_feedback(t, seed, p2, 10)]
            moved_rand += out2 != ref
            dn_rand += ndcg_at_k(out2, qr[q], 10) - c0
        k = len(qs)
        print(f"{s:9s} {name:14s} B {clean_sum / k:.4f}  next{N_ADD}: moved {moved:3d} dnDCG {dn / k:+.5f}   "
              f"sample-of-next100+shuffle: moved {moved_rand:3d} dnDCG {dn_rand / k:+.5f}", flush=True)
