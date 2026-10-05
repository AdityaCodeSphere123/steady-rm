"""How often documents added to the candidate pool land in the base ranking's
top 10 (lambda=1, so Track B is the shared model without the rank prior)."""
import json, random, sys, zlib
from harness.metrics import ndcg_at_k
from submission import feedback as fb
from tools.track_a_sweep import load
from tools.pool_noise import corpus_ids

configs = json.load(open(sys.argv[1]))
base = dict({"ax_z": 1, "fb_prior_w": -1}, **fb.PARAMS)
for s in sys.argv[2:]:
    d = load(s); fb.set_params(**base); fb.prepare(d[0]); _, qs, qr, cands = d
    rid = corpus_ids(d[0], 20000)
    qids = [q for q, _ in qs]
    for name, c in configs.items():
        fb.set_params(**dict(base, **c))
        hits = {"rand": 0, "other": 0}; drop = {"rand": 0.0, "other": 0.0}; nq = 0
        for q, t in qs:
            pool = cands[q]; ps = set(pool)
            seed = [x for x, _ in fb.score_candidates(t, pool, 10)]
            clean = [x for x, _ in fb.relevance_model_feedback(t, seed, pool, 10)]
            c0 = ndcg_at_k(clean, qr[q], 10)
            rng = random.Random(zlib.crc32(q.encode()))
            inj = {"rand": [x for x in rng.sample(rid, 30) if x not in ps],
                   "other": [x for k in rng.sample(qids, 6) if k != q for x in cands[k][:10] if x not in ps][:30]}
            for kind, extra in inj.items():
                out = [x for x, _ in fb.relevance_model_feedback(t, seed, pool + extra, 10)]
                hits[kind] += any(x in set(extra) for x in out)
                drop[kind] += ndcg_at_k(out, qr[q], 10) - c0
            nq += 1
        print(f"{s:9s} {name:14s} queries with an added doc in top10: random {hits['rand']:3d}  "
              f"other-query {hits['other']:3d}   mean nDCG change: {drop['rand']/nq:+.5f} {drop['other']/nq:+.5f}", flush=True)
