"""Track B must not depend on the supplied seed at all (fb_lock). For every
query, try many seeds against the same pool and count differing outputs."""
import random, sys, zlib
from submission import feedback as fb
from tools.track_a_sweep import load
from tools.pool_noise import corpus_ids

for s in sys.argv[1:] or ["scifact", "antique"]:
    data = load(s); fb.prepare(data[0]); _, qs, qr, cands = data
    rand_ids = corpus_ids(data[0], 5000)
    bad = {}
    for q, t in qs:
        pool = cands[q]
        a10 = [d for d, _ in fb.score_candidates(t, pool, 10)]
        a20 = [d for d, _ in fb.score_candidates(t, pool, 20)]
        ref = fb.relevance_model_feedback(t, a10, pool, 10)
        rng = random.Random(zlib.crc32(q.encode()))
        seeds = {
            "empty": [], "one": a10[:1], "top5": a10[:5], "top20": a20, "pool": list(pool),
            "unknown": ["zz_nope_1", "zz_nope_2"], "dups": a10 + a10, "reversed": a10[::-1],
            "outside": rng.sample(rand_ids, 10), "mixed": a10[:5] + rng.sample(rand_ids, 5),
            "tail": pool[-10:], "rand": rng.sample(pool, min(10, len(pool))),
            "pool_order_top10": pool[:10],
        }
        for name, sd in seeds.items():
            if fb.relevance_model_feedback(t, sd, pool, 10) != ref:
                bad[name] = bad.get(name, 0) + 1
        for k in (5, 20, 100):
            r = fb.relevance_model_feedback(t, a10, pool, k)
            if [d for d, _ in r[:min(k, 10)]] != [d for d, _ in ref[:min(k, 10)]]:
                bad[f"k{k}"] = bad.get(f"k{k}", 0) + 1
    print(s, len(qs), "queries; differing outputs by seed type:", bad or "none")
