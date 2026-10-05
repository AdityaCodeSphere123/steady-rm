"""Clean vs noisy Track B with harness.noise_injection, full (id, score) lists."""
import random, sys
from harness.noise_injection import perturb_pseudo_relevant_set as noisy
from submission import feedback as fb
from tools.track_a_sweep import load

for s in sys.argv[1:]:
    d = load(s); fb.prepare(d[0]); _, qs, qr, cands = d
    bad, calls, expl = [], 0, 0
    for q, t in qs:
        pool = cands[q]
        clean = [x for x, _ in fb.score_candidates(t, pool, 10)]
        base = fb.relevance_model_feedback(t, clean, pool, 10)
        e0 = fb.explain_feedback(t, clean, pool)
        for frac in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
            for i in range(10):
                ns = noisy(clean, pool, frac, random.Random(i))
                if i % 3 == 0:
                    ns = ns + ns[:3]                       # duplicates
                out = fb.relevance_model_feedback(t, ns, pool, 10)
                calls += 1
                if out != base:
                    bad.append((q, frac, i, sum(a[0] != b[0] for a, b in zip(out, base))))
                if i == 0:
                    e1 = fb.explain_feedback(t, ns, pool)
                    expl += (e1.get("seed_weights"), e1.get("lambda")) != (e0.get("seed_weights"), e0.get("lambda"))
    print(f"{s}: {calls} noisy calls, differing lists (ids or scores): {len(bad)}, "
          f"explain() seed weights/lambda differ: {expl}", bad[:5])
