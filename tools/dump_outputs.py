"""Dump every Track A / Track B output (clean, harness noise, odd seeds) to a
file, so two builds of the engine can be diffed.

    python -m tools.dump_outputs out.txt scifact,antique
"""
import random
import sys

from harness import noise_injection
from submission import feedback as fb
from tools.track_a_sweep import load

out = open(sys.argv[1], "w")
SETS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["scifact", "antique"]
print(fb.backend(), file=sys.stderr)
for s in SETS:
    d = load(s)
    _, qs, _, cands = d
    fb.prepare(d[0])
    for q, t in qs:
        pool = cands[q]
        a = fb.score_candidates(t, pool, 10)
        seed = [x for x, _ in a]
        out.write(f"{s} {q} A {a!r}\n")
        out.write(f"{s} {q} B {fb.relevance_model_feedback(t, seed, pool, 10)!r}\n")
        rng = random.Random(noise_injection.stable_seed(q, 0))
        for f in (0.1, 0.3, 0.5, 0.7, 1.0):
            p = noise_injection.perturb_pseudo_relevant_set(seed, pool, f, rng)
            out.write(f"{s} {q} N{f} {fb.relevance_model_feedback(t, p, pool, 10)!r}\n")
        out.write(f"{s} {q} R {fb.relevance_model_feedback(t, pool[:10], pool, 10)!r}\n")
out.close()
