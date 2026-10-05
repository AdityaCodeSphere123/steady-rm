"""Worst-case wall time per call, in the harness call order.

    python -m tools.call_timing scifact,antique
"""
import random
import sys
import time

from harness import noise_injection
from submission import feedback as fb
from tools.track_a_sweep import load

SETS = sys.argv[1].split(",") if len(sys.argv) > 1 else \
    "scifact nfcorpus fiqa vaswani antique touche2020 scidocs cacm cranfield arguana full quora".split()

for s in SETS:
    d = load(s)
    _, qs, _, cands = d
    t0 = time.perf_counter(); fb.prepare(d[0]); tp = time.perf_counter() - t0
    worst = []
    for q, t in qs:
        pool = [x for x, _ in cands[q]] if cands[q] and isinstance(cands[q][0], (list, tuple)) else cands[q]
        times = []
        t0 = time.perf_counter(); fb.score_candidates(t, pool, 10); times.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); seed = [x for x, _ in fb.score_candidates(t, pool, 10)]; times.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); fb.relevance_model_feedback(t, seed, pool, 10); times.append(time.perf_counter() - t0)
        rng = random.Random(noise_injection.stable_seed(q, 0))
        for f in (0.1, 0.25, 0.5, 0.75, 1.0):
            p = noise_injection.perturb_pseudo_relevant_set(seed, pool, f, rng)
            t0 = time.perf_counter(); fb.relevance_model_feedback(t, p, pool, 10); times.append(time.perf_counter() - t0)
        worst.append((max(times), q, len(t.split()), times.index(max(times))))
    worst.sort(reverse=True)
    print(f"{s:10s} prepare {tp:6.1f}s  q={len(qs)}  worst " +
          "  ".join(f"{w*1000:.0f}ms(q{q},len{l},call{c})" for w, q, l, c in worst[:3]), flush=True)
