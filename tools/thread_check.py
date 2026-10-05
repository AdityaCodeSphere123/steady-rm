"""Do concurrent calls (threads) give the same outputs as serial calls?

    python -m tools.thread_check scifact,antique
"""
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from harness import noise_injection
from submission import feedback as fb
from tools.track_a_sweep import load

SETS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["scifact", "antique"]
for s in SETS:
    d = load(s)
    _, qs, _, cands = d
    fb.prepare(d[0])
    jobs = []
    for q, t in qs:
        pool = cands[q]
        seed = [x for x, _ in fb.score_candidates(t, pool, 10)]
        jobs.append(("A", t, None, pool))
        jobs.append(("B", t, seed, pool))
        rng = random.Random(noise_injection.stable_seed(q, 0))
        for f in (0.25, 0.5, 1.0):
            jobs.append(("B", t, noise_injection.perturb_pseudo_relevant_set(seed, pool, f, rng), pool))

    def run(j):
        kind, t, seed, pool = j
        return fb.score_candidates(t, pool, 10) if kind == "A" else fb.relevance_model_feedback(t, seed, pool, 10)

    serial = [run(j) for j in jobs]
    for workers in (2, 8):
        bad = 0
        for rep in range(3):
            order = list(range(len(jobs)))
            random.Random(rep).shuffle(order)
            with ThreadPoolExecutor(workers) as ex:
                out = list(ex.map(lambda i: (i, run(jobs[i])), order))
            bad += sum(1 for i, r in out if [x for x, _ in r] != [x for x, _ in serial[i]])
        print(f"{s:10s} threads={workers}: {bad} of {3 * len(jobs)} calls ranked differently", flush=True)
