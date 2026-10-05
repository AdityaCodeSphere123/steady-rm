"""Track B for a query must not depend on which calls came before it."""
import random, sys
from submission import feedback as fb
from tools.track_a_sweep import load

for s in sys.argv[1:]:
    d = load(s); fb.prepare(d[0]); _, qs, qr, cands = d
    seeds = {q: [x for x, _ in fb.score_candidates(t, cands[q], 10)] for q, t in qs}
    ref = {q: fb.relevance_model_feedback(t, seeds[q], cands[q], 10) for q, t in qs}
    order = list(qs); random.Random(3).shuffle(order)
    diff = 0
    for q, t in order:
        fb.score_candidates(t, cands[q], 100)
        o = random.choice(qs)
        fb.relevance_model_feedback(o[1], seeds[o[0]][:3], cands[o[0]], 50)
        if fb.relevance_model_feedback(t, seeds[q][::-1], cands[q], 10) != ref[q]:
            diff += 1
    a2 = sum(fb.score_candidates(t, cands[q], 10) != fb.score_candidates(t, cands[q], 10) for q, t in qs)
    print(s, "B differs after other calls:", diff, "| A non-repeatable:", a2)
