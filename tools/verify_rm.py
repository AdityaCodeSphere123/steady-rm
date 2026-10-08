"""Hand-checkable demo: engine vs textbook formulas on a 5-document corpus.

    python -m tools.verify_rm                 # query "apple fruit", seed d2 d5 d1
    python -m tools.verify_rm --query "apple juice" --seed d2 d3 --lam 0.7 --mu 5

Prints, side by side with a direct Python transcription of each formula:
  1. Dirichlet query likelihood   log P(Q|D) = sum_q log (tf(q,D) + mu P(q|C)) / (|D| + mu)
  2. RM1  P(w|R) = sum_D P(w|D) P(D|Q),  P(D|Q) proportional to P(Q|D)  (uniform prior)
  3. RM2  P(w|R) proportional to P(w) prod_q sum_D P(q|D) P(D|w)
  4. Rankings: RM1 / RM2 score sum_w P(w|R) log P(w|D);
     RM3 score lambda log P(Q|D)/|Q| + (1-lambda) sum_w P_RM1(w|R) log P(w|D)
"""
import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from submission import feedback as fb
from tests.test_engine import DOCS, TEXTBOOK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="apple fruit")
    ap.add_argument("--seed", nargs="+", default=["d2", "d5", "d1"])
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--mu", type=float, default=10)
    a = ap.parse_args()
    q, seed, pool, mu = a.query.split(), [d for d in a.seed if d in DOCS], list(DOCS), a.mu

    tf = {d: Counter(t.split()) for d, t in DOCS.items()}
    dl = {d: sum(c.values()) for d, c in tf.items()}
    cf = Counter()
    for c in tf.values():
        cf.update(c)
    C = sum(cf.values())
    pwd = lambda w, d: (tf[d][w] + mu * cf[w] / C) / (dl[d] + mu)
    ql = {d: sum(math.log(pwd(t, d)) for t in q) for d in pool}

    pdq = {d: math.exp(ql[d]) for d in seed}
    z = sum(pdq.values())
    pdq = {d: v / z for d, v in pdq.items()}
    rm1 = Counter()
    for d in seed:
        for t, n in tf[d].items():
            rm1[t] += pdq[d] * n / dl[d]

    pd = 1.0 / len(seed)
    rm2 = {}
    for t in set().union(*[tf[d] for d in seed]):
        pw = sum(tf[d][t] / dl[d] * pd for d in seed)
        v = pw
        for qi in q:
            v *= sum(pwd(qi, d) * (tf[d][t] / dl[d]) * pd / pw for d in seed)
        rm2[t] = v
    z = sum(rm2.values())
    rm2 = {t: v / z for t, v in rm2.items()}

    path = os.path.join(tempfile.mkdtemp(), "corpus.jsonl")
    with open(path, "w") as f:
        for d, t in DOCS.items():
            f.write(json.dumps({"doc_id": d, "text": t}) + "\n")
    fb.set_params(**{**TEXTBOOK, "mu": mu, "lambda": a.lam})
    fb.prepare(path)

    def show(title, ref, got):
        print(f"\n{title}\n  {'':10}{'hand':>12}{'engine':>12}")
        worst = 0.0
        for key in sorted(ref, key=lambda x: -ref[x]):
            worst = max(worst, abs(ref[key] - got.get(key, float("nan"))))
            print(f"  {key:10}{ref[key]:12.6f}{got.get(key, float('nan')):12.6f}")
        print(f"  max |hand - engine| = {worst:.2e}  {'OK' if worst < 1e-6 else 'MISMATCH'}")

    print("corpus:")
    for d, t in DOCS.items():
        print(f"  {d}: {t}")
    print(f"query {q}, seed {seed}, Dirichlet mu {mu}, lambda {a.lam}")
    print("P(D|Q) over the seed: " + ", ".join(f"{d} {p:.4f}" for d, p in pdq.items()))

    show("1. log P(Q|D)", ql, dict(fb.score_candidates(a.query, pool, len(pool))))
    show("2. RM1 P(w|R)", dict(rm1), dict(fb.relevance_model(a.query, seed, 1)))
    show("3. RM2 P(w|R)", rm2, dict(fb.relevance_model(a.query, seed, 2)))
    for rm, m, name in ((1, rm1, "RM1"), (2, rm2, "RM2"), (3, rm1, "RM3")):
        ref = {d: sum(p * math.log(pwd(t, d)) for t, p in m.items()) for d in pool}
        if rm == 3:
            ref = {d: a.lam * ql[d] / len(q) + (1 - a.lam) * ref[d] for d in pool}
        fb.set_params(rm_model=rm)
        show(f"4. {name} ranking score", ref, dict(fb.relevance_model_feedback(a.query, seed, pool, len(pool))))
    fb.use_model()


if __name__ == "__main__":
    main()
