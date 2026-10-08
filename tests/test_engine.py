"""
tests/test_engine.py -- hand-checkable correctness of the C++ engine.

A 5-document corpus; with every robustness knob switched off, the engine's
query likelihood, RM1, RM2 and RM3 ranking must equal a direct Python
transcription of the textbook formulas. Also checks the conformance rules.
Run: python -m tests.test_engine
"""
import json, math, os, tempfile
from collections import Counter

from submission import feedback as fb

DOCS = {
    "d1": "apple banana apple cherry",
    "d2": "apple fruit juice fresh apple",
    "d3": "banana split dessert",
    "d4": "car engine repair shop",
    "d5": "apple orchard harvest fruit",
}
TEXTBOOK = dict(stemmer=0, query_stop=0, smoothing=0, mu=10, w_od=0, w_uw=0, lead_w=0,
                fb_mu=0, fb_docs=100, fb_terms=1000, beta=-1, support_frac=0, idf_power=0,
                fb_qterms=0, fb_cov=0, fb_anchor_exp=0, fb_mode=0, fb_sdm=0, tr_w=0)
MU = 10.0

tf = {d: Counter(t.split()) for d, t in DOCS.items()}
dl = {d: sum(c.values()) for d, c in tf.items()}
cf = Counter(); [cf.update(c) for c in tf.values()]
C = sum(cf.values())
pc = lambda w: cf[w] / C
pwd_dir = lambda w, d: (tf[d][w] + MU * pc(w)) / (dl[d] + MU)


def close(a, b, eps=1e-6):
    assert abs(a - b) <= eps * max(1.0, abs(a), abs(b)), (a, b)


def rm1(q, seed, pwd=None):
    pwd = pwd or pwd_dir
    w = {d: math.exp(sum(math.log(pwd(t, d)) for t in q)) for d in seed}
    z = sum(w.values()); w = {d: v / z for d, v in w.items()}
    m = Counter()
    for d in seed:
        for t, n in tf[d].items():
            m[t] += w[d] * n / dl[d]
    z = sum(m.values())
    return {t: v / z for t, v in m.items()}


def rm2(q, seed):
    PD = 1.0 / len(seed)
    terms = set().union(*[tf[d] for d in seed])
    out = {}
    for t in terms:
        pw = sum(tf[d][t] / dl[d] * PD for d in seed)
        v = pw
        for qi in q:
            v *= sum(pwd_dir(qi, d) * (tf[d][t] / dl[d]) * PD / pw for d in seed)
        out[t] = v
    z = sum(out.values())
    return {t: v / z for t, v in out.items() if v > 0}


def main():
    path = os.path.join(tempfile.mkdtemp(), "corpus.jsonl")
    with open(path, "w") as f:
        for d, t in DOCS.items():
            f.write(json.dumps({"doc_id": d, "text": t}) + "\n")
    assert fb.backend() == "cpp", "C++ engine did not build"
    fb.set_params(**TEXTBOOK)
    fb.prepare(path)
    q, seed, pool = ["apple", "fruit"], ["d2", "d5", "d1"], list(DOCS)

    # 1. Query likelihood.
    ql = dict(fb.score_candidates("apple fruit", pool, 5))
    for d in pool:
        close(ql[d], sum(math.log(pwd_dir(t, d)) for t in q))

    # 2. RM1 and RM2 estimates.
    for variant, ref in ((1, rm1(q, seed)), (2, rm2(q, seed))):
        got = dict(fb.relevance_model("apple fruit", seed, variant))
        assert set(got) == set(ref), (variant, set(got) ^ set(ref))
        for t in ref:
            close(got[t], ref[t])

    # 3. RM3 ranking: lambda * mean log P(q|D) + (1-lambda) * sum_w P_R(w) log P(w|D)
    lam = 0.4
    fb.set_params(**{"lambda": lam})
    m = rm1(q, seed)
    got = dict(fb.relevance_model_feedback("apple fruit", seed, pool, 5))
    for d in pool:
        ref = lam * sum(math.log(pwd_dir(t, d)) for t in q) / len(q) + \
              (1 - lam) * sum(p * math.log(pwd_dir(t, d)) for t, p in m.items())
        close(got[d], ref)

    # 3b. Same with two-stage smoothing (smoothing=2): the absent-term part of
    # sum_w P_R(w) log P(w|D) must use the two-stage background, not Dirichlet's.
    L = 0.5
    fb.set_params(smoothing=2, jm_lambda=L)
    pwd_ts = lambda w, d: (1 - L) * pwd_dir(w, d) + L * pc(w)
    m = rm1(q, seed, pwd_ts)
    got = dict(fb.relevance_model_feedback("apple fruit", seed, pool, 5))
    for d in pool:
        ref = lam * sum(math.log(pwd_ts(t, d)) for t in q) / len(q) + \
              (1 - lam) * sum(p * math.log(pwd_ts(t, d)) for t, p in m.items())
        close(got[d], ref)
    fb.set_params(smoothing=0)

    # 4. Conformance: only pool ids, no duplicates, sorted, k respected.
    fb.set_params(**{k: v for k, v in fb.PARAMS.items() if k != "stemmer"})
    for fn in (lambda k: fb.score_candidates("apple fruit", ["d3", "d4", "d3"], k),
               lambda k: fb.relevance_model_feedback("apple fruit", ["d2", "zzz", "d4"], ["d3", "d4"], k)):
        for k in (0, 1, 2, 10):
            r = fn(k)
            ids = [d for d, _ in r]
            assert len(ids) == len(set(ids)) == min(k, 2) and set(ids) <= {"d3", "d4"}
            assert all(r[i][1] >= r[i + 1][1] for i in range(len(r) - 1))
    assert fb.score_candidates("apple", [], 10) == []
    assert fb.relevance_model_feedback("", ["d1"], ["d1", "d2"], 10)  # empty query -> pool order
    # No usable seed: no feedback, exactly the Track A ranking and scores.
    pool2 = ["d1", "d2"]
    assert fb.relevance_model_feedback("apple", [], pool2, 2) == fb.score_candidates("apple", pool2, 2)
    # The supplied seed is used as given: unknown ids leave no usable seed.
    assert fb.relevance_model_feedback("apple", ["zzz"], pool2, 2) == fb.score_candidates("apple", pool2, 2)
    # Feedback off (lambda = 1): exactly the Track A ranking and scores.
    fb.set_params(**{**{k: v for k, v in fb.PARAMS.items() if k != "stemmer"}, "lambda": 1.0})
    for q, seed in (("apple fruit", ["d1", "d3"]), ("apple juice", ["d2", "zzz", "d4"])):
        assert fb.relevance_model_feedback(q, seed, pool, 5) == fb.score_candidates(q, pool, 5)
    fb.set_params(**{k: v for k, v in fb.PARAMS.items() if k != "stemmer"})

    # 5. Call-order independence: identical output with or without a prior score_candidates().
    a = fb.relevance_model_feedback("apple juice", ["d3", "d4"], pool, 5)
    fb.score_candidates("apple juice", pool, 5)
    assert fb.relevance_model_feedback("apple juice", ["d3", "d4"], pool, 5) == a
    print("all engine tests passed")


if __name__ == "__main__":
    main()
