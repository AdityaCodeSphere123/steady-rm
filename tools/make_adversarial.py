"""
tools/make_adversarial.py -- build submission/adversarial_set.json.

Bonus track (docs/GRADING.md): our doc lists are fed as the seed to every
other team's relevance_model_feedback() on the same dev candidate pool; the
attack score is the nDCG@10 drop vs. their clean run on that query, averaged
over queries and defenders. We cannot run their code, so the search attacks a
panel of simulated defenders spanning the likely range of the class, all
emulated by our own engine (see DEFENDERS).

Attack documents are non-relevant pool documents (qrels <= 0 or unjudged):
a seed drawn from the pool is the strongest attack, because the relevance
model then promotes exactly those documents into the top 10. Candidates are
the non-relevant pool documents ranked highest by the defenders' first stage
(high P(Q|D) gets a document past likelihood weighting, coverage filters
and query anchoring). Each list is grown greedily, best document first (so
defenders that truncate the seed keep the strongest part), to maximise the
mean drop over the panel; the --n queries with the largest drop are kept.

Other teams tokenise differently, and an attack tuned to one tokeniser
barely transfers to another, so every defender is run twice: on our
tokeniser (Porter, query stopwords removed) and on a raw one (no stemming,
no stopword removal), each in its own copy of the engine.

  python -m tools.make_adversarial --root data/full --n 5
  python -m tools.make_adversarial --root data/full --eval submission/adversarial_set.json
"""
import argparse, ctypes, glob, importlib.util, json, os, shutil, tempfile

from submission import feedback as fb
from tools.sweep import ndcg10, read_candidates, read_qrels, read_queries

_PLAIN = dict(w_od=0, w_uw=0, lead_w=0, fb_mu=0, beta=-1, support_frac=0, idf_power=0,
              fb_qterms=0, fb_cov=0, fb_anchor_exp=0, fb_mode=0, fb_sdm=0, smoothing=0)
_TYPICAL = dict(_PLAIN, mu=1000, fb_docs=10, fb_terms=50, **{"lambda": 0.5})

# name: (model, rm, overrides)
DEFENDERS = {
    "ours":        ("competition", 3, {}),
    "ours_rm1":    ("competition", 1, {}),
    "dir_tuned":   ("dirichlet", 3, {}),
    "jm_tuned":    ("jm", 3, {}),
    "typical":     ("dirichlet", 3, _TYPICAL),
    "typical_rm1": ("dirichlet", 1, _TYPICAL),
    "typical_rm2": ("dirichlet", 2, _TYPICAL),
    "uniform_pd":  ("dirichlet", 3, dict(_TYPICAL, beta=0)),
    "cautious":    ("dirichlet", 3, dict(_TYPICAL, mu=2000, fb_terms=20, **{"lambda": 0.7})),
    "mu2000_rm3":  ("dirichlet", 3, dict(_TYPICAL, mu=2000)),
}


def find(d, pats):
    for p in pats:
        hits = sorted(glob.glob(os.path.join(d, p)))
        if hits:
            return hits[0]
    return None


TOKENISERS = {"porter": {}, "raw": {"stemmer": 0, "query_stop": 0}}


def engine_copy(tok):
    """A second, independent instance of submission.feedback and its library."""
    spec = importlib.util.spec_from_file_location(f"feedback_{len(tok)}_{id(tok)}", fb.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = os.path.join(tempfile.mkdtemp(), "librm_copy.so")
    shutil.copy(mod._lib._name, path)
    mod._lib = ctypes.CDLL(path)
    mod._declare(mod._lib)
    for cfg in mod.MODELS.values():
        cfg.update(tok)
    return mod


def panel():
    return [(t, name) for t in TOKENISERS for name in DEFENDERS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="collection dir (corpus/queries/qrels/candidates)")
    ap.add_argument("--n", type=int, default=5, help="queries to keep")
    ap.add_argument("--max-docs", type=int, default=10)
    ap.add_argument("--pool-cands", type=int, default=40)
    ap.add_argument("--out", default="submission/adversarial_set.json")
    ap.add_argument("--eval", help="only score an existing adversarial set against the panel")
    a = ap.parse_args()
    corpus = find(a.root, ["corpus.jsonl"])
    queries = dict(read_queries(find(a.root, ["queries_dev.tsv", "queries*.tsv", "*.tsv"])))
    qrels = read_qrels(find(a.root, ["qrels_dev.txt", "qrels*.txt", "qrels*.json"]))
    cands = read_candidates(find(a.root, ["candidates_dev.jsonl", "candidates*.jsonl"]))
    engines = {}
    for t, tok in TOKENISERS.items():
        engines[t] = fb if not tok else engine_copy(tok)
        engines[t].use_model()
        engines[t].prepare(corpus)

    def use(key):
        model, rm, ov = DEFENDERS[key[1]]
        e = engines[key[0]]
        e.use_model(model, rm, **{**ov, **TOKENISERS[key[0]]})
        return e

    def pool_of(qid):
        return [p if isinstance(p, str) else p[0] for p in cands.get(qid, [])]

    def clean_scores(text, pool, rel):
        out = {}
        for key in panel():
            e = use(key)
            seed = [d for d, _ in e.score_candidates(text, pool, 10)]
            out[key] = ndcg10([d for d, _ in e.relevance_model_feedback(text, seed, pool, 10)], rel)
        return out

    def drops(text, pool, rel, clean, seed):
        out = {}
        for key in panel():
            e = use(key)
            got = ndcg10([d for d, _ in e.relevance_model_feedback(text, seed, pool, 10)], rel)
            out[key] = clean[key] - got
        return out

    def report(rows, label):
        avg = {k: sum(r[3][k] for r in rows) / len(rows) for k in panel()}
        print(f"\n{label}: {len(rows)} queries, mean drop {sum(avg.values()) / len(avg):.3f}")
        print(f"  {'':12}" + "".join(f"{t:>8}" for t in TOKENISERS))
        for name in DEFENDERS:
            print(f"  {name:12}" + "".join(f"{avg[(t, name)]:8.3f}" for t in TOKENISERS))

    if a.eval:
        rows = []
        for qid, seed in json.load(open(a.eval)).items():
            text, pool, rel = queries[qid], pool_of(qid), qrels[qid]
            dd = drops(text, pool, rel, clean_scores(text, pool, rel), seed)
            rows.append((sum(dd.values()) / len(dd), qid, seed, dd))
            print(f"{qid}: mean drop {rows[-1][0]:.3f}  ({len(seed)} docs)")
        report(rows, a.eval)
        fb.use_model()
        return

    results = []
    for qid, text in queries.items():
        pool, rel = pool_of(qid), qrels.get(qid)
        if not pool or not rel:
            continue
        clean = clean_scores(text, pool, rel)
        rank = {}
        for key in panel():
            e = use(key)
            for i, (d, _) in enumerate(e.score_candidates(text, pool, len(pool))):
                rank[d] = min(rank.get(d, i), i)
        bad = sorted((d for d in pool if rel.get(d, 0) <= 0), key=rank.get)[: a.pool_cands]
        if not bad:
            continue

        chosen, best, best_seed, best_d = [], -1e9, [], {}
        while len(chosen) < a.max_docs:
            step = None
            for d in bad:
                if d in chosen:
                    continue
                dd = drops(text, pool, rel, clean, chosen + [d])
                m = sum(dd.values()) / len(dd)
                if step is None or m > step[0]:
                    step = (m, d, dd)
            if step is None:
                break
            chosen.append(step[1])
            if step[0] > best:
                best, best_seed, best_d = step[0], list(chosen), step[2]
        results.append((best, qid, best_seed, best_d))
        print(f"{qid}: mean drop {best:.3f}  min {min(best_d.values()):.3f}  ({len(best_seed)} docs)", flush=True)

    results.sort(reverse=True)
    keep = results[: a.n]
    out = {qid: seed for _, qid, seed, _ in keep}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    report(keep, f"wrote {a.out}")
    fb.use_model()


if __name__ == "__main__":
    main()
