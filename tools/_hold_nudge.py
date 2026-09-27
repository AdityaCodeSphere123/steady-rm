"""Holdout one-knobs from 0.5344 champ on Antique + SciDocs. lambda=1."""
import math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from submission import feedback as fb
from tools.sweep import read_candidates, read_qrels, read_queries

BASE = dict(
    stemmer=2, doc_stop=0, query_stop=2, qstop_min_len=0, qstop_list=0,
    smoothing=2, jm_lambda=0.5, mu=0, mu_factor=8, mu_min=250, mu_max=2500, mu_mode=1,
    w_od=0.05, w_uw=0.15, uw_window=12,
    lead_w=0.22, lead_len=8, lead_mu=60, len_prior=0.3, longq_n=0,
    prior_w=0.12, prior_k=7,
    bm25_w=1.0, bm25_k1=1.2, bm25_b=0.4, bm25_delta=0.5,
    pool_lambda=1.0, ranker=1, ax_b=0.5, ax_k=0.35, ax_c=1.0, ax_w=0.0,
    lambda_=1.0,
)

def ndcg10(ranked, rel):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted((r for r in rel.values() if r > 0), reverse=True)[:10]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def ap10(ranked, rel):
    hits = s = 0.0
    R = sum(1 for v in rel.values() if v > 0)
    if R <= 0:
        return 0.0
    for i, d in enumerate(ranked[:10], 1):
        if rel.get(d, 0) > 0:
            hits += 1
            s += hits / i
    return s / R

def load(name):
    d = os.path.join(ROOT, "data", name)
    q = read_queries(os.path.join(d, "queries_dev.tsv"))
    r = read_qrels(os.path.join(d, "qrels_dev.txt"))
    c = read_candidates(os.path.join(d, "candidates_dev.jsonl"))
    pairs = [(qid, t) for qid, t in q if qid in c and qid in r]
    return d, pairs, r, c

def apply(p):
    kw = {k: float(v) for k, v in p.items()}
    kw["lambda"] = kw.pop("lambda_")
    fb.set_params(**fb.PARAMS)
    fb.set_params(**kw)

def eval_pairs(pairs, qrels, cands):
    nd, mp = [], []
    for qid, text in pairs:
        ranked = [d for d, _ in fb.score_candidates(text, cands[qid], 10)]
        nd.append(ndcg10(ranked, qrels[qid]))
        mp.append(ap10(ranked, qrels[qid]))
    n = len(nd) or 1
    return sum(nd) / n, sum(mp) / n

VARIANTS = [
    ("base", {}),
    ("lead_len=10", {"lead_len": 10}),
    ("lead_len=12", {"lead_len": 12}),
    ("lead_w=0.18", {"lead_w": 0.18}),
    ("lead_w=0.20", {"lead_w": 0.20}),
    ("prior_w=0.08", {"prior_w": 0.08}),
    ("prior_w=0.18", {"prior_w": 0.18}),
    ("prior_k=10", {"prior_k": 10}),
    ("w_od=0.08", {"w_od": 0.08}),
    ("w_od=0.10", {"w_od": 0.10}),
    ("bm25_b=0.50", {"bm25_b": 0.50}),
    ("bm25_k1=1.4", {"bm25_k1": 1.4}),
    ("bm25_w=1.2", {"bm25_w": 1.2}),
    ("len_prior=0.22", {"len_prior": 0.22}),
    ("uw_window=10", {"uw_window": 10}),
]

def main():
    ant, apairs, ar, ac = load("antique")
    tune, hold = apairs[:100], apairs[100:]
    sci, spairs, sr, sc = load("scidocs")
    # 200-query subsample of scidocs for speed (even ids)
    ssub = spairs[::5][:200]

    apply(BASE)
    fb.prepare(os.path.join(ant, "corpus.jsonl"))
    print("name\tant_all\tant_tune\tant_hold\tant_map", flush=True)
    rows = []
    for name, delta in VARIANTS:
        p = dict(BASE); p.update(delta)
        apply(p)
        n_all, m_all = eval_pairs(apairs, ar, ac)
        n_t, _ = eval_pairs(tune, ar, ac)
        n_h, _ = eval_pairs(hold, ar, ac)
        rows.append((name, n_all, n_t, n_h, m_all))
        print(f"{name}\t{n_all:.4f}\t{n_t:.4f}\t{n_h:.4f}\t{m_all:.4f}", flush=True)

    apply(BASE)
    fb.prepare(os.path.join(sci, "corpus.jsonl"))
    print("\nscidocs subsample", flush=True)
    print("name\tsci_n\tsci_m", flush=True)
    for name, delta in VARIANTS:
        p = dict(BASE); p.update(delta)
        apply(p)
        n, m = eval_pairs(ssub, sr, sc)
        print(f"{name}\t{n:.4f}\t{m:.4f}", flush=True)

if __name__ == "__main__":
    main()
