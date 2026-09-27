"""Fine grid around Antique-max. Writes nothing to PARAMS."""
import math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from submission import feedback as fb
from tools.sweep import read_candidates, read_qrels, read_queries

D = os.path.join(ROOT, "data/antique")
queries = read_queries(os.path.join(D, "queries_dev.tsv"))
qrels = read_qrels(os.path.join(D, "qrels_dev.txt"))
cands = read_candidates(os.path.join(D, "candidates_dev.jsonl"))
pairs = [(qid, t) for qid, t in queries if qid in cands and qid in qrels]


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


def eval_ql():
    nd, mp = [], []
    for qid, text in pairs:
        rel = qrels[qid]
        ranked = [d for d, _ in fb.score_candidates(text, cands[qid], 10)]
        nd.append(ndcg10(ranked, rel))
        mp.append(ap10(ranked, rel))
    return sum(nd) / len(nd), sum(mp) / len(mp)

PEAK = dict(
    lambda_=1.0, ranker=1, ax_w=0, bm25_w=1.0, pool_lambda=1.0,
    smoothing=2, jm_lambda=0.5, stemmer=2, doc_stop=0,
    query_stop=2, qstop_min_len=0, qstop_list=0,
    mu_factor=8, mu_min=250, mu_max=2500, mu_mode=1, len_prior=0.3,
    w_od=0.1, w_uw=0.15, uw_window=12,
    lead_w=0.22, lead_len=8, lead_mu=60,
    prior_w=0.15, prior_k=5, longq_n=0,
    bm25_k1=1.2, bm25_b=0.4, bm25_delta=0.5,
    ax_b=0.5, ax_k=0.35, ax_c=1.0,
)

def apply(p):
    kw = {k: float(v) for k, v in p.items()}
    if "lambda_" in kw:
        kw["lambda"] = kw.pop("lambda_")
    fb.set_params(**fb.PARAMS)
    fb.set_params(**kw)

apply(PEAK)
fb.prepare(os.path.join(D, "corpus.jsonl"))
best_n, best_m = eval_ql()
best = dict(PEAK)
print(f"peak start nDCG={best_n:.4f} MAP={best_m:.4f}", flush=True)

AXES = [
    ("bm25_k1", [0.7, 0.9, 1.2, 1.6, 2.2, 2.4]),
    ("bm25_b", [0.2, 0.3, 0.4, 0.6, 0.75]),
    ("bm25_delta", [0.0, 0.3, 0.5, 0.8, 1.0]),
    ("ax_b", [0.3, 0.5, 0.8, 1.0]),
    ("ax_k", [0.2, 0.35, 0.5, 0.7]),
    ("ax_c", [0.5, 1.0, 1.5]),
    ("prior_w", [0.08, 0.12, 0.15, 0.18, 0.22]),
    ("prior_k", [3, 5, 7, 10]),
    ("lead_w", [0.18, 0.22, 0.26, 0.30]),
    ("w_od", [0.05, 0.10, 0.15]),
    ("w_uw", [0.10, 0.15, 0.20]),
    ("len_prior", [0.15, 0.25, 0.3, 0.4]),
    ("bm25_w", [0.7, 1.0, 1.4, 2.0]),
]

def try_axis(name, values):
    global best_n, best_m, best
    for v in values:
        p = dict(best)
        p[name] = v
        apply(p)
        n, m = eval_ql()
        mark = ""
        if n > best_n + 1e-5 or (abs(n - best_n) < 1e-5 and m > best_m + 1e-5):
            best_n, best_m, best = n, m, p
            mark = " KEEP"
        print(f"  {name}={v:g} nDCG={n:.4f} MAP={m:.4f}{mark}", flush=True)

for rnd in range(2):
    print(f"\n=== fine {rnd+1} nDCG={best_n:.4f} ===", flush=True)
    for name, values in AXES:
        try_axis(name, values)

print("FINE-BEST", best)
print(f"FINAL nDCG={best_n:.4f} MAP={best_m:.4f}")
