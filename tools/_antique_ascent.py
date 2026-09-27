"""One-off Antique Track-A ascent. Does not touch live PARAMS in feedback.py."""
import math
import os
import sys
import time

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
print(f"queries={len(pairs)}", flush=True)


def ndcg10(ranked, rel):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted((r for r in rel.values() if r > 0), reverse=True)[:10]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def ap10(ranked, rel):
    hits = 0
    s = 0.0
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


def champ():
    fb.set_params(**fb.PARAMS)
    fb.set_params(
        **{
            "lambda": 1.0,
            "ranker": 0,
            "ax_w": 0.0,
            "bm25_w": 0.0,
            "pool_lambda": 1.0,
            "smoothing": 0,
            "stemmer": 2,
            "doc_stop": 0,
            "query_stop": 1,
            "qstop_min_len": 4,
            "qstop_list": 0,
            "mu_factor": 3,
            "len_prior": 0.08,
            "w_od": 0.1,
            "w_uw": 0.15,
            "uw_window": 8,
            "lead_w": 0.14,
            "lead_len": 12,
            "lead_mu": 60,
            "prior_w": 0.35,
            "prior_k": 20,
            "longq_n": 0,
            "mu_mode": 0,
        }
    )


t0 = time.perf_counter()
champ()
fb.prepare(os.path.join(D, "corpus.jsonl"))
print(f"prepare={time.perf_counter()-t0:.1f}s stats={fb.stats()}", flush=True)
best_n, best_m = eval_ql()
print(f"BASE champ nDCG={best_n:.4f} MAP={best_m:.4f}", flush=True)
best = {k: fb.get_param(k) for k in (
    "mu_factor","len_prior","lead_w","lead_len","lead_mu","w_od","w_uw","uw_window",
    "prior_w","prior_k","qstop_min_len","qstop_list","query_stop","doc_stop","stemmer",
    "bm25_w","ax_w","ranker","smoothing","jm_lambda","mu_min","mu_max","longq_n","mu_mode",
    "bm25_k1","bm25_b","bm25_delta",
)}

AXES = [
    ("mu_factor", [1, 2, 3, 4, 6, 8, 12]),
    ("len_prior", [0, 0.04, 0.08, 0.12, 0.2, 0.3]),
    ("lead_w", [0, 0.08, 0.14, 0.22, 0.35, 0.5, 0.65]),
    ("lead_len", [6, 8, 12, 16, 20]),
    ("lead_mu", [8, 20, 40, 60, 100]),
    ("w_od", [0, 0.05, 0.1, 0.15, 0.25, 0.35]),
    ("w_uw", [0, 0.08, 0.15, 0.22, 0.3]),
    ("uw_window", [4, 8, 12]),
    ("prior_w", [0, 0.15, 0.25, 0.35, 0.5, 0.7]),
    ("prior_k", [5, 10, 20, 40]),
    ("qstop_min_len", [0, 3, 4, 5, 8]),
    ("qstop_list", [0, 1, 2, 3]),
    ("query_stop", [0, 1, 2]),
    ("longq_n", [0, 6, 16]),
    ("mu_min", [50, 150, 250]),
    ("bm25_w", [0, 0.08, 0.15, 0.4, 1.0]),
    ("ax_w", [0, 0.08, 0.2]),
    ("smoothing", [0, 1, 2]),
    ("jm_lambda", [0.2, 0.5, 0.8]),
    ("mu_mode", [0, 1]),
    ("ranker", [0, 1]),
]

INDEX = [
    ("doc_stop", [0, 1]),
    ("stemmer", [0, 1, 2]),
]


def apply_best():
    fb.set_params(**best)


def try_axis(name, values, reindex=False):
    global best_n, best_m, best
    kept = best[name]
    improved = False
    for v in values:
        if abs(v - kept) < 1e-12:
            continue
        apply_best()
        fb.set_params(**{name: float(v)})
        if reindex:
            fb.prepare(os.path.join(D, "corpus.jsonl"))
        n, m = eval_ql()
        mark = ""
        if n > best_n + 1e-5 or (abs(n - best_n) < 1e-5 and m > best_m + 1e-5):
            best_n, best_m = n, m
            best = {**best, name: float(v)}
            kept = v
            improved = True
            mark = " KEEP"
        print(f"  {name}={v:g} nDCG={n:.4f} MAP={m:.4f}{mark}", flush=True)
    apply_best()
    if reindex:
        fb.prepare(os.path.join(D, "corpus.jsonl"))
    return improved


for rnd in range(1, 4):
    print(f"\n=== runtime round {rnd} nDCG={best_n:.4f} MAP={best_m:.4f} ===", flush=True)
    moved = False
    for name, values in AXES:
        if try_axis(name, values):
            moved = True
    if not moved:
        print("no runtime move", flush=True)
        break

print("\n=== index knobs ===", flush=True)
for name, values in INDEX:
    try_axis(name, values, reindex=True)

print("\n=== polish runtime ===", flush=True)
for name, values in AXES:
    try_axis(name, values)

print("\nANTIQUE-MAX", {k: best[k] for k in best if abs(best[k] - fb.PARAMS.get(k, best[k])) > 1e-9 or k in (
    "mu_factor","len_prior","lead_w","w_od","w_uw","prior_w","qstop_min_len","doc_stop","stemmer","bm25_w","ranker"
)})
print(f"FINAL nDCG={best_n:.4f} MAP={best_m:.4f}")
print("BEST", best)
