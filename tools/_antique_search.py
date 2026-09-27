"""Coordinate search of every Track-A knob on Antique. Does not edit PARAMS."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from submission import feedback as fb
from tools.qrel_oracle import ap10
from tools.sweep import ndcg10, read_candidates, read_qrels, read_queries

D = os.path.join(ROOT, "data", "antique")
queries = read_queries(os.path.join(D, "queries_dev.tsv"))
qrels = read_qrels(os.path.join(D, "qrels_dev.txt"))
cands = read_candidates(os.path.join(D, "candidates_dev.jsonl"))
pairs = [(qid, text) for qid, text in queries if qid in cands and qid in qrels]
CORPUS = os.path.join(D, "corpus.jsonl")

# Scoring knobs. Relevance-model knobs are omitted: lambda stays 1, so feedback
# returns score_candidates unchanged.
AXES = [
    ("smoothing", [0, 1, 2]),
    ("mu", [0, 250, 800, 2000]),
    ("mu_factor", [0.5, 1, 2, 4, 6, 8, 12, 16, 24]),
    ("mu_min", [50, 100, 150, 250, 500, 1000]),
    ("mu_max", [800, 1500, 2500, 4000]),
    ("mu_mode", [0, 1]),
    ("mu_loo_scale", [0.5, 1.0, 1.5, 2.0]),
    ("jm_lambda", [0.1, 0.2, 0.4, 0.5, 0.7, 0.9]),
    ("w_od", [0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.25, 0.35]),
    ("w_uw", [0, 0.05, 0.1, 0.15, 0.2, 0.3]),
    ("uw_window", [4, 8, 12, 16, 24]),
    ("prior_w", [0, 0.05, 0.08, 0.12, 0.15, 0.2, 0.3, 0.45, 0.6]),
    ("prior_k", [3, 5, 7, 10, 15, 20, 40]),
    ("lead_w", [0, 0.08, 0.14, 0.18, 0.22, 0.28, 0.35, 0.5]),
    ("lead_len", [4, 6, 8, 10, 12, 16, 24]),
    ("lead_mu", [8, 20, 40, 60, 100, 200]),
    ("len_prior", [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]),
    ("longq_n", [0, 4, 6, 8, 12, 16]),
    ("bm25_w", [0, 0.3, 0.6, 1.0, 1.4, 2.0, 3.0]),
    ("bm25_k1", [0.4, 0.6, 0.9, 1.2, 1.6, 2.0, 2.4]),
    ("bm25_b", [0, 0.15, 0.25, 0.4, 0.55, 0.75, 1.0]),
    ("bm25_delta", [0, 0.25, 0.5, 0.75, 1.0, 1.5]),
    ("pool_lambda", [0.7, 0.85, 1.0]),
    ("pool_docs", [4, 8, 16, 32]),
    ("ranker", [0, 1]),
    ("ax_b", [0.2, 0.35, 0.5, 0.75, 1.0]),
    ("ax_k", [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]),
    ("ax_c", [0.3, 0.5, 1.0, 1.5, 2.0]),
    ("ax_w", [0, 0.05, 0.1, 0.2, 0.4]),
    ("qspell", [0, 1, 2]),
    ("query_stop", [0, 1, 2]),
    ("qstop_min_len", [0, 2, 3, 4, 5, 8]),
    ("qstop_list", [0, 1, 2, 3]),
]
INDEX = [
    ("doc_stop", [0, 1]),
    ("stemmer", [0, 1, 2]),
]


def eval_ql():
    nd, mp = [], []
    for qid, text in pairs:
        rel = qrels[qid]
        ranked = [doc_id for doc_id, _ in fb.score_candidates(text, cands[qid], 10)]
        nd.append(ndcg10(ranked, rel))
        mp.append(ap10(ranked, rel))
    return sum(nd) / len(nd), sum(mp) / len(mp)


def apply(params):
    fb.set_params(**fb.PARAMS)
    fb.set_params(**params)


best = {k: float(v) for k, v in fb.PARAMS.items()}
apply(best)
t0 = time.perf_counter()
fb.prepare(CORPUS)
print(f"prepare={time.perf_counter() - t0:.1f}s queries={len(pairs)}", flush=True)
t1 = time.perf_counter()
best_n, best_m = eval_ql()
print(f"START nDCG={best_n:.4f} MAP={best_m:.4f} eval={time.perf_counter() - t1:.2f}s", flush=True)


def try_axis(name, values, reindex=False):
    global best_n, best_m, best
    improved = False
    for raw in values:
        v = float(raw)
        if abs(v - best.get(name, v + 1)) < 1e-12:
            continue
        trial = dict(best)
        trial[name] = v
        apply(trial)
        if reindex:
            fb.prepare(CORPUS)
        n, m = eval_ql()
        mark = ""
        if n > best_n + 1e-5 or (abs(n - best_n) < 1e-5 and m > best_m + 1e-5):
            best_n, best_m, best = n, m, trial
            improved = True
            mark = " KEEP"
        print(f"  {name}={v:g} nDCG={n:.4f} MAP={m:.4f}{mark}", flush=True)
    apply(best)
    if reindex:
        fb.prepare(CORPUS)
    return improved


for rnd in range(1, 4):
    print(f"\n=== round {rnd} nDCG={best_n:.4f} MAP={best_m:.4f} ===", flush=True)
    if not any(try_axis(name, values) for name, values in AXES):
        print("no runtime move", flush=True)
        break

print("\n=== index knobs ===", flush=True)
for name, values in INDEX:
    try_axis(name, values, reindex=True)

print("\n=== polish ===", flush=True)
for name, values in AXES:
    try_axis(name, values)

changed = {k: best[k] for k, v in fb.PARAMS.items() if abs(best.get(k, v) - float(v)) > 1e-9}
print("CHANGED", changed)
print(f"FINAL nDCG={best_n:.4f} MAP={best_m:.4f}")
