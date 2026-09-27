"""Compare the live ranker with a ranking that sorts each candidate pool by qrel grade.

The qrel sort is an oracle for this labeled file only. It is not wired into
score_candidates(), so a submission zip is unchanged.

    python -m tools.qrel_oracle
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from submission import feedback as fb
from tools.sweep import ndcg10, read_candidates, read_qrels, read_queries


def ap10(ranked, rel):
    hits = score = 0.0
    relevant = sum(1 for grade in rel.values() if grade > 0)
    if relevant <= 0:
        return 0.0
    for i, doc_id in enumerate(ranked[:10], 1):
        if rel.get(doc_id, 0) > 0:
            hits += 1
            score += hits / i
    return score / relevant


def oracle_rank(pool, rel):
    """Highest judgment first. Ties keep the candidate-file order."""
    return sorted(pool, key=lambda doc_id: rel.get(doc_id, 0), reverse=True)


def main():
    data = os.path.join(ROOT, "data", "antique")
    corpus = os.path.join(data, "corpus.jsonl")
    queries = read_queries(os.path.join(data, "queries_dev.tsv"))
    qrels = read_qrels(os.path.join(data, "qrels_dev.txt"))
    cands = read_candidates(os.path.join(data, "candidates_dev.jsonl"))
    pairs = [(qid, text) for qid, text in queries if qid in cands and qid in qrels]
    if not pairs:
        raise SystemExit(f"no overlapping queries under {data}")

    t0 = time.perf_counter()
    fb.prepare(corpus)
    print(f"backend={fb.backend()} prepare={time.perf_counter() - t0:.1f}s", flush=True)

    model_n, model_m, oracle_n, oracle_m = [], [], [], []
    shown = False
    for qid, text in pairs:
        rel, pool = qrels[qid], cands[qid]
        ranked = [doc_id for doc_id, _ in fb.score_candidates(text, pool, 10)]
        ideal = oracle_rank(pool, rel)
        model_n.append(ndcg10(ranked, rel))
        model_m.append(ap10(ranked, rel))
        oracle_n.append(ndcg10(ideal, rel))
        oracle_m.append(ap10(ideal, rel))
        if not shown:
            print(f"example qid={qid} text={text[:70]!r}")
            print(f"  model top10 grades  {[rel.get(d, 0) for d in ranked]}")
            print(f"  oracle top10 grades {[rel.get(d, 0) for d in ideal[:10]]}")
            shown = True

    n = len(pairs)
    mn, mm = sum(model_n) / n, sum(model_m) / n
    on, om = sum(oracle_n) / n, sum(oracle_m) / n
    print(f"queries={n}")
    print(f"current model     nDCG@10={mn:.4f}  MAP@10={mm:.4f}")
    print(f"sort pool by qrel nDCG@10={on:.4f}  MAP@10={om:.4f}")
    print(f"delta             nDCG@10={on - mn:+.4f}  MAP@10={om - mm:+.4f}")


if __name__ == "__main__":
    main()
