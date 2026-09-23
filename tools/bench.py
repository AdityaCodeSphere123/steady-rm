"""
tools/bench.py -- timing of prepare() and of each call type.

  RM_TIMING=1 python -m tools.bench --corpus data/full/corpus.jsonl \
      --queries data/full/queries_dev.tsv --candidates data/full/candidates_dev.jsonl
RM_TIMING=1 adds the per-phase prepare() breakdown; RM_THREADS=1 gives the
single-thread baseline for the parallel speed-up.
"""
import argparse, statistics, time
from submission import feedback as fb
from tools.sweep import read_candidates, read_queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--queries", required=True)
    ap.add_argument("--candidates", required=True); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    t = time.perf_counter(); fb.prepare(a.corpus); tp = time.perf_counter() - t
    print(f"backend={fb.backend()} threads={fb.stats().get('threads')} prepare={tp:.2f}s")
    qs, cands = read_queries(a.queries), read_candidates(a.candidates)
    first, warm, fbt = [], [], []
    for qid, text in qs[: a.n]:
        pool = cands.get(qid)
        if not pool:
            continue
        t = time.perf_counter(); r = fb.score_candidates(text, pool, 10); first.append(time.perf_counter() - t)
        t = time.perf_counter(); r = fb.score_candidates(text, pool, 10); warm.append(time.perf_counter() - t)
        t = time.perf_counter(); fb.relevance_model_feedback(text, [d for d, _ in r], pool, 10); fbt.append(time.perf_counter() - t)
    ms = lambda v: f"median {statistics.median(v) * 1e3:.2f} ms, max {max(v) * 1e3:.2f} ms"
    print("score_candidates, new query :", ms(first))
    print("score_candidates, repeat    :", ms(warm))
    print("relevance_model_feedback    :", ms(fbt))


if __name__ == "__main__":
    main()
