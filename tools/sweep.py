"""
tools/sweep.py -- evaluate / tune submission.feedback outside the harness.

Mirrors harness/run_harness.py (seed = score_candidates top-10, practice
noise from harness/noise_injection.py) and adds harsher noise conditions
the grading recipe might use:
  pool25/pool50   practice noise (random swaps from the candidate pool)
  pool75          heavier practice-style noise
  out50           50% swapped for random documents OUTSIDE the pool
  hard50          50% swapped for the highest-QL NON-relevant pool docs
                  outside the seed (uses qrels: a near-worst-case attack)

Usage (from the repo root):
  python -m tools.sweep --corpus data/full/corpus.jsonl --queries Q --qrels R --candidates C \
      [--set lambda=0.5 --set rm2_mix=0.3] [--grid lambda=0.3,0.5,0.7] [--folds 2]

Queries: TSV "qid<TAB>text"; qrels: TREC "qid 0 docid rel" or JSON
{qid:{docid:rel}}; candidates: JSONL {"qid":..,"candidates":[[docid,score],..]}
(or the harness's own readers are used when importable).
"""
import argparse, itertools, json, math, random, sys, time, zlib
from typing import Dict, List

from submission import feedback as fb

try:
    from harness import noise_injection
except ImportError:  # pragma: no cover
    noise_injection = None


def read_queries(p):
    out = []
    for line in open(p, encoding="utf-8"):
        if line.strip():
            q, t = line.rstrip("\n").split("\t", 1)
            out.append((q, t))
    return out


def read_qrels(p):
    if p.endswith(".json"):
        return {q: {d: int(r) for d, r in v.items()} for q, v in json.load(open(p)).items()}
    out: Dict[str, Dict[str, int]] = {}
    for line in open(p):
        s = line.split()
        if len(s) >= 4:
            out.setdefault(s[0], {})[s[2]] = int(s[3])
    return out


def read_candidates(p):
    out = {}
    for line in open(p):
        if line.strip():
            o = json.loads(line)
            out[str(o["qid"])] = [c[0] if isinstance(c, (list, tuple)) else c for c in o["candidates"]]
    return out


def ndcg10(ranked: List[str], rel: Dict[str, int]) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted((r for r in rel.values() if r > 0), reverse=True)[:10]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(queries, qrels, cands, all_docs, conds=("pool25", "pool50", "pool75", "out50", "hard50")):
    res = {k: [] for k in ("ql", "clean", *conds)}
    for qid, text in queries:
        if qid not in cands or qid not in qrels:
            continue
        pool, rel = cands[qid], qrels[qid]
        ql = fb.score_candidates(text, pool, 10)
        res["ql"].append(ndcg10([d for d, _ in ql], rel))
        seed = [d for d, _ in ql][:10]
        run = lambda s: ndcg10([d for d, _ in fb.relevance_model_feedback(text, s, pool, 10)], rel)
        res["clean"].append(run(seed))
        rng = random.Random(zlib.crc32(qid.encode()))
        for c in conds:
            frac = int(c[-2:]) / 100
            s = list(seed)
            n = int(round(frac * len(s)))
            pos = rng.sample(range(len(s)), n)
            if c.startswith("pool"):
                if noise_injection and frac in (0.25, 0.5):
                    lv = dict(noise_injection.sweep(seed, pool, seed=noise_injection.stable_seed(qid, 0)))
                    s = lv.get(frac, s)
                else:
                    rest = [d for d in pool if d not in seed]
                    for p, d in zip(pos, rng.sample(rest, min(n, len(rest)))):
                        s[p] = d
            elif c.startswith("out"):
                pset = set(pool)
                outside = [d for d in all_docs if d not in pset]
                for p in pos:
                    if outside:  # full-recall pools (toy set) have no outside docs
                        s[p] = rng.choice(outside)
            else:  # hard: best-ranked non-relevant pool docs outside the seed
                full = [d for d, _ in fb.score_candidates(text, pool, len(pool))]
                bad = [d for d in full if d not in seed and rel.get(d, 0) <= 0][:n]
                for p, d in zip(pos, bad):
                    s[p] = d
            res[c].append(run(s))
    m = {k: (sum(v) / len(v) if v else 0.0) for k, v in res.items()}
    noisy = [m[c] for c in ("pool25", "pool50")]
    m["retention"] = min(1.0, (sum(noisy) / len(noisy)) / m["clean"]) if m["clean"] > 0 else 0.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True); ap.add_argument("--candidates", required=True)
    ap.add_argument("--set", action="append", default=[]); ap.add_argument("--grid", action="append", default=[])
    ap.add_argument("--folds", type=int, default=1)
    a = ap.parse_args()
    t0 = time.perf_counter(); fb.prepare(a.corpus); tp = time.perf_counter() - t0
    print(f"backend={fb.backend()} prepare={tp:.2f}s stats={fb.stats()}", file=sys.stderr)
    base = {k: float(v) for k, v in (s.split("=") for s in a.set)}
    fb.set_params(**base)
    queries, qrels, cands = read_queries(a.queries), read_qrels(a.qrels), read_candidates(a.candidates)
    all_docs = [json.loads(l)["doc_id"] for l in open(a.corpus) if l.strip()]
    grid = [(k, [float(x) for x in v.split(",")]) for k, v in (g.split("=") for g in a.grid)]
    combos = list(itertools.product(*[v for _, v in grid])) or [()]
    folds = [queries[i::a.folds] for i in range(a.folds)]
    cols = ["ql", "clean", "pool25", "pool50", "pool75", "out50", "hard50", "retention"]
    print("\t".join([k for k, _ in grid] + ["fold"] + cols))
    for combo in combos:
        fb.set_params(**dict(zip([k for k, _ in grid], combo)))
        fb.prepare(a.corpus)  # no-op unless a tokenisation parameter changed
        for fi, fq in enumerate(folds):
            m = evaluate(fq, qrels, cands, all_docs)
            print("\t".join([f"{x:g}" for x in combo] + [str(fi)] + [f"{m[c]:.4f}" for c in cols]), flush=True)
        fb.set_params(**base)
        fb.prepare(a.corpus)


if __name__ == "__main__":
    main()
