"""
tools/multi_sweep.py -- tune across EVERY local collection at once.

Picks parameters that are good on average instead of good on one corpus
(the held-out grading collection is unseen, so this is what matters).

Layout: <root>/<name>/ with corpus.jsonl, a queries *.tsv, a qrels file
(qrels*.txt / *.json) and candidates*.jsonl -- the same files as data/toy.

  python -m tools.multi_sweep --root data --grid lambda=-1,0.5 --grid qlen_shrink=0,12
  python -m tools.multi_sweep --root data --only scifact,fiqa --set qpp_w=0.5

Per combo it prints one row per collection and a MEAN row with:
  ql, clean, n25, n50, retention (clamped at 1, as in leaderboard.py),
  fb_gain = clean - ql  (negative means feedback hurts on that collection).
"""
import argparse, glob, itertools, os, sys, time

from submission import feedback as fb
from tools.sweep import evaluate, read_candidates, read_qrels, read_queries


def find(d, pats):
    for p in pats:
        hits = sorted(glob.glob(os.path.join(d, p)))
        if hits:
            return hits[0]
    return None


def collections(root, only):
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d) or (only and name not in only):
            continue
        files = (find(d, ["corpus.jsonl"]), find(d, ["queries_dev.tsv", "queries*.tsv", "*.tsv"]),
                 find(d, ["qrels_dev.txt", "qrels*.txt", "qrels*.json"]),
                 find(d, ["candidates_dev.jsonl", "candidates*.jsonl"]))
        if all(files) and os.path.getsize(files[3]) > 0:
            out.append((name, *files))
        else:
            print(f"skip {name}: missing/empty files", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--only", default="")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--grid", action="append", default=[])
    ap.add_argument("--hard", action="store_true", help="also run 75%%/out-of-pool/hard noise")
    a = ap.parse_args()
    base = {k: float(v) for k, v in (s.split("=") for s in a.set)}
    grid = [(k, [float(x) for x in v.split(",")]) for k, v in (g.split("=") for g in a.grid)]
    cols = collections(a.root, set(filter(None, a.only.split(","))))
    loaded = {}
    for name, corpus, q, r, c in cols:
        loaded[name] = (corpus, read_queries(q), read_qrels(r), read_candidates(c),
                        [l.split('"doc_id"', 1)[1].split('"')[1] for l in open(corpus) if '"doc_id"' in l])
    conds = ("pool25", "pool50", "pool75", "out50", "hard50") if a.hard else ("pool25", "pool50")
    keys = [k for k, _ in grid]
    print("\t".join(keys + ["collection", "ql", "clean", "n25", "n50", "retention", "fb_gain"]))
    for combo in (list(itertools.product(*[v for _, v in grid])) or [()]):
        params = dict(base); params.update(zip(keys, combo))
        rows = []
        for name, (corpus, queries, qrels, cands, docs) in loaded.items():
            fb.set_params(**{k: v for k, v in fb.PARAMS.items()})
            fb.set_params(**params)
            fb.prepare(corpus)
            m = evaluate(queries, qrels, cands, docs, conds)
            rows.append((name, m))
            print("\t".join([f"{x:g}" for x in combo] + [name] + [f"{m[k]:.4f}" for k in
                  ("ql", "clean", "pool25", "pool50", "retention")] + [f"{m['clean'] - m['ql']:+.4f}"]), flush=True)
        mean = {k: sum(m[k] for _, m in rows) / len(rows) for k in ("ql", "clean", "pool25", "pool50", "retention")}
        worst = min(m["clean"] - m["ql"] for _, m in rows)
        print("\t".join([f"{x:g}" for x in combo] + ["MEAN"] + [f"{mean[k]:.4f}" for k in
              ("ql", "clean", "pool25", "pool50", "retention")] + [f"{mean['clean'] - mean['ql']:+.4f} (worst {worst:+.4f})"]))
        print(flush=True)


if __name__ == "__main__":
    main()
