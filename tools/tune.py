"""
tools/tune.py -- parallel coordinate ascent over ALL parameters, on ALL
local collections at once, against a proxy of the leaderboard score.

  python -m tools.tune --root data --rounds 2 --workers 4 --log tune_log.tsv

Objective per collection (Section 7 weights; retention enters through the
absolute noisy nDCG, which rewards both quality and robustness):
    J_c = 0.30 * QL + 0.25 * clean_FB + 0.35 * mean(noisy_FB at 25%, 50%)
Overall J = weighted mean of J_c (--full-weight counts data/full extra,
since the dev corpus is the closest thing to the held-out collection).

A move is kept only if J improves by more than --eps (default 0.0003):
anything smaller is query-sampling noise and would just overfit.
Collections are evaluated in parallel worker processes (1 engine thread
each, so 4 workers = 4 cores). Prints the final PARAMS to paste into
submission/feedback.py.
"""
import argparse, json, multiprocessing as mp, os, sys, time

SPACE = {
    # Round 2 grid: centred on the first tuning run, extended where the kept
    # value sat on the edge of the old grid (lead_w, w_od, fb_mu, prior_k,
    # idf_power).
    # Track A
    "mu": [100, 250, 500, 1000],
    "prior_w": [0.25, 0.3, 0.35],
    "prior_k": [10, 20, 40],
    "len_prior": [0.15, 0.25, 0.35],
    "lead_w": [0.25, 0.35, 0.45],
    "longq_n": [0, 16, 24],
    "w_od": [0.05, 0.1, 0.15],
    "w_uw": [0.05, 0.1, 0.15],
    # Feedback
    "fb_terms": [15, 20, 25],
    "beta": [1, 1.5, 2],
    "fb_mu": [100, 200, 400, 800],
    "idf_power": [0.25, 0.5, 0.75, 1.0],
    "term_support": [1, 2],
    "lambda_min": [0.4, 0.45, 0.5],
    "lambda_max": [0.8, 0.85, 0.9],
    "coh_w": [0.5, 0.65, 0.8],
}

_fb = None
_cache = {}


def _init():
    global _fb
    os.environ["RM_THREADS"] = "1"
    from submission import feedback
    _fb = feedback


def _job(args):
    name, files, params = args
    from tools.sweep import evaluate, read_candidates, read_qrels, read_queries
    if name not in _cache:
        corpus, q, r, c = files
        ids = [l.split('"doc_id"', 1)[1].split('"')[1] for l in open(corpus) if '"doc_id"' in l]
        _cache.clear()   # keep one collection's data per worker
        _cache[name] = (read_queries(q), read_qrels(r), read_candidates(c), ids)
    queries, qrels, cands, ids = _cache[name]
    _fb.set_params(**_fb.PARAMS)
    _fb.set_params(**params)
    _fb.prepare(files[0])
    m = evaluate(queries, qrels, cands, ids, ("pool25", "pool50"))
    return name, m


def objective(m):
    return 0.30 * m["ql"] + 0.25 * m["clean"] + 0.35 * 0.5 * (m["pool25"] + m["pool50"])


def main():
    from tools.multi_sweep import collections
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--only", default="")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.0003)
    ap.add_argument("--full-weight", type=float, default=2.0)
    ap.add_argument("--params", default="", help="comma list to restrict the search")
    ap.add_argument("--log", default="tune_log.tsv")
    a = ap.parse_args()
    cols = collections(a.root, set(filter(None, a.only.split(","))))
    weights = {c[0]: (a.full_weight if c[0] == "full" else 1.0) for c in cols}
    space = {k: v for k, v in SPACE.items() if not a.params or k in a.params.split(",")}
    from submission import feedback as fb
    best = {k: fb.PARAMS.get(k, None) for k in space}
    best = {k: v for k, v in best.items() if v is not None}
    for k in space:  # params not in PARAMS start at the engine default
        if k not in best:
            best[k] = fb.get_param(k)
    # Big collections first so workers finish together.
    cols.sort(key=lambda c: -os.path.getsize(c[1]))
    log = open(a.log, "a")
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        def run(params):
            t = time.perf_counter()
            res = dict(pool.map(_job, [(c[0], c[1:], params) for c in cols], chunksize=1))
            J = sum(weights[n] * objective(m) for n, m in res.items()) / sum(weights.values())
            worst = min(m["clean"] - m["ql"] for m in res.values())
            mean = {k: sum(m[k] for m in res.values()) / len(res) for k in ("ql", "clean", "pool50")}
            log.write(json.dumps({"J": J, "params": params, "per": {n: {k: round(v, 4) for k, v in m.items()}
                                  for n, m in res.items()}}) + "\n"); log.flush()
            return J, mean, worst, time.perf_counter() - t
        J0, mean, worst, dt = run(best)
        print(f"start J={J0:.5f} ql={mean['ql']:.4f} clean={mean['clean']:.4f} n50={mean['pool50']:.4f} "
              f"worst_fb={worst:+.4f} ({dt:.0f}s/eval)", flush=True)
        for rnd in range(a.rounds):
            moved = False
            for k, values in space.items():
                for v in values:
                    if v == best[k]:
                        continue
                    trial = dict(best); trial[k] = v
                    J, mean, worst, dt = run(trial)
                    tag = ""
                    if J > J0 + a.eps:
                        best, J0, moved, tag = trial, J, True, "  <- keep"
                    print(f"r{rnd} {k}={v:<6g} J={J:.5f} ql={mean['ql']:.4f} clean={mean['clean']:.4f} "
                          f"n50={mean['pool50']:.4f} worst_fb={worst:+.4f}{tag}", flush=True)
            if not moved:
                break
    print("\nFinal J =", round(J0, 5))
    print("Paste into PARAMS in submission/feedback.py:")
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    sys.exit(main())
