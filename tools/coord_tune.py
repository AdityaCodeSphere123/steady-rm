"""Coordinate ascent over Track A parameters on many collections at once.

One worker process per collection keeps its corpus prepared; the parent sends
parameter dicts and gets back (nDCG@10, MAP@10). A move is kept when

    obj = mean_s clip(dN_s) + 0.5 * mean_s clip(dMAP_s) - 0.5 * sum_s max(0, -dN_s - tol)

rises, where d* is against the start point and clip limits any one
collection to +-0.02 so a single large set cannot carry the average.

    python -m tools.coord_tune --start '{"ax_z":0,"smoothing":2}' --rounds 2
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SETS = ["scifact", "nfcorpus", "fiqa", "vaswani", "antique", "touche2020",
        "scidocs", "cacm", "cranfield", "arguana", "full"]

GRID = {
    "mu": [200, 300, 400, 550, 700, 900, 1200, 1600],
    "jm_lambda": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "lead_w": [0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "lead_len": [6, 8, 12, 16, 24, 32],
    "lead_mu": [20, 40, 60, 100, 200],
    "len_prior": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
    "prior_w": [0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.2],
    "prior_k": [2, 4, 7, 10, 20, 60],
    "w_od": [0, 0.02, 0.05, 0.1],
    "w_uw": [0, 0.02, 0.05, 0.1],
}


def worker(name, conn):
    from submission import feedback as fb
    from tools.track_a_sweep import evaluate, load
    data = load(name)
    fb.prepare(data[0])
    while True:
        msg = conn.recv()
        if msg is None:
            return
        fb.set_params(**msg)
        n, m, _ = evaluate(data)
        conn.send((n, m))


class Pool:
    def __init__(self, sets, base):
        self.sets = sets
        self.conns, self.procs = [], []
        for s in sets:
            a, b = mp.Pipe()
            p = mp.Process(target=worker, args=(s, b), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.base = base

    def run(self, params):
        p = dict(self.base)
        p.update(params)
        for c in self.conns:
            c.send(p)
        return {s: c.recv() for s, c in zip(self.sets, self.conns)}

    def close(self):
        for c in self.conns:
            c.send(None)


def objective(res, ref, tol, clip=0.02):
    dn = {s: res[s][0] - ref[s][0] for s in ref}
    dm = {s: res[s][1] - ref[s][1] for s in ref}
    k = len(ref)
    c = lambda x: max(-clip, min(clip, x))
    obj = sum(c(x) for x in dn.values()) / k + 0.5 * sum(c(x) for x in dm.values()) / k
    obj -= 0.5 * sum(max(0.0, -x - tol) for x in dn.values())
    return obj, dn, dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="{}", help="JSON params on top of feedback.PARAMS")
    ap.add_argument("--sets", nargs="+", default=SETS)
    ap.add_argument("--params", nargs="+", default=list(GRID))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--out", default="runs/sweeps/coord.txt")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    from submission import feedback as fb
    base = dict(fb.PARAMS)
    cur = json.loads(a.start)
    pool = Pool(a.sets, base)
    log = open(a.out, "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    t0 = time.time()
    ref = pool.run(cur)
    say("start " + json.dumps(cur))
    say("ref " + " ".join(f"{s} {ref[s][0]:.4f}/{ref[s][1]:.4f}" for s in a.sets))
    best_obj = 0.0
    for r in range(a.rounds):
        moved = False
        for name in a.params:
            for v in GRID[name]:
                if cur.get(name, base.get(name)) == v:
                    continue
                trial = dict(cur, **{name: v})
                res = pool.run(trial)
                obj, dn, dm = objective(res, ref, a.tol)
                if a.verbose:
                    say(f"   try {name}={v} obj {obj:+.5f} meanN {sum(dn.values())/len(dn):+.5f}")
                if obj > best_obj + 1e-5:
                    best_obj, cur, moved = obj, trial, True
                    say(f"r{r} {name}={v} obj {obj:+.5f} meanN {sum(dn.values())/len(dn):+.5f} "
                        f"meanMAP {sum(dm.values())/len(dm):+.5f} | "
                        + " ".join(f"{x:+.4f}" for x in dn.values()) + f"  ({time.time()-t0:.0f}s)")
        if not moved:
            break
    say("final " + json.dumps(cur))
    res = pool.run(cur)
    say("final " + " ".join(f"{s} {res[s][0]:.4f}/{res[s][1]:.4f}" for s in a.sets))
    pool.close()


if __name__ == "__main__":
    main()
