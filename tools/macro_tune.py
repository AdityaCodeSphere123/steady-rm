"""
One-parameter sweep, then a short coordinate pass, on SciFact only.

    source .venv/bin/activate
    python tools/macro_tune.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from submission import feedback as fb
from tools.sweep import read_queries, read_qrels, read_candidates, ndcg10

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "macro_tune_out.txt")
RESULT = os.path.join(ROOT, "macro_tune_result.json")

def ticks(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


# Trailing comma matters: ("scifact") is a string, not a one-item tuple.
SETS = ("scifact",)
WEIGHT = {"scifact": 1.0}

BASE = dict(
    stemmer=2, doc_stop=0, query_stop=2, qstop_min_len=0, qstop_list=0, qspell=0,
    smoothing=2, jm_lambda=0.5, mu=100, mu_factor=8, mu_min=250, mu_max=2500,
    mu_mode=1, mu_loo_scale=1.0, mu_loo_docs=5000,
    w_od=0.16, w_uw=0.5, uw_window=16,
    lead_w=0.22, lead_len=16, lead_mu=60,
    len_prior=0.3, longq_n=0,
    prior_w=0.0, prior_k=7,
    bm25_w=1.2, bm25_k1=1.2, bm25_b=0.4, bm25_delta=0.25, bm25_z=0.0,
    pool_lambda=1.0, pool_docs=8,
    ranker=1, ax_b=0.6, ax_k=0.35, ax_c=1.0, ax_w=0.0,
    lambda_=1.0,
)

# Tokenisation knobs are applied at prepare(), so each tick rebuilds the index.
PREP = {"stemmer", "doc_stop", "mu_mode", "mu_loo_docs"}

GRID = [
    ("smoothing", [0, 1, 2]),
    ("jm_lambda", ticks(0.0, 1.0, 0.05)),
    ("mu", [0, 25, 50, 100, 200, 400, 800, 1500, 2500, 4000, 8000]),
    ("w_od", ticks(0.0, 1.0, 0.05)),
    ("w_uw", ticks(0.0, 1.5, 0.05)),
    ("uw_window", [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96]),
    ("lead_w", ticks(0.0, 1.0, 0.05)),
    ("lead_len", [0, 4, 8, 12, 16, 24, 32, 48, 64, 96]),
    ("lead_mu", [5, 20, 40, 60, 100, 200, 400, 800, 1600]),
    ("len_prior", ticks(0.0, 1.2, 0.05)),
    ("bm25_w", ticks(0.0, 4.0, 0.2)),
    ("bm25_k1", ticks(0.2, 4.0, 0.2)),
    ("bm25_b", ticks(0.0, 1.5, 0.1)),
    ("bm25_delta", ticks(0.0, 2.0, 0.1)),
    ("ax_b", ticks(0.0, 2.0, 0.1)),
    ("ax_k", ticks(0.05, 1.5, 0.05)),
    ("ax_c", [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]),
]


def ap10(ranked, rel):
    hits = score = 0.0
    R = sum(1 for g in rel.values() if g > 0)
    if not R:
        return 0.0
    for i, d in enumerate(ranked[:10], 1):
        if rel.get(d, 0) > 0:
            hits += 1
            score += hits / i
    return score / R


def log(line):
    print(line, flush=True)
    with open(OUT, "a") as fh:
        fh.write(line + "\n")


def params_of(cur):
    p = {k: float(v) for k, v in cur.items() if k != "lambda_"}
    p["lambda"] = float(cur["lambda_"])
    return p


def load_sets():
    names, data = [], {}
    for name in SETS:
        D = os.path.join(ROOT, "data", name)
        need = [f"{D}/queries_dev.tsv", f"{D}/qrels_dev.txt",
                f"{D}/candidates_dev.jsonl", f"{D}/corpus.jsonl"]
        if not all(os.path.exists(x) for x in need):
            continue
        qr = read_qrels(need[1])
        cd = read_candidates(need[2])
        pairs = [(q, t) for q, t in read_queries(need[0]) if q in cd and q in qr]
        if len(pairs) < 5:
            continue
        data[name] = (qr, cd, pairs)
        names.append(name)
    return names, data


def score_name(name, data, p):
    qr, cd, pairs = data[name]
    ns = ms = 0.0
    for q, t in pairs:
        ranked = [d for d, _ in fb.score_candidates(t, cd[q], 10)]
        ns += ndcg10(ranked, qr[q])
        ms += ap10(ranked, qr[q])
    n = len(pairs)
    return ns / n, ms / n


def sweep_knob(key, vals, cur, names, data):
    """Score every tick. Tokenisation knobs rebuild each corpus per tick."""
    out = {}
    trial = params_of(cur)
    prep = key in PREP
    for name in names:
        corpus = os.path.join(ROOT, "data", name, "corpus.jsonl")
        if not prep:
            fb.set_params(**trial)
            fb.prepare(corpus)
        for v in vals:
            trial[key] = float(v)
            fb.set_params(**trial)
            if prep:
                fb.prepare(corpus)
            out.setdefault(float(v), {})[name] = score_name(name, data, trial)
    return out


def macro(tab):
    w = sum(WEIGHT[s] for s in tab)
    return (sum(WEIGHT[s] * v[0] for s, v in tab.items()) / w,
            sum(WEIGHT[s] * v[1] for s, v in tab.items()) / w)


def both_vote(tab, ctab, names):
    """Both datasets must keep nDCG and MAP. Equal weight is applied by macro()."""
    for s in names:
        if tab[s][0] < ctab[s][0] - 1e-9 or tab[s][1] < ctab[s][1] - 1e-9:
            return False
    return True


def main():
    os.chdir(ROOT)
    open(OUT, "w").close()
    names, data = load_sets()
    if not names:
        raise SystemExit("no datasets loaded")
    cur = dict(BASE)
    log("datasets " + " ".join(f"{n}:{len(data[n][2])} w={WEIGHT[n]:g}" for n in names))
    log("base " + " ".join(f"{k}={cur[k]:g}" for k, _ in GRID))
    t0 = time.perf_counter()

    # Phase 1: every tick, holding every other knob at the start point.
    phase1 = []
    base_tab = None
    for key, vals in GRID:
        vals = sorted(set(float(v) for v in vals) | {float(cur[key])})
        t1 = time.perf_counter()
        tabs = sweep_knob(key, vals, cur, names, data)
        if base_tab is None:
            base_tab = tabs[float(cur[key])]
        ctab = tabs[float(cur[key])]
        cn, cm = macro(ctab)
        log(f"\n# {key}  base {cn:.4f}/{cm:.4f}  ({time.perf_counter()-t1:.0f}s)")
        best = None
        for v in vals:
            n, m = macro(tabs[v])
            worst = min(tabs[v][s][0] - ctab[s][0] for s in names)
            flag = ""
            if v != float(cur[key]) and both_vote(tabs[v], ctab, names) and (n - cn) + (m - cm) >= 0.0003:
                flag = " VOTE"
                rec = ((n - cn) + (m - cm), v, n, m, worst)
                if best is None or rec[0] > best[0]:
                    best = rec
            log(f"  {key}={v:g}  macro {n:.4f}/{m:.4f}  d {n-cn:+.4f}/{m-cm:+.4f}  worstN {worst:+.4f}{flag}")
            phase1.append({"param": key, "value": v, "ndcg": n, "map": m,
                           "d_ndcg": n - cn, "d_map": m - cm, "worst_ndcg": worst,
                           "vote": flag == " VOTE"})
        if best:
            log(f"  -> best tick {key}={best[1]:g}  d {best[2]-cn:+.4f}/{best[3]-cm:+.4f}")

    # Phase 2: coordinate descent from the champ, same ticks, both metrics
    # must rise and no dataset may lose more than 0.008 nDCG.
    log("\n# coordinate")
    cur = dict(BASE)
    accepted = []
    for rnd in (1, 2):
        moved = False
        for key, vals in GRID:
            vals = sorted(set(float(v) for v in vals) | {float(cur[key])})
            tabs = sweep_knob(key, vals, cur, names, data)
            ctab = tabs[float(cur[key])]
            cn, cm = macro(ctab)
            best_v, best = float(cur[key]), (cn, cm, ctab)
            for v, tab in tabs.items():
                n, m = macro(tab)
                if not both_vote(tab, ctab, names):
                    continue
                if (n - cn) + (m - cm) < 0.0004:
                    continue
                if n + m > best[0] + best[1] + 1e-12:
                    best_v, best = float(v), (n, m, tab)
            if best_v != float(cur[key]):
                log(f"r{rnd} {key} {float(cur[key]):g}->{best_v:g}  {cn:.4f}/{cm:.4f} -> {best[0]:.4f}/{best[1]:.4f}")
                cur[key] = best_v
                accepted.append({"round": rnd, "param": key, "value": best_v,
                                 "ndcg": best[0], "map": best[1]})
                moved = True
            else:
                log(f"r{rnd} {key} stay {float(cur[key]):g}")
        if not moved:
            log(f"round {rnd} no move")
            break

    # Final table against the original champ.
    champ = dict(BASE)
    log("\n# final vs start point")
    rows = {}
    for label, pcur in (("base", champ), ("tuned", cur)):
        acc = {}
        trial = params_of(pcur)
        fb.set_params(**trial)
        for name in names:
            fb.prepare(os.path.join(ROOT, "data", name, "corpus.jsonl"))
            acc[name] = score_name(name, data, trial)
        rows[label] = acc
        n, m = macro(acc)
        log(f"{label} macro {n:.4f}/{m:.4f}")
    for name in names:
        b, t = rows["base"][name], rows["tuned"][name]
        log(f"  {name:12} {b[0]:.4f}/{b[1]:.4f} -> {t[0]:.4f}/{t[1]:.4f}  {t[0]-b[0]:+.4f}/{t[1]-b[1]:+.4f}")
    tuned_params = {k: cur[k] for k in cur if k != "lambda_"}
    tuned_params["lambda"] = cur["lambda_"]
    changed = {}
    for k, v in tuned_params.items():
        src = "lambda_" if k == "lambda" else k
        if src in BASE and abs(float(v) - float(BASE[src])) > 1e-9:
            changed[k] = v
    log("CHANGED " + " ".join(f"{k}={v:g}" for k, v in changed.items()))
    log(f"elapsed {time.perf_counter()-t0:.0f}s")
    with open(RESULT, "w") as fh:
        json.dump({"changed": changed, "accepted": accepted, "phase1_votes": [
            r for r in phase1 if r["vote"]
        ]}, fh, indent=2)
    log(f"wrote {RESULT}")


if __name__ == "__main__":
    main()
