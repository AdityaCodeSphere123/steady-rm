"""
Fine one-parameter sweep, then a short coordinate pass, over every local
dataset. Starts from the hidden 0.5369 / 0.4570 parameters.

A full Cartesian product of these ticks is not runnable. This script tries
every tick of every parameter from that point, then walks the improving
ticks together one knob at a time.

Writes macro_tune_out.txt in the repo root after every line, and
macro_tune_result.json when it finishes.

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

# Hidden champ. lambda stays 1 so Track B equals Track A.
BASE = dict(
    stemmer=2, doc_stop=0, query_stop=2, qstop_min_len=0, qstop_list=0,
    smoothing=2, jm_lambda=0.5, mu=0, mu_factor=8, mu_min=250, mu_max=2500,
    mu_mode=1, mu_loo_scale=1.0,
    w_od=0.25, w_uw=0.15, uw_window=16,
    lead_w=0.22, lead_len=8, lead_mu=60,
    len_prior=0.3, longq_n=0,
    prior_w=0.12, prior_k=7,
    bm25_w=1.0, bm25_k1=1.2, bm25_b=0.4, bm25_delta=0.25,
    ranker=1, ax_b=0.5, ax_k=0.35, ax_c=1.0, qspell=0,
    lambda_=1.0,
)

# Small ticks around the champ. Integer-like knobs step by their natural unit.
GRID = [
    ("w_od", [0.15, 0.18, 0.21, 0.25, 0.28, 0.31, 0.35]),
    ("w_uw", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
    ("uw_window", [8, 12, 16, 20, 24]),
    ("lead_w", [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34]),
    ("lead_len", [4, 6, 8, 10, 12, 14]),
    ("lead_mu", [30, 40, 50, 60, 80, 100]),
    ("len_prior", [0.10, 0.18, 0.24, 0.30, 0.36, 0.42]),
    ("longq_n", [0, 4, 6, 8, 12]),
    ("prior_w", [0.04, 0.08, 0.12, 0.16, 0.20, 0.24]),
    ("prior_k", [4, 5, 7, 9, 12, 16]),
    ("bm25_w", [0.6, 0.8, 1.0, 1.2, 1.4]),
    ("bm25_k1", [0.8, 1.0, 1.2, 1.4, 1.6]),
    ("bm25_b", [0.25, 0.30, 0.35, 0.40, 0.50, 0.55]),
    ("bm25_delta", [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]),
    ("ax_b", [0.30, 0.40, 0.50, 0.60, 0.70]),
    ("ax_k", [0.20, 0.28, 0.35, 0.42, 0.50]),
    ("ax_c", [0.60, 0.80, 1.00, 1.20, 1.50]),
    ("jm_lambda", [0.20, 0.35, 0.50, 0.65, 0.80]),
    ("mu_factor", [2, 4, 6, 8, 10, 12]),
    ("query_stop", [1, 2]),
    ("qstop_min_len", [0, 3, 4]),
    ("qstop_list", [0, 1, 2]),
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
    for name in sorted(os.listdir(os.path.join(ROOT, "data"))):
        if name in ("toy", "quora", "quora_hold"):
            continue
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
    """Prepare each corpus once, score every tick of this knob."""
    out = {}
    trial = params_of(cur)
    for name in names:
        fb.prepare(os.path.join(ROOT, "data", name, "corpus.jsonl"))
        for v in vals:
            trial[key] = float(v)
            fb.set_params(**trial)
            out.setdefault(float(v), {})[name] = score_name(name, data, trial)
    return out


def macro(tab):
    n = len(tab)
    return (sum(v[0] for v in tab.values()) / n,
            sum(v[1] for v in tab.values()) / n)


def main():
    os.chdir(ROOT)
    open(OUT, "w").close()
    names, data = load_sets()
    cur = dict(BASE)
    log("datasets " + " ".join(f"{n}:{len(data[n][2])}" for n in names))
    log("base " + " ".join(f"{k}={cur[k]:g}" for k in (
        "w_od", "w_uw", "uw_window", "lead_w", "lead_len", "lead_mu",
        "len_prior", "prior_w", "prior_k", "bm25_w", "bm25_k1", "bm25_b",
        "bm25_delta", "ax_b", "ax_k", "ax_c", "jm_lambda", "mu_factor",
        "query_stop")))
    t0 = time.perf_counter()

    # Phase 1: every tick, holding every other knob at the 0.5369 point.
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
            if v != float(cur[key]) and n >= cn - 1e-9 and m >= cm - 1e-9 and (n - cn) + (m - cm) >= 0.0003 and worst >= -0.008:
                flag = " KEEP"
                rec = ((n - cn) + (m - cm), v, n, m, worst)
                if best is None or rec[0] > best[0]:
                    best = rec
            log(f"  {key}={v:g}  macro {n:.4f}/{m:.4f}  d {n-cn:+.4f}/{m-cm:+.4f}  worstN {worst:+.4f}{flag}")
            phase1.append({"param": key, "value": v, "ndcg": n, "map": m,
                           "d_ndcg": n - cn, "d_map": m - cm, "worst_ndcg": worst})
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
                if n < cn - 1e-9 or m < cm - 1e-9:
                    continue
                if (n - cn) + (m - cm) < 0.0004:
                    continue
                if any(tab[s][0] < ctab[s][0] - 0.008 for s in names):
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
    log("\n# final vs 0.5369 point")
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
        json.dump({"changed": changed, "accepted": accepted, "phase1_keeps": [
            r for r in phase1 if r["d_ndcg"] >= -1e-9 and r["d_map"] >= -1e-9 and r["d_ndcg"] + r["d_map"] >= 0.0003
        ]}, fh, indent=2)
    log(f"wrote {RESULT}")


if __name__ == "__main__":
    main()
