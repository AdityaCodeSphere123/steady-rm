"""
One-parameter sweep, then a short coordinate pass, on Touche, Antique,
FiQA, SciFact, and Vaswani. A tick is kept when the five-set macro of
Track A nDCG, Track A MAP, and clean feedback nDCG rises, and no set
loses more than 0.001 nDCG on Track A or clean feedback.

Starts from the retention-1 point in submission/feedback.py.
Locked, not swept: prior_w, w_od, w_uw, lead_w, fb_lock, small_seed_n,
qlen_shrink. Those are what dropped retention on the hidden set.

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

# Trailing comma matters: ("scifact") is a string, not a one-item tuple.
SETS = ("touche2020", "antique", "fiqa", "scifact", "vaswani")
WEIGHT = {n: 1.0 for n in SETS}

# Retention-1 start from submission/feedback.py. lambda_ is the engine's lambda.
BASE = dict(
    stemmer=2, doc_stop=0, query_stop=2, qstop_min_len=0, qstop_list=0,
    smoothing=0, jm_lambda=0.50, mu=500,
    w_od=0, w_uw=0, uw_window=8,
    lead_w=0, lead_len=8, lead_mu=60,
    len_prior=0,
    prior_w=0, prior_k=4,
    fb_mu=250, fb_docs=10, fb_terms=15, idf_power=0.5,
    support_frac=0.20, fb_conf=0.30,
    fb_lock=1,
    lambda_=0.9,
)

# Tokenisation knobs are applied at prepare(), so each tick rebuilds the index.
PREP = {"stemmer", "doc_stop"}

# Track A, then Track B. Current value is added again at sweep time.
# Locked, not swept: prior_w, w_od, w_uw, lead_w, fb_lock.
GRID = [
    ("mu", [250, 500, 600, 750, 1000, 1500]),
    ("lambda_", [0.75, 0.8, 0.85, 0.9, 0.95]),
    ("fb_terms", [10, 15, 25, 40]),
    ("fb_mu", [100, 250, 500]),
    ("idf_power", [0.25, 0.5, 0.75]),
    ("support_frac", [0.1, 0.2, 0.3]),
    ("fb_conf", [0.2, 0.3, 0.4]),
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


def params_of(cur, key=None, value=None):
    p = {k: float(v) for k, v in cur.items() if k != "lambda_"}
    p["lambda"] = float(cur["lambda_"])
    if key == "lambda_":
        p["lambda"] = float(value)
    elif key is not None:
        p[key] = float(value)
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


def score_name(name, data):
    """Track A nDCG, Track A MAP, clean feedback nDCG."""
    qr, cd, pairs = data[name]
    ns = ms = bs = 0.0
    for q, t in pairs:
        pool = cd[q]
        ranked = [d for d, _ in fb.score_candidates(t, pool, 10)]
        ns += ndcg10(ranked, qr[q])
        ms += ap10(ranked, qr[q])
        seed = ranked[:10]
        fb_ids = [d for d, _ in fb.relevance_model_feedback(t, seed, pool, 10)]
        bs += ndcg10(fb_ids, qr[q])
    n = len(pairs)
    return ns / n, ms / n, bs / n


def sweep_knob(key, vals, cur, names, data):
    """Score every tick. Tokenisation knobs rebuild each corpus per tick."""
    out = {}
    prep = key in PREP
    for name in names:
        corpus = os.path.join(ROOT, "data", name, "corpus.jsonl")
        if not prep:
            fb.set_params(**params_of(cur))
            fb.prepare(corpus)
        for v in vals:
            fb.set_params(**params_of(cur, key, v))
            if prep:
                fb.prepare(corpus)
            out.setdefault(float(v), {})[name] = score_name(name, data)
    return out


def macro(tab):
    w = sum(WEIGHT[s] for s in tab)
    def avg(i):
        return sum(WEIGHT[s] * v[i] for s, v in tab.items()) / w
    return avg(0), avg(1), avg(2)


def both_vote(tab, ctab, names):
    """Macro can rise only if no set loses more than 0.001 nDCG on A or feedback."""
    for s in names:
        if tab[s][0] < ctab[s][0] - 0.001:
            return False
        if tab[s][2] < ctab[s][2] - 0.001:
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
        cn, cm, cb = macro(ctab)
        log(f"\n# {key}  base {cn:.4f}/{cm:.4f}/{cb:.4f}  ({time.perf_counter()-t1:.0f}s)")
        best = None
        for v in vals:
            n, m, b = macro(tabs[v])
            worst = min(tabs[v][s][0] - ctab[s][0] for s in names)
            gain = (n - cn) + (m - cm) + (b - cb)
            flag = ""
            if v != float(cur[key]) and both_vote(tabs[v], ctab, names) and gain >= 0.0003:
                flag = " VOTE"
                rec = (gain, v, n, m, b, worst)
                if best is None or rec[0] > best[0]:
                    best = rec
            log(f"  {key}={v:g}  macro {n:.4f}/{m:.4f}/{b:.4f}  d {n-cn:+.4f}/{m-cm:+.4f}/{b-cb:+.4f}  worstN {worst:+.4f}{flag}")
            phase1.append({"param": key, "value": v, "ndcg": n, "map": m, "fb": b,
                           "d_ndcg": n - cn, "d_map": m - cm, "d_fb": b - cb, "worst_ndcg": worst,
                           "vote": flag == " VOTE"})
        if best:
            log(f"  -> best tick {key}={best[1]:g}  d {best[2]-cn:+.4f}/{best[3]-cm:+.4f}/{best[4]-cb:+.4f}")

    # Phase 2: coordinate descent. A tick is kept when the five-set macro
    # rises and no collection loses more than 0.001 nDCG on A or feedback.
    log("\n# coordinate")
    cur = dict(BASE)
    accepted = []
    for rnd in (1, 2):
        moved = False
        for key, vals in GRID:
            vals = sorted(set(float(v) for v in vals) | {float(cur[key])})
            tabs = sweep_knob(key, vals, cur, names, data)
            ctab = tabs[float(cur[key])]
            cn, cm, cb = macro(ctab)
            best_v, best = float(cur[key]), (cn, cm, cb, ctab)
            for v, tab in tabs.items():
                n, m, b = macro(tab)
                if not both_vote(tab, ctab, names):
                    continue
                if (n - cn) + (m - cm) + (b - cb) < 0.0004:
                    continue
                if n + m + b > best[0] + best[1] + best[2] + 1e-12:
                    best_v, best = float(v), (n, m, b, tab)
            if best_v != float(cur[key]):
                log(f"r{rnd} {key} {float(cur[key]):g}->{best_v:g}  {cn:.4f}/{cm:.4f}/{cb:.4f} -> {best[0]:.4f}/{best[1]:.4f}/{best[2]:.4f}")
                cur[key] = best_v
                accepted.append({"round": rnd, "param": key, "value": best_v,
                                 "ndcg": best[0], "map": best[1], "fb": best[2]})
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
        fb.set_params(**params_of(pcur))
        for name in names:
            fb.prepare(os.path.join(ROOT, "data", name, "corpus.jsonl"))
            acc[name] = score_name(name, data)
        rows[label] = acc
        n, m, b = macro(acc)
        log(f"{label} macro {n:.4f}/{m:.4f}/{b:.4f}")
    for name in names:
        b, t = rows["base"][name], rows["tuned"][name]
        log(f"  {name:12} {b[0]:.4f}/{b[1]:.4f}/{b[2]:.4f} -> {t[0]:.4f}/{t[1]:.4f}/{t[2]:.4f}  {t[0]-b[0]:+.4f}/{t[1]-b[1]:+.4f}/{t[2]-b[2]:+.4f}")
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
