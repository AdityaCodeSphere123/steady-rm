"""Competition entrypoint. C++ engine in submission/cpp, compiled on import.
RM_THREADS caps threads at 4.

    prepare(corpus_path)
    score_candidates(query, candidate_doc_ids, k=10)
    relevance_model_feedback(query, pseudo_relevant_doc_ids, candidate_doc_ids, k=10)

All retrieval work is done in C++ (submission/cpp/, loaded with ctypes).
The shared library is compiled from source on first import, on the machine
running the harness, into submission/_build/ (or the temp dir if that is
read-only). Its file name carries a hash of sources + flags, so edited
sources are always rebuilt and a binary built elsewhere is never loaded
(a -march=native binary from another CPU could crash the process). Building
or loading failure raises.

Model (maths in cpp/rm_engine.cpp, every knob in cpp/rm_engine.h):
  Track A  Query likelihood with the SPUD (Polya urn) document model
           (spud_mu=800, df background), mixed with a lead-field LM over
           the first 20 tokens (weight 0.1, Dirichlet 60), plus sequential
           dependence (ordered 0.05, unordered window 12 0.1, Dirichlet 550).
  Track B  RM3 on that same model: 0.95 * S_A(D) + 0.05 * E(D) / 4, where
           S_A is the exact score_candidates() score and E(D) = sum_w
           P(w|R) log P(w|D), centred over the pool. P(w|R) = RM1 over the
           supplied seed (P(D|Q) sharpened by beta=12, 40 terms). With
           lambda set to 1, feedback returns the Track A ranking and scores.
RM_THREADS=n caps threads (max 4).
"""
import ctypes
import hashlib
import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP = os.path.join(_HERE, "cpp")
_SOURCES = ["rm_engine.cpp", "analysis.cpp", "c_api.cpp"]
_HEADERS = ["rm_engine.h", "util.h", "analysis.h"]
_FLAG_SETS = [
    ["-std=c++17", "-O3", "-march=native", "-fPIC", "-pthread", "-shared"],
    ["-std=c++17", "-O2", "-fPIC", "-pthread", "-shared"],
]

# SPUD + lead-field + SDM query likelihood (Track A), tuned on 2*SciFact +
# ANTIQUE. Track B: RM3 on the Track A score (fb_mode 1), lambda 0.95,
# expansion scale 4, estimated from the supplied seed (fb_lock off).
# best so far: 0.53633065	0.45904990	0.53655742	0.99990104	0.99815275
PARAMS: Dict[str, float] = {
    "stemmer": 2, "doc_stop": 0, "query_stop": 2, "qstop_min_len": 0, "qstop_list": 0,
    "qspell": 0,
    "smoothing": 3, "spud_mu": 800, "spud_bg": 1, "jm_lambda": 0.50, "mu": 450,
    "w_od": 0.05, "w_uw": 0.1, "uw_window": 9,
    "lead_w": 0.1, "lead_len": 20, "lead_mu": 60,
    "len_prior": 0,
    "prior_w": 0, "prior_k": 4,
    "rm_variant": 3,
    "fb_mu": 250, "fb_docs": 10, "fb_terms": 40, "beta": 12, "idf_power": 0.5,
    "idf_select": 1, "fb_qterms": 1, "support_frac": 0.20,
    "anchor_min": 0, "term_support": 2, "per_doc_cap": 1, "min_cf": 1,
    "lambda": 0.93,
    "qlen_shrink": 0, "small_seed_n": 0,
    "fb_sdm": 1,
    "fb_mode": 1, "fb_loo": 0, "fb_scale": 4,
    "fb_cov": 0.5, "fb_conf": 0, "fb_anchor_exp": 1.0,
    "fb_lock": 0,
    # Engine options that are off; listed so set_params(**PARAMS) resets them.
    "spud_l": 0, "lead_decay": 0, "fb_mass": 0, "fb_mass_pow": 1,
    "fdm_gap": 0, "fdm_w": 0, "qw_idf": 0, "prox_w": 0, "prox_a": 0.3, "prox_s": 1,
    "tok_keep1": 0, "tok_hyph": 0,
}

_lib = None

def _digest(flags: Sequence[str]) -> str:
    h = hashlib.sha256()
    for name in _SOURCES + _HEADERS:
        with open(os.path.join(_CPP, name), "rb") as fh:
            h.update(name.encode()); h.update(fh.read())
    h.update(" ".join(flags).encode())
    return h.hexdigest()[:16]


def _compile(out: str, flags: Sequence[str]) -> bool:
    """Compile the three translation units in parallel (one g++ each, the
    grading VM has 4 cores), then link. Atomic rename at the end."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cxx = os.environ.get("CXX", "g++")
    tag = f"{os.getpid()}"
    objs, procs = [], []
    cflags = [f for f in flags if f != "-shared"]
    try:
        for src in _SOURCES:
            obj = f"{out}.{src}.{tag}.o"
            objs.append(obj)
            procs.append(subprocess.Popen([cxx, *cflags, "-c", os.path.join(_CPP, src), "-o", obj],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        ok = all(p.wait(timeout=300) == 0 for p in procs)
        tmp = f"{out}.{tag}.tmp"
        if ok:
            r = subprocess.run([cxx, *flags, *objs, "-o", tmp], stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=300)
            ok = r.returncode == 0 and os.path.exists(tmp)
        if ok:
            os.replace(tmp, out)  # atomic: a concurrent importer never sees half a file
        return ok
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        for f in objs + [f"{out}.{tag}.tmp"]:
            try:
                os.remove(f)
            except OSError:
                pass


def _declare(lib) -> None:
    c = ctypes
    I32, F64 = c.POINTER(c.c_int32), c.POINTER(c.c_double)
    sig = {
        "rm_prepare": ([c.c_char_p], c.c_int),
        "rm_last_error": ([], c.c_char_p),
        "rm_set_param": ([c.c_char_p, c.c_double], c.c_int),
        "rm_get_param": ([c.c_char_p, F64], c.c_int),
        "rm_stats": ([c.c_char_p, c.c_int], c.c_int),
        "rm_score": ([c.c_char_p, c.c_char_p, c.c_int, c.c_int, I32, F64], c.c_int),
        "rm_feedback": ([c.c_char_p, c.c_char_p, c.c_int, c.c_char_p, c.c_int, c.c_int, I32, F64], c.c_int),
        "rm_explain": ([c.c_char_p, c.c_char_p, c.c_int, c.c_char_p, c.c_int, c.c_char_p, c.c_int], c.c_int),
        "rm_relevance_model": ([c.c_char_p, c.c_char_p, c.c_int, c.c_int, c.c_char_p, c.c_int], c.c_int),
        "rm_analyze": ([c.c_char_p, c.c_char_p, c.c_int], c.c_int),
    }
    for name, (args, res) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res


def _load_native():
    dirs = [os.path.join(_HERE, "_build"), os.path.join(tempfile.gettempdir(), "rm_engine_build")]
    extra = os.environ.get("RM_EXTRA_FLAGS", "").split()   # dev only, e.g. sanitizers
    for flags in ([f + extra for f in _FLAG_SETS] if extra else _FLAG_SETS):
        try:
            digest = _digest(flags)
        except OSError:
            return None
        for d in dirs:
            path = os.path.join(d, f"librm_{digest}.so")
            try:
                if not os.path.exists(path) and not _compile(path, flags):
                    continue
                lib = ctypes.CDLL(path)
                _declare(lib)
                return lib
            except OSError:
                continue
    return None


# ---------------------------------------------------------------------------
# Helpers (not part of the graded interface)
# ---------------------------------------------------------------------------

def backend() -> str:
    return "cpp"


def set_params(**kw: float) -> None:
    """Override engine parameters by name (tools/sweep.py)."""
    for k, v in kw.items():
        if _lib.rm_set_param(k.encode(), float(v)) != 0:
            raise KeyError(f"unknown parameter {k!r}")


def get_param(name: str) -> float:
    v = ctypes.c_double()
    if _lib.rm_get_param(name.encode(), ctypes.byref(v)) != 0:
        raise KeyError(name)
    return v.value


def _json_call(fn, *args):
    size = 1 << 16
    while True:
        buf = ctypes.create_string_buffer(size)
        n = fn(*args, buf, size)
        if n >= 0:
            return json.loads(buf.value.decode("utf-8", "replace"))
        size = -n + 16


def _join(ids: Sequence[str]) -> Tuple[bytes, int]:
    ids = list(ids)
    return "\n".join(ids).encode("utf-8"), len(ids)


def stats() -> dict:
    return _json_call(_lib.rm_stats)


def analyze(query: str) -> List[str]:
    """The query terms after tokenising, stopping and stemming."""
    return _json_call(_lib.rm_analyze, query.encode())


def explain_feedback(query: str, pseudo_relevant_doc_ids: List[str],
                     candidate_doc_ids: Optional[List[str]] = None) -> dict:
    """Seed weights, estimated model, lambda for one feedback call."""
    s, ns = _join(pseudo_relevant_doc_ids)
    c, nc = _join(candidate_doc_ids or [])
    return _json_call(_lib.rm_explain, query.encode(), s, ns, c, nc)


def relevance_model(query: str, pseudo_relevant_doc_ids: List[str], variant: int = 1) -> List[Tuple[str, float]]:
    """The estimated P(w|R) alone: variant 1 = RM1, 2 = RM2."""
    s, ns = _join(pseudo_relevant_doc_ids)
    return [tuple(x) for x in _json_call(_lib.rm_relevance_model, query.encode(), s, ns, int(variant))]


# ---------------------------------------------------------------------------
# Required entrypoints
# ---------------------------------------------------------------------------

def prepare(corpus_path: str) -> None:
    """One pass over the corpus (4 threads): tokenise, build P(w|C).
    Idempotent for the same path."""
    if _lib.rm_prepare(os.fspath(corpus_path).encode()) != 0:
        raise RuntimeError(_lib.rm_last_error().decode())


def _run(fn, pool: List[str], k: int, *args) -> List[Tuple[str, float]]:
    k = min(int(k), len(pool))
    if k <= 0:
        return []
    c, nc = _join(pool)
    idx = (ctypes.c_int32 * k)()
    sc = (ctypes.c_double * k)()
    n = fn(*args, c, nc, k, idx, sc)
    if n < 0:
        raise RuntimeError(_lib.rm_last_error().decode())
    return [(pool[idx[i]], float(sc[i])) for i in range(n)]


def score_candidates(query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """Rerank candidate_doc_ids with the query-likelihood model (Track A)."""
    return _run(_lib.rm_score, list(dict.fromkeys(candidate_doc_ids)), k, query.encode("utf-8"))


def relevance_model_feedback(query: str, pseudo_relevant_doc_ids: List[str],
                             candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """Rerank candidate_doc_ids with a relevance model estimated from
    pseudo_relevant_doc_ids (Tracks B/C). The seed only feeds estimation;
    output is drawn from candidate_doc_ids only."""
    s, ns = _join(list(dict.fromkeys(pseudo_relevant_doc_ids)))
    return _run(_lib.rm_feedback, list(dict.fromkeys(candidate_doc_ids)), k, query.encode("utf-8"), s, ns)


_lib = _load_native()
if _lib is None:
    raise RuntimeError("C++ engine failed to build or load")
set_params(**PARAMS)
