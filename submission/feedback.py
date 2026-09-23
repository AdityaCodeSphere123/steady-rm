"""
submission/feedback.py -- THE REQUIRED COMPETITION ENTRYPOINT.

    prepare(corpus_path)
    score_candidates(query, candidate_doc_ids, k=10)
    relevance_model_feedback(query, pseudo_relevant_doc_ids, candidate_doc_ids, k=10)

All retrieval work is done in C++ (submission/cpp/, loaded with ctypes).
The shared library is compiled from source on first import, on the machine
running the harness, into submission/_build/ (or the temp dir if that is
read-only). Its file name carries a hash of sources + flags, so edited
sources are always rebuilt and a binary built elsewhere is never loaded
(a -march=native binary from another CPU could crash the process). If
building or loading fails, the module falls back to the pure-Python
implementation in _feedback_py.py, so the entrypoints always work.

Model (maths in cpp/rm_engine.cpp, every knob in cpp/rm_engine.h):
  Track A  Dirichlet query likelihood (JM optional) + sequential-dependence
           proximity features + rank prior from the candidate order,
           fused by within-pool z-score.
  Track B  RM3: lambda * query part + (1 - lambda) * relevance model;
           the relevance model is RM1, RM2 or a mixture of the two.
  Track C  Robustness from how the seed is weighted and filtered:
           P(D|Q)-weighted seeds, query anchoring, term consensus, RM2's
           co-occurrence requirement, optional outlier down-weighting and
           adaptive lambda. Nothing reads score_candidates() output or keys
           off the query text. The only cache is collection counts of query
           term PAIRS: background statistics, independent of any seed.

RM_FORCE_PYTHON=1 forces the fallback; RM_THREADS=n caps threads (max 4).
"""
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
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

# Final parameters (names as in cpp/rm_engine.h); anything not listed keeps
# the C++ default. Tuned on a Cranfield proxy -- RETUNE ON THE DEV SET with
# tools/sweep.py before the deadline.
PARAMS: Dict[str, float] = {
    "stemmer": 2, "doc_stop": 0,
    "query_stop": 1, "qstop_min_len": 3, "qstop_list": 1,
    "smoothing": 0, "mu": 0, "mu_factor": 4, "mu_min": 250, "mu_max": 2500, "mu_mode": 0, "mu_loo_scale": 5,
    "w_od": 0.20, "w_uw": 0.10, "uw_window": 8,
    "lead_w": 0.15, "lead_len": 12, "lead_mu": 50,   # A1 title-window idea as a field LM
    "prior_w": 0.30,
    "rm_variant": 3, "rm2_mix": 0.0,
    "fb_mu": 0, "fb_docs": 10, "fb_terms": 20, "beta": 1.0,
    "anchor_min": 2, "term_support": 2, "per_doc_cap": 1.0, "min_cf": 2,
    "agree_gamma": 1.0, "lambda": -1,
    "qlen_shrink": 16,   # long (document-like) queries: shrink expansion (A1 rule)
    "qpp_w": 0.0,        # pool-QPP gate: tune on your collections with tools/multi_sweep.py
    "fb_sdm": 1, "fb_prior_w": -1,
}

_lib = None
_py = None


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def _digest(flags: Sequence[str]) -> str:
    h = hashlib.sha256()
    for name in _SOURCES + _HEADERS:
        with open(os.path.join(_CPP, name), "rb") as fh:
            h.update(name.encode()); h.update(fh.read())
    h.update(" ".join(flags).encode())
    return h.hexdigest()[:16]


def _compile(out: str, flags: Sequence[str]) -> bool:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.{os.getpid()}.tmp"
    cmd = [os.environ.get("CXX", "g++"), *flags, *[os.path.join(_CPP, s) for s in _SOURCES], "-o", tmp]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return False
    if r.returncode != 0 or not os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False
    os.replace(tmp, out)  # atomic: a concurrent importer never sees half a file
    return True


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
    if os.environ.get("RM_FORCE_PYTHON") == "1":
        return None
    dirs = [os.path.join(_HERE, "_build"), os.path.join(tempfile.gettempdir(), "rm_engine_build")]
    for flags in _FLAG_SETS:
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


def _load_fallback():
    spec = importlib.util.spec_from_file_location("_rm_feedback_py", os.path.join(_HERE, "_feedback_py.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helpers (not part of the graded interface)
# ---------------------------------------------------------------------------

def backend() -> str:
    return "cpp" if _lib is not None else "python"


def set_params(**kw: float) -> None:
    """Override engine parameters by name (tools/sweep.py)."""
    if _lib is None:
        return
    for k, v in kw.items():
        if _lib.rm_set_param(k.encode(), float(v)) != 0:
            raise KeyError(f"unknown parameter {k!r}")


def get_param(name: str) -> float:
    v = ctypes.c_double()
    if _lib is None or _lib.rm_get_param(name.encode(), ctypes.byref(v)) != 0:
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
    return _json_call(_lib.rm_stats) if _lib is not None else {}


def analyze(query: str) -> List[str]:
    """The query terms after tokenising, stopping and stemming."""
    return _json_call(_lib.rm_analyze, query.encode()) if _lib is not None else []


def explain_feedback(query: str, pseudo_relevant_doc_ids: List[str],
                     candidate_doc_ids: Optional[List[str]] = None) -> dict:
    """Seed weights, estimated model, lambda for one feedback call."""
    if _lib is None:
        return _py.explain_feedback(query, pseudo_relevant_doc_ids, candidate_doc_ids or [])
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
    if _lib is None:
        return _py.prepare(corpus_path)
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
    if _lib is None:
        return _py.score_candidates(query, candidate_doc_ids, k)
    return _run(_lib.rm_score, list(dict.fromkeys(candidate_doc_ids)), k, query.encode("utf-8"))


def relevance_model_feedback(query: str, pseudo_relevant_doc_ids: List[str],
                             candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """Rerank candidate_doc_ids with a relevance model estimated from
    pseudo_relevant_doc_ids (Tracks B/C). The seed only feeds estimation;
    output is drawn from candidate_doc_ids only."""
    if _lib is None:
        return _py.relevance_model_feedback(query, pseudo_relevant_doc_ids, candidate_doc_ids, k)
    s, ns = _join(list(dict.fromkeys(pseudo_relevant_doc_ids)))
    return _run(_lib.rm_feedback, list(dict.fromkeys(candidate_doc_ids)), k, query.encode("utf-8"), s, ns)


_lib = _load_native()
if _lib is not None:
    set_params(**PARAMS)
else:
    _py = _load_fallback()
