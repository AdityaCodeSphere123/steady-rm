"""COL764 Assignment 2: QL reranking and relevance model feedback.

score_candidates() reranks a candidate pool by unigram query likelihood
(Track A); relevance_model_feedback() estimates a relevance model (RM1, RM2
or RM3) from the pseudo-relevant documents and reranks the pool with it
(Tracks B and C). By default both run the submitted model,
MODELS["competition"] with RM3. The plain Dirichlet / Jelinek-Mercer
baselines and RM1/RM2 can be selected with use_model(), rerank(),
rm_feedback() or the FEEDBACK_MODEL / FEEDBACK_RM / FEEDBACK_LAMBDA
environment variables; see README.md for how to call each combination and
the results.

The scoring is done by the C++ engine in cpp/, compiled from source on first
import and loaded with ctypes.
"""
import ctypes
import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP = os.path.join(_HERE, "cpp")
_SOURCES = ["rm_engine.cpp", "analysis.cpp", "c_api.cpp"]
_HEADERS = ["rm_engine.h", "util.h", "analysis.h"]
_FLAG_SETS = [
    ["-std=c++17", "-O3", "-march=native", "-fPIC", "-pthread", "-shared"],
    ["-std=c++17", "-O2", "-fPIC", "-pthread", "-shared"],
]

# Hidden-leaderboard best (A 0.53809, MAP 0.46063, B 0.53814, C1 1.0, C2 0.99854).
PARAMS: Dict[str, float] = {
    "stemmer": 2, "query_stop": 1,
    "smoothing": 3, "spud_mu": 800, "spud_bg": 1, "mu": 450, "jm_lambda": 0.5,
    "w_od": 0.05, "w_uw": 0.1, "uw_window": 9,
    "lead_w": 0.1, "lead_len": 20, "lead_mu": 60,
    "fb_mu": 100, "fb_docs": 5, "fb_terms": 80, "beta": 40,
    "support_frac": 0.2, "idf_power": 0.5, "fb_qterms": 1,
    "fb_cov": 0.25, "fb_anchor_exp": 1.0,
    "rm_model": 3, "lambda": 0.9, "fb_mode": 1, "fb_scale": 12, "fb_sdm": 1,
}

_BASELINE = {
    "w_od": 0, "w_uw": 0, "lead_w": 0,
    "fb_mu": 0, "beta": -1,
    "support_frac": 0, "idf_power": 0, "fb_qterms": 0, "fb_cov": 0, "fb_anchor_exp": 0,
    "fb_mode": 0, "fb_sdm": 0,
}

MODELS: Dict[str, Dict[str, float]] = {
    "competition": PARAMS,
    "dirichlet": {**PARAMS, **_BASELINE, "smoothing": 0, "mu": 360,"lambda": 0.45, "fb_docs": 2, "fb_terms": 30},
    "jm": {**PARAMS, **_BASELINE, "smoothing": 1, "jm_lambda": 0.35,"lambda": 0.3, "fb_docs": 7, "fb_terms": 100},
}
_RM = {"rm1": 1, "rm2": 2, "rm3": 3, 1: 1, 2: 2, 3: 3}

_lib = None


def _digest(flags: Sequence[str]) -> str:
    """Hash of the C++ sources and compiler flags, used as the library filename."""
    h = hashlib.sha256()
    for name in _SOURCES + _HEADERS:
        with open(os.path.join(_CPP, name), "rb") as fh:
            h.update(name.encode()); h.update(fh.read())
    h.update(" ".join(flags).encode())
    return h.hexdigest()[:16]


def _compile(out: str, flags: Sequence[str]) -> bool:
    """Compile the three translation units in parallel and link them into one shared library."""
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
            r = subprocess.run([cxx, *flags, *objs, "-o", tmp], stdout=subprocess.DEVNULL,stderr=subprocess.PIPE, timeout=300)
            ok = r.returncode == 0 and os.path.exists(tmp)
        if ok:
            os.replace(tmp, out)
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
    """Tell ctypes the argument and return types of every C entry point."""
    c = ctypes
    I32, F64 = c.POINTER(c.c_int32), c.POINTER(c.c_double)
    sig = {
        "rm_prepare": ([c.c_char_p], c.c_int),
        "rm_last_error": ([], c.c_char_p),
        "rm_set_param": ([c.c_char_p, c.c_double], c.c_int),
        "rm_stats": ([c.c_char_p, c.c_int], c.c_int),
        "rm_score": ([c.c_char_p, c.c_char_p, c.c_int, c.c_int, I32, F64], c.c_int),
        "rm_feedback": ([c.c_char_p, c.c_char_p, c.c_int, c.c_char_p, c.c_int, c.c_int, I32, F64], c.c_int),
        "rm_relevance_model": ([c.c_char_p, c.c_char_p, c.c_int, c.c_int, c.c_char_p, c.c_int], c.c_int),
    }
    for name, (args, res) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res


def _load_native():
    """Load a cached build of this source, or compile one; try -O3 then a plain -O2 fallback."""
    dirs = [os.path.join(_HERE, "_build"), os.path.join(tempfile.gettempdir(), "rm_engine_build")]
    extra = os.environ.get("RM_EXTRA_FLAGS", "").split()
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


def backend() -> str:
    """Name of the scoring backend. Always "cpp" once this module has imported."""
    return "cpp"


def set_params(**kw: float) -> None:
    """Write named engine parameters (smoothing, lambda, fb_docs, ...) into the live engine."""
    for k, v in kw.items():
        if _lib.rm_set_param(k.encode(), float(v)) != 0:
            raise KeyError(f"unknown parameter {k!r}")


def _json_call(fn, *args):
    """Call a C function that writes JSON into a buffer, growing the buffer if it was too small."""
    size = 1 << 16
    while True:
        buf = ctypes.create_string_buffer(size)
        n = fn(*args, buf, size)
        if n >= 0:
            return json.loads(buf.value.decode("utf-8", "replace"))
        size = -n + 16


def _join(ids: Sequence[str]) -> Tuple[bytes, int]:
    """Pack document ids as one newline-separated byte string plus the count."""
    ids = list(ids)
    return "\n".join(ids).encode("utf-8"), len(ids)


def stats() -> dict:
    """Collection statistics from the last prepare(): docs, vocab, tokens, average length."""
    return _json_call(_lib.rm_stats)


def _config(model: str, rm=3, lam: Optional[float] = None, **overrides: float) -> Dict[str, float]:
    """Build a parameter dict for one named model and one of RM1, RM2 or RM3."""
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
    if rm not in _RM:
        raise KeyError(f"unknown relevance model {rm!r}; use 1/2/3 or 'rm1'/'rm2'/'rm3'")
    cfg = {**MODELS[model], **overrides, "rm_model": _RM[rm]}
    if lam is not None:
        cfg["lambda"] = lam
    return cfg


_active: Dict[str, float] = {}


def use_model(model: str = "competition", rm=3, lam: Optional[float] = None, **overrides: float) -> None:
    """Make the graded functions use this model until the next use_model() call."""
    global _active
    _active = _config(model, rm, lam, **overrides)
    set_params(**_active)


@contextmanager
def _using(cfg: Dict[str, float]):
    """Apply cfg for the body of a with-block, then restore the active model."""
    set_params(**cfg)
    try:
        yield
    finally:
        set_params(**_active)


def rerank(query: str, candidate_doc_ids: List[str], k: int = 10, model: str = "competition",
           **overrides: float) -> List[Tuple[str, float]]:
    """score_candidates() under a chosen model, without changing the active model afterwards."""
    with _using(_config(model, **overrides)):
        return score_candidates(query, candidate_doc_ids, k)


def rm_feedback(query: str, pseudo_relevant_doc_ids: List[str], candidate_doc_ids: List[str], k: int = 10,
                model: str = "competition", rm=3, lam: Optional[float] = None,
                **overrides: float) -> List[Tuple[str, float]]:
    """relevance_model_feedback() for one model and one RM variant, then restore the active model."""
    with _using(_config(model, rm, lam, **overrides)):
        return relevance_model_feedback(query, pseudo_relevant_doc_ids, candidate_doc_ids, k)


def relevance_model(query: str, pseudo_relevant_doc_ids: List[str], variant: int = 1,
                    model: Optional[str] = None, **overrides: float) -> List[Tuple[str, float]]:
    """Return P(w|R) as (term, probability) pairs; variant 1 is RM1 and variant 2 is RM2."""
    s, ns = _join(pseudo_relevant_doc_ids)
    if model is None and not overrides:
        return [tuple(x) for x in _json_call(_lib.rm_relevance_model, query.encode(), s, ns, int(variant))]
    with _using(_config(model or "competition", **overrides)):
        return relevance_model(query, pseudo_relevant_doc_ids, variant)


def prepare(corpus_path: str) -> None:
    """Read the corpus once and store collection background statistics P(w|C)."""
    if _lib.rm_prepare(os.fspath(corpus_path).encode()) != 0:
        raise RuntimeError(_lib.rm_last_error().decode())


def _run(fn, pool: List[str], k: int, *args) -> List[Tuple[str, float]]:
    """Call a C ranker and turn pool indexes back into (doc_id, score) pairs."""
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
    """Rerank the given candidate pool by unigram query likelihood. Graded Track A entry point."""
    return _run(_lib.rm_score, list(dict.fromkeys(candidate_doc_ids)), k, query.encode("utf-8"))


def relevance_model_feedback(query: str, pseudo_relevant_doc_ids: List[str],
                             candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """Estimate RM1, RM2 or RM3 from the seed ids and rerank the candidate pool. Graded Tracks B and C."""
    s, ns = _join(list(dict.fromkeys(pseudo_relevant_doc_ids)))
    return _run(_lib.rm_feedback, list(dict.fromkeys(candidate_doc_ids)), k, query.encode("utf-8"), s, ns)


_lib = _load_native()
if _lib is None:
    raise RuntimeError("C++ engine failed to build or load")
use_model(os.environ.get("FEEDBACK_MODEL", "competition"),
          int(os.environ.get("FEEDBACK_RM", "3")),
          float(os.environ["FEEDBACK_LAMBDA"]) if os.environ.get("FEEDBACK_LAMBDA") else None)
