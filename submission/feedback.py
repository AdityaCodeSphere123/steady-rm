"""COL764 Assignment 2: query-likelihood reranking and relevance-model feedback.

Graded entrypoints (the harness calls only these three):
    prepare(corpus_path)
    score_candidates(query, candidate_doc_ids, k=10)                  Track A
    relevance_model_feedback(query, pseudo_relevant_doc_ids,
                             candidate_doc_ids, k=10)                 Tracks B/C
They use the active model, which is the submitted one (MODELS["competition"]
with RM3) unless use_model() or the FEEDBACK_* environment variables below
select another.

Models (complete parameter sets in MODELS; all share one prepare()):
  competition  Submitted. Track A: SPUD (Polya urn) unigram query likelihood
               (spud_mu 800, df background) mixed with a lead-field model of
               the first 20 tokens (weight 0.1, Dirichlet 60), plus sequential
               dependence (ordered 0.05, unordered window 9 weight 0.1, both
               Dirichlet 450). RM3: 0.9 * S_A(D) + 0.1 * (E(D) - mean E) / 12,
               S_A the score_candidates() score, E(D) = sum_w P(w|R) log P(w|D);
               P(w|R) from the top 5 seed docs covering >= 1/4 of the query,
               P(D|Q) sharpened by beta 40, 80 terms.
  dirichlet    Required baseline: multinomial unigram query likelihood with
               Dirichlet smoothing (mu 500), no SPUD, lead field or proximity.
               RM3: 0.5 * log P(Q|D)/|Q| + 0.5 * sum_w P(w|R) log P(w|D), with
               RM1 over 10 seed docs, exact P(D|Q) = P(Q|D), 50 terms.
  jm           Same as dirichlet with Jelinek-Mercer smoothing (lambda 0.3).

Relevance models (rm=):
  1  RM1  rerank by sum_w P_RM1(w|R) log P(w|D)   (no query side)
  2  RM2  the same with the RM2 estimate of P(w|R)
  3  RM3  RM1 interpolated with the query, weight lambda on the query
          (lambda 1 returns exactly score_candidates())

Calling any combination (after fb.prepare("data/full/corpus.jsonl")):
  from submission import feedback as fb
  fb.rerank(q, pool, 10)                               competition query likelihood
  fb.rerank(q, pool, 10, model="dirichlet")            plain Dirichlet QL
  fb.rm_feedback(q, seed, pool, 10)                    competition + RM3 (= graded)
  fb.rm_feedback(q, seed, pool, 10, rm=1)              competition + RM1
  fb.rm_feedback(q, seed, pool, 10, rm=2)              competition + RM2
  fb.rm_feedback(q, seed, pool, 10, model="dirichlet", rm=1)   Dirichlet QL + RM1
  fb.rm_feedback(q, seed, pool, 10, model="dirichlet", rm=2)   Dirichlet QL + RM2
  fb.rm_feedback(q, seed, pool, 10, model="dirichlet", rm=3)   Dirichlet QL + RM3
  fb.rm_feedback(q, seed, pool, 10, model="jm", rm=3, lam=0.7) JM QL + RM3, lambda 0.7
  fb.relevance_model(q, seed, variant=2, model="dirichlet")    P(w|R) itself, [(term, p)]
  fb.use_model("dirichlet", rm=1)   graded functions now run Dirichlet QL + RM1
  fb.use_model()                    back to the submitted model
Any engine parameter can be overridden in these calls, e.g. mu=1000.

Running the harness on another combination (unset = submitted model):
  FEEDBACK_MODEL=dirichlet FEEDBACK_RM=3 FEEDBACK_LAMBDA=0.5 \
      python -m harness.run_harness --corpus ... --queries ... --qrels ... --candidates ...

Implementation: the C++ engine in submission/cpp (maths in rm_engine.cpp,
every parameter documented in rm_engine.h), loaded with ctypes and compiled
from source on first import into submission/_build/ (or the temp dir if that
is read-only). The library name carries a hash of the sources and flags, so
edited sources are always rebuilt. RM_THREADS=n caps the threads (at most 4).
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
    "fb_mu": 0, "fb_docs": 10, "fb_terms": 50, "beta": -1,
    "support_frac": 0, "idf_power": 0, "fb_qterms": 0, "fb_cov": 0, "fb_anchor_exp": 0,
    "lambda": 0.5, "fb_mode": 0, "fb_sdm": 0,
}

MODELS: Dict[str, Dict[str, float]] = {
    "competition": PARAMS,
    "dirichlet": {**PARAMS, **_BASELINE, "smoothing": 0, "mu": 500},
    "jm": {**PARAMS, **_BASELINE, "smoothing": 1, "jm_lambda": 0.3},
}
_RM = {"rm1": 1, "rm2": 2, "rm3": 3, 1: 1, 2: 2, 3: 3}

_lib = None


def _digest(flags: Sequence[str]) -> str:
    h = hashlib.sha256()
    for name in _SOURCES + _HEADERS:
        with open(os.path.join(_CPP, name), "rb") as fh:
            h.update(name.encode()); h.update(fh.read())
    h.update(" ".join(flags).encode())
    return h.hexdigest()[:16]


def _compile(out: str, flags: Sequence[str]) -> bool:
    """Compile the translation units in parallel, then link. The library is
    renamed into place atomically."""
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
            os.replace(tmp, out)  # a concurrent importer never sees half a file
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
        "rm_stats": ([c.c_char_p, c.c_int], c.c_int),
        "rm_score": ([c.c_char_p, c.c_char_p, c.c_int, c.c_int, I32, F64], c.c_int),
        "rm_feedback": ([c.c_char_p, c.c_char_p, c.c_int, c.c_char_p, c.c_int, c.c_int, I32, F64], c.c_int),
        "rm_relevance_model": ([c.c_char_p, c.c_char_p, c.c_int, c.c_int, c.c_char_p, c.c_int], c.c_int),
    }
    for name, (args, res) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res


def _load_native():
    dirs = [os.path.join(_HERE, "_build"), os.path.join(tempfile.gettempdir(), "rm_engine_build")]
    extra = os.environ.get("RM_EXTRA_FLAGS", "").split()  # dev only, e.g. sanitizers
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
    """Override engine parameters by name."""
    for k, v in kw.items():
        if _lib.rm_set_param(k.encode(), float(v)) != 0:
            raise KeyError(f"unknown parameter {k!r}")


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


def _config(model: str, rm=3, lam: Optional[float] = None, **overrides: float) -> Dict[str, float]:
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
    """Select the model the graded entrypoints use (see the module docstring)."""
    global _active
    _active = _config(model, rm, lam, **overrides)
    set_params(**_active)


@contextmanager
def _using(cfg: Dict[str, float]):
    set_params(**cfg)
    try:
        yield
    finally:
        set_params(**_active)


def rerank(query: str, candidate_doc_ids: List[str], k: int = 10, model: str = "competition",
           **overrides: float) -> List[Tuple[str, float]]:
    """Query-likelihood reranking with the given model."""
    with _using(_config(model, **overrides)):
        return score_candidates(query, candidate_doc_ids, k)


def rm_feedback(query: str, pseudo_relevant_doc_ids: List[str], candidate_doc_ids: List[str], k: int = 10,
                model: str = "competition", rm=3, lam: Optional[float] = None,
                **overrides: float) -> List[Tuple[str, float]]:
    """Relevance-model reranking with the given model and RM1/RM2/RM3."""
    with _using(_config(model, rm, lam, **overrides)):
        return relevance_model_feedback(query, pseudo_relevant_doc_ids, candidate_doc_ids, k)


def relevance_model(query: str, pseudo_relevant_doc_ids: List[str], variant: int = 1,
                    model: Optional[str] = None, **overrides: float) -> List[Tuple[str, float]]:
    """The estimated P(w|R) alone: variant 1 = RM1, 2 = RM2. model None keeps the current parameters."""
    s, ns = _join(pseudo_relevant_doc_ids)
    if model is None and not overrides:
        return [tuple(x) for x in _json_call(_lib.rm_relevance_model, query.encode(), s, ns, int(variant))]
    with _using(_config(model or "competition", **overrides)):
        return relevance_model(query, pseudo_relevant_doc_ids, variant)


# ---------------------------------------------------------------------------
# Required entrypoints
# ---------------------------------------------------------------------------

def prepare(corpus_path: str) -> None:
    """One pass over the corpus: tokenise and build the collection model.
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
use_model(os.environ.get("FEEDBACK_MODEL", "competition"),
          int(os.environ.get("FEEDBACK_RM", "3")),
          float(os.environ["FEEDBACK_LAMBDA"]) if os.environ.get("FEEDBACK_LAMBDA") else None)
