"""Python fallback used only when the C++ engine fails to build."""
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Tunable constants. Everything the report discusses is named here.
# ---------------------------------------------------------------------------

# --- Track A: query likelihood --------------------------------------------
# Dirichlet mu. Zhai & Lafferty (SIGIR 2001) find the optimum tracks
# average document length rather than sitting at one universal value, so
# mu is derived in prepare() as MU_AVGDL_FACTOR * avgdl, clamped. The
# clamp matters more than the factor: it keeps a pathological collection
# (near-duplicate titles at one end, full articles at the other) inside a
# range where Dirichlet still behaves.
MU_AVGDL_FACTOR = 12.0
MU_MIN, MU_MAX = 250.0, 2500.0

# Weight on the rank prior carried by the order of candidate_doc_ids,
# fused with the LM score after within-pool z-normalisation. 0.0 disables
# it and leaves a pure LM ranking.
RANK_PRIOR_WEIGHT = 0.30
RANK_PRIOR_K = 60.0  # the 60 of standard reciprocal-rank fusion

# Drop stopwords from queries only once the query is long enough that they
# stop being discriminative. Short keyword queries keep every term.
QUERY_STOPWORD_MIN_LEN = 4

# --- Tracks B/C: relevance model ------------------------------------------
FEEDBACK_MU = 400.0        # lighter smoothing when reading a seed document
MAX_FEEDBACK_DOCS = 10     # cap on surviving seed documents
MAX_EXPANSION_TERMS = 60   # vocabulary of the truncated relevance model

WEIGHT_BETA = 1.6          # softmax sharpness on P(D|Q)
TRIM_FLOOR = 0.15          # drop seed docs below this x the best weight
MIN_TERM_SUPPORT = 2       # distinct seed docs that must attest a term
MIN_ANCHORED = 2           # query-anchored filter needs this many docs
PER_DOC_TERM_CAP = 0.20    # winsorise one document's P(w|D) contribution

LAMBDA_MIN, LAMBDA_MAX = 0.45, 0.85   # adaptive interpolation range
LAMBDA_FIXED: Optional[float] = None  # set a float to pin lam (ablations)

# Expansion terms rarer than this in the collection are almost always
# corpus junk (OCR fragments, identifiers) rather than topical signal.
MIN_EXPANSION_CF = 2

# Ablation switches, all default-on. The report's ablation table flips
# these one at a time.
USE_SOFT_WEIGHTS = True
USE_TRIMMING = True
USE_QUERY_ANCHOR = True
USE_TERM_CONSENSUS = True
USE_ADAPTIVE_LAMBDA = True
USE_STEMMER = True

_LOG_FLOOR = -50.0  # same floor convention as lm_utils

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A short, conventional stoplist. Applied only to QUERIES and to
# expansion-term selection, never to documents: removing stopwords from
# documents would distort |D| and the background model for no gain, since
# Dirichlet smoothing already drives a stopword's contribution to roughly
# the same value for every document.
_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself
no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what
when where which while who whom why will with you your yours yourself yourselves
""".split())

_stem_cache: Dict[str, str] = {}


def _s_stem(word: str) -> str:
    """Conservative plural stripper (the "S-stemmer", Harman 1991).

    Deliberately not Porter. On a ~100-document candidate pool the win
    from stemming is recall of morphological variants of query terms;
    Porter's aggressive conflations (organization/organ) buy little of
    that and cost precision, and a mis-conflation inside the RELEVANCE
    MODEL is worse than one inside a first-pass ranker, because the wrong
    term is then promoted into the expanded query for every candidate at
    once.
    """
    if len(word) < 4 or word[-1] != "s":
        return word
    if word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        if word[-3] in "sxzh":
            return word[:-2]
        return word[:-1]
    return word[:-1]


def tokenize(text: str) -> List[str]:
    """The one tokeniser, applied identically to documents and queries."""
    raw = _TOKEN_RE.findall(text.lower())
    if not USE_STEMMER:
        return [t for t in raw if len(t) > 1]
    cache = _stem_cache
    out = []
    for t in raw:
        if len(t) < 2:
            continue
        s = cache.get(t)
        if s is None:
            s = _s_stem(t)
            cache[t] = s
        out.append(s)
    return out


def _query_terms(query: str) -> List[str]:
    """Query tokens, with stopwords dropped only for verbose queries."""
    terms = tokenize(query)
    if len(terms) >= QUERY_STOPWORD_MIN_LEN:
        content = [t for t in terms if t not in _STOPWORDS]
        if content:
            return content
    return terms


# ---------------------------------------------------------------------------
# Collection statistics
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class _Collection:
    """Collection-wide unigram statistics from one pass over the corpus.

    Per-document term counts are NOT retained from that pass. At most
    ~100 documents per query are ever scored, so we keep byte offsets and
    re-read the handful we are actually asked about, caching as we go.
    Peak memory during prepare() is then proportional to the vocabulary
    rather than to the corpus text, which is what keeps a
    several-hundred-thousand-document collection comfortable inside the
    10-minute prepare() budget (docs/GRADING.md).
    """

    __slots__ = (
        "corpus_path", "cf", "total_tokens", "doc_len", "offsets",
        "n_docs", "avgdl", "mu", "_counts_cache", "_texts",
    )

    def __init__(self) -> None:
        self.corpus_path: str = ""
        self.cf: Counter = Counter()
        self.total_tokens: int = 0
        self.doc_len: Dict[str, int] = {}
        self.offsets: Dict[str, Tuple[int, int]] = {}
        self.n_docs: int = 0
        self.avgdl: float = 0.0
        self.mu: float = 1000.0
        self._counts_cache: Dict[str, Counter] = {}
        self._texts: Dict[str, str] = {}

    @classmethod
    def from_path(cls, corpus_path: str) -> "_Collection":
        self = cls()
        self.corpus_path = corpus_path
        cf = self.cf
        doc_len = self.doc_len
        offsets = self.offsets
        total = 0

        with open(corpus_path, "rb") as fh:
            pos = 0
            for raw in fh:
                n = len(raw)
                line = raw.strip()
                if not line:
                    pos += n
                    continue
                obj = json.loads(line)
                doc_id = obj["doc_id"]
                tokens = tokenize(obj.get("text", "") or "")
                cf.update(tokens)
                doc_len[doc_id] = len(tokens)
                offsets[doc_id] = (pos, n)
                total += len(tokens)
                pos += n

        self.total_tokens = total
        self.n_docs = len(doc_len)
        self.avgdl = (total / self.n_docs) if self.n_docs else 0.0
        self.mu = (
            _clamp(MU_AVGDL_FACTOR * self.avgdl, MU_MIN, MU_MAX)
            if self.avgdl else 1000.0
        )
        return self

    def p_collection(self, term: str) -> float:
        if not self.total_tokens:
            return 0.0
        return self.cf.get(term, 0) / self.total_tokens

    def counts(self, doc_id: str) -> Counter:
        """Term counts for one document, read and cached on first use."""
        hit = self._counts_cache.get(doc_id)
        if hit is not None:
            return hit
        text = self._text(doc_id)
        counts = Counter(tokenize(text)) if text else Counter()
        self._counts_cache[doc_id] = counts
        return counts

    def _text(self, doc_id: str) -> str:
        cached = self._texts.get(doc_id)
        if cached is not None:
            return cached
        span = self.offsets.get(doc_id)
        if span is None:
            return ""
        offset, length = span
        try:
            with open(self.corpus_path, "rb") as fh:
                fh.seek(offset)
                raw = fh.read(length)
            text = json.loads(raw.strip()).get("text", "") or ""
        except (OSError, ValueError, KeyError):
            text = ""
        self._texts[doc_id] = text
        return text

    def length(self, doc_id: str) -> int:
        return self.doc_len.get(doc_id, 0)


_STATS: Optional[_Collection] = None


# ---------------------------------------------------------------------------
# Required entrypoint 1: prepare()
# ---------------------------------------------------------------------------


def prepare(corpus_path: str) -> None:
    """One pass over the corpus to build background statistics.

    Idempotent: re-calling with the same path is a no-op, so the
    conformance tests' repeated prepare() calls cost nothing.
    """
    global _STATS
    path = corpus_path
    if os.path.isdir(path):
        nested = os.path.join(path, "corpus.jsonl")
        if os.path.isfile(nested):
            path = nested
    if _STATS is not None and _STATS.corpus_path == path:
        return
    _STATS = _Collection.from_path(path)


def _require_stats() -> _Collection:
    if _STATS is None:
        raise RuntimeError(
            "prepare(corpus_path) must be called before any retrieval call; "
            "the harness always does this."
        )
    return _STATS


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


def _dirichlet_log_prob(tf: int, dlen: int, p_c: float, mu: float) -> float:
    num = tf + mu * p_c
    den = dlen + mu
    if num <= 0.0 or den <= 0.0:
        return _LOG_FLOOR
    return math.log(num / den)


def _query_likelihood(
    terms: Sequence[str], doc_ids: Sequence[str], stats: _Collection
) -> Dict[str, float]:
    """log P(Q|D) for each document, Dirichlet-smoothed.

    P(w|C) is looked up once per query term rather than once per
    (term, document) pair -- with ~100 candidates that is the difference
    between ~100 and ~10,000 dictionary probes per query.
    """
    mu = stats.mu
    p_c = {t: stats.p_collection(t) for t in set(terms)}
    out: Dict[str, float] = {}
    for doc_id in doc_ids:
        counts = stats.counts(doc_id)
        dlen = stats.length(doc_id)
        total = 0.0
        for t in terms:
            total += _dirichlet_log_prob(counts.get(t, 0), dlen, p_c[t], mu)
        out[doc_id] = total
    return out


def _z_normalise(values: Dict[str, float]) -> Dict[str, float]:
    """Within-pool z-scores. A degenerate (all-equal) pool maps to zeros,
    making fusion a no-op there rather than an amplifier of noise."""
    if not values:
        return {}
    xs = list(values.values())
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    sd = math.sqrt(var)
    if sd < 1e-12:
        return {key: 0.0 for key in values}
    return {key: (x - mean) / sd for key, x in values.items()}


def _rank_prior(doc_ids: Sequence[str]) -> Dict[str, float]:
    """Reciprocal-rank prior from the order of the provided pool.

    `candidate_doc_ids` arrives best-first from the reference retriever
    (harness/candidates_io.py). That ordering is an input, not a secret,
    and it is evidence from a source independent of our own LM -- exactly
    the condition under which rank fusion reliably helps. It is fused at
    a deliberately modest weight and z-normalised first, so a pool whose
    order is uninformative (the toy set, where most candidates are tied
    at 0.0) contributes a flat, harmless signal rather than a misleading
    one.
    """
    return {doc_id: 1.0 / (RANK_PRIOR_K + i) for i, doc_id in enumerate(doc_ids)}


def _fuse_and_rank(
    lm_scores: Dict[str, float], doc_ids: Sequence[str], k: int
) -> List[Tuple[str, float]]:
    """Fuse LM score with the pool's rank prior and return the top k.
    Ties break on doc_id so the output is deterministic across runs."""
    if not lm_scores:
        return []
    if RANK_PRIOR_WEIGHT <= 0.0:
        fused = dict(lm_scores)
    else:
        z_lm = _z_normalise(lm_scores)
        z_prior = _z_normalise(_rank_prior(doc_ids))
        fused = {
            doc_id: z_lm[doc_id] + RANK_PRIOR_WEIGHT * z_prior.get(doc_id, 0.0)
            for doc_id in lm_scores
        }
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[: max(0, k)]


def _dedup(doc_ids: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for d in doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Required entrypoint 2: score_candidates()  [Track A]
# ---------------------------------------------------------------------------


def score_candidates(
    query: str, candidate_doc_ids: List[str], k: int = 10
) -> List[Tuple[str, float]]:
    """Rerank the provided candidate pool with a Dirichlet-smoothed
    unigram query-likelihood model, fused with the pool's own rank prior.
    Returns up to k (doc_id, score) pairs, best first, drawn only from
    `candidate_doc_ids`."""
    stats = _require_stats()
    pool = _dedup(candidate_doc_ids)
    if not pool:
        return []
    terms = _query_terms(query)
    if not terms:
        # No usable query terms: fall back to the pool's own order rather
        # than returning nothing, which would score 0 on every metric.
        return [(doc_id, -float(i)) for i, doc_id in enumerate(pool[: max(0, k)])]
    lm = _query_likelihood(terms, pool, stats)
    return _fuse_and_rank(lm, pool, k)


# ---------------------------------------------------------------------------
# Relevance model: RM1 -> RM3
# ---------------------------------------------------------------------------


def _seed_weights(
    terms: Sequence[str], seed_ids: Sequence[str], stats: _Collection
) -> List[Tuple[str, float]]:
    """P(D|Q) over the seed set: softmax of per-query-term mean log
    likelihood, then trimmed.

    Dividing by |Q| before the softmax matters: it makes WEIGHT_BETA mean
    the same thing for a two-word query and a twelve-word one, so the
    sharpness of the weighting does not silently drift with query length.
    """
    seed = [d for d in _dedup(seed_ids) if d in stats.doc_len]
    if not seed:
        return []
    raw = _query_likelihood(terms, seed, stats)
    n_terms = max(1, len(terms))
    per_term = {d: raw[d] / n_terms for d in seed}

    if not USE_SOFT_WEIGHTS:
        w = 1.0 / len(seed)
        return [(d, w) for d in seed]

    best = max(per_term.values())
    exp_w = {d: math.exp(WEIGHT_BETA * (s - best)) for d, s in per_term.items()}

    if USE_TRIMMING:
        cutoff = TRIM_FLOOR * max(exp_w.values())
        kept = {d: w for d, w in exp_w.items() if w >= cutoff}
        if kept:
            exp_w = kept

    ordered = sorted(exp_w.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_FEEDBACK_DOCS]
    total = sum(w for _d, w in ordered)
    if total <= 0.0:
        w = 1.0 / len(ordered)
        return [(d, w) for d, _ in ordered]
    return [(d, w / total) for d, w in ordered]


def _apply_query_anchor(
    seed_ids: Sequence[str], terms: Sequence[str], stats: _Collection
) -> List[str]:
    """Keep only seed documents containing at least one query term,
    provided enough of them do.

    A poisoned seed document sharing no vocabulary with the query has no
    business informing what "documents relevant to this query" look like.
    Gated on MIN_ANCHORED so a genuinely hard query, where few documents
    match any query term literally, is not left with an empty feedback set.
    """
    if not USE_QUERY_ANCHOR:
        return list(seed_ids)
    term_set = set(terms)
    anchored = [d for d in seed_ids if any(t in stats.counts(d) for t in term_set)]
    if len(anchored) >= MIN_ANCHORED:
        return anchored
    return list(seed_ids)


def _estimate_rm1(
    weighted_seed: Sequence[Tuple[str, float]], stats: _Collection
) -> Dict[str, float]:
    """RM1: P(w|R) = sum_D P(w|D) P(D|Q), with per-document winsorising
    and term-consensus filtering, truncated to MAX_EXPANSION_TERMS."""
    if not weighted_seed:
        return {}

    accum: Dict[str, float] = defaultdict(float)
    support: Dict[str, int] = defaultdict(int)

    for doc_id, weight in weighted_seed:
        counts = stats.counts(doc_id)
        dlen = stats.length(doc_id)
        denom = dlen + FEEDBACK_MU
        if denom <= 0:
            continue
        for term, tf in counts.items():
            p_wd = (tf + FEEDBACK_MU * stats.p_collection(term)) / denom
            if p_wd > PER_DOC_TERM_CAP:
                p_wd = PER_DOC_TERM_CAP
            accum[term] += weight * p_wd
            support[term] += 1

    min_support = (
        MIN_TERM_SUPPORT if (USE_TERM_CONSENSUS and len(weighted_seed) >= 3) else 1
    )
    filtered = {
        term: mass
        for term, mass in accum.items()
        if support[term] >= min_support
        and term not in _STOPWORDS
        and stats.cf.get(term, 0) >= MIN_EXPANSION_CF
    }
    if not filtered:
        filtered = {t: m for t, m in accum.items() if t not in _STOPWORDS}
    if not filtered:
        return {}

    top = sorted(filtered.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_EXPANSION_TERMS]
    total = sum(m for _t, m in top)
    if total <= 0.0:
        return {}
    return {t: m / total for t, m in top}


def _seed_coherence(
    weighted_seed: Sequence[Tuple[str, float]], stats: _Collection
) -> float:
    """Mean pairwise cosine similarity of the surviving seed documents, on
    IDF-weighted term vectors, mapped to [0, 1].

    Documents about one topic look alike; a set into which unrelated
    documents have been swapped does not. Computed from the seed alone --
    no judgments, and no knowledge of whether this call is clean or
    perturbed.
    """
    docs = [d for d, _w in weighted_seed]
    if len(docs) < 2:
        return 0.5
    n_docs = max(1, stats.n_docs)
    vectors = []
    for doc_id in docs:
        counts = stats.counts(doc_id)
        if not counts:
            vectors.append(({}, 0.0))
            continue
        vec = {}
        for term, tf in counts.items():
            if term in _STOPWORDS:
                continue
            cf = stats.cf.get(term, 0)
            if cf <= 0:
                continue
            idf = math.log(1.0 + n_docs / cf)
            vec[term] = (1.0 + math.log(tf)) * idf
        norm = math.sqrt(sum(v * v for v in vec.values()))
        vectors.append((vec, norm))

    sims, pairs = 0.0, 0
    for i in range(len(vectors)):
        vi, ni = vectors[i]
        if ni <= 0.0:
            continue
        for j in range(i + 1, len(vectors)):
            vj, nj = vectors[j]
            if nj <= 0.0:
                continue
            small, large = (vi, vj) if len(vi) <= len(vj) else (vj, vi)
            dot = 0.0
            for term, val in small.items():
                other = large.get(term)
                if other:
                    dot += val * other
            sims += dot / (ni * nj)
            pairs += 1
    if pairs == 0:
        return 0.5
    return _clamp(sims / pairs, 0.0, 1.0)


def _model_clarity(rm: Dict[str, float], stats: _Collection) -> float:
    """KL(P(.|R) || P(.|C)), squashed to [0, 1].

    The clarity score (Cronen-Townsend et al., SIGIR 2002) used as a drift
    detector: a relevance model estimated from a coherent topical set sits
    far from the collection background; one estimated from a set of
    unrelated documents collapses towards it.
    """
    if not rm:
        return 0.0
    kl = 0.0
    for term, p_r in rm.items():
        p_c = stats.p_collection(term)
        if p_c <= 0.0 or p_r <= 0.0:
            continue
        kl += p_r * math.log(p_r / p_c)
    # ~6 nats is a strongly focused model on a typical collection; the
    # exact scale only sets where the middle of the lambda range falls.
    return _clamp(kl / 6.0, 0.0, 1.0)


def _adaptive_lambda(
    weighted_seed: Sequence[Tuple[str, float]], rm: Dict[str, float], stats: _Collection
) -> float:
    """Pick the RM3 interpolation weight from how trustworthy the seed looks.

    lambda is the weight on the ORIGINAL query, so high lambda = cautious.
    Quality 1.0 -> LAMBDA_MIN (lean on the relevance model);
    quality 0.0 -> LAMBDA_MAX (fall back towards what the user typed).
    """
    if LAMBDA_FIXED is not None:
        return _clamp(LAMBDA_FIXED, 0.0, 1.0)
    if not USE_ADAPTIVE_LAMBDA:
        return 0.5 * (LAMBDA_MIN + LAMBDA_MAX)
    if not weighted_seed or not rm:
        return 1.0  # nothing usable: pure query likelihood

    coherence = _seed_coherence(weighted_seed, stats)
    clarity = _model_clarity(rm, stats)
    # Coherence is the more direct contamination signal; clarity mostly
    # guards the case where a seed is self-similar but generic.
    quality = _clamp(0.65 * coherence + 0.35 * clarity, 0.0, 1.0)
    return LAMBDA_MAX - (LAMBDA_MAX - LAMBDA_MIN) * quality


def _query_model(terms: Sequence[str]) -> Dict[str, float]:
    """Maximum-likelihood P(w|Q) over the query's own terms."""
    counts = Counter(terms)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {t: c / total for t, c in counts.items()}


def _rm3_interpolate(
    rm: Dict[str, float], qm: Dict[str, float], lam: float
) -> Dict[str, float]:
    """P_RM3(w) = lam P(w|Q) + (1 - lam) P(w|R)."""
    out: Dict[str, float] = {}
    for term, p in qm.items():
        out[term] = lam * p
    for term, p in rm.items():
        out[term] = out.get(term, 0.0) + (1.0 - lam) * p
    total = sum(out.values())
    if total <= 0.0:
        return dict(qm)
    return {t: p / total for t, p in out.items()}


def _rank_by_model(
    model: Dict[str, float], doc_ids: Sequence[str], stats: _Collection, k: int
) -> List[Tuple[str, float]]:
    """Rank by cross-entropy: sum_w P_model(w) log P(w|D), Dirichlet-smoothed.

    This is -KL(P_model || P(.|D)) plus the model's own entropy, which is
    constant across documents for a given query -- the same ranking as the
    primer's KL formulation, computed without the constant.
    """
    if not model:
        return []
    mu = stats.mu
    p_c = {t: stats.p_collection(t) for t in model}
    scores: Dict[str, float] = {}
    for doc_id in doc_ids:
        counts = stats.counts(doc_id)
        dlen = stats.length(doc_id)
        total = 0.0
        for term, weight in model.items():
            total += weight * _dirichlet_log_prob(
                counts.get(term, 0), dlen, p_c[term], mu
            )
        scores[doc_id] = total
    return _fuse_and_rank(scores, doc_ids, k)


# ---------------------------------------------------------------------------
# Required entrypoint 3: relevance_model_feedback()  [Tracks B and C]
# ---------------------------------------------------------------------------


def relevance_model_feedback(
    query: str,
    pseudo_relevant_doc_ids: List[str],
    candidate_doc_ids: List[str],
    k: int = 10,
) -> List[Tuple[str, float]]:
    """Rerank `candidate_doc_ids` under an RM3 model estimated from
    `pseudo_relevant_doc_ids`.

    The seed only ever feeds estimation; the pool being ranked and
    returned from is `candidate_doc_ids`, always.
    """
    stats = _require_stats()
    pool = _dedup(candidate_doc_ids)
    if not pool:
        return []
    terms = _query_terms(query)
    if not terms:
        return [(doc_id, -float(i)) for i, doc_id in enumerate(pool[: max(0, k)])]

    seed = [d for d in _dedup(pseudo_relevant_doc_ids) if d in stats.doc_len]
    if not seed:
        # No usable feedback: this is exactly query likelihood, which is
        # the right answer here, not a degraded one.
        return score_candidates(query, candidate_doc_ids, k)

    seed = _apply_query_anchor(seed, terms, stats)
    weighted_seed = _seed_weights(terms, seed, stats)
    rm = _estimate_rm1(weighted_seed, stats)
    if not rm:
        return score_candidates(query, candidate_doc_ids, k)

    lam = _adaptive_lambda(weighted_seed, rm, stats)
    model = _rm3_interpolate(rm, _query_model(terms), lam)
    return _rank_by_model(model, pool, stats, k)


# ---------------------------------------------------------------------------
# Introspection helper for the report's ablation tables. Not part of the
# graded interface; the harness never calls this.
# ---------------------------------------------------------------------------


def explain_feedback(
    query: str, pseudo_relevant_doc_ids: List[str], candidate_doc_ids: List[str]
) -> dict:
    """Return the intermediate quantities behind one feedback call."""
    stats = _require_stats()
    terms = _query_terms(query)
    seed = [d for d in _dedup(pseudo_relevant_doc_ids) if d in stats.doc_len]
    anchored = _apply_query_anchor(seed, terms, stats)
    weighted_seed = _seed_weights(terms, anchored, stats)
    rm = _estimate_rm1(weighted_seed, stats)
    lam = _adaptive_lambda(weighted_seed, rm, stats)
    return {
        "query_terms": terms,
        "seed_in": list(pseudo_relevant_doc_ids),
        "seed_after_anchor": anchored,
        "seed_weights": weighted_seed,
        "coherence": _seed_coherence(weighted_seed, stats),
        "clarity": _model_clarity(rm, stats),
        "lambda": lam,
        "top_expansion_terms": sorted(rm.items(), key=lambda kv: -kv[1])[:15],
        "mu": stats.mu,
        "avgdl": stats.avgdl,
    }
