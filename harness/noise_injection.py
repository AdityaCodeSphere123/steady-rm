"""
harness/noise_injection.py -- a fully public, documented query-drift
stress-test recipe you can run locally (spec Section 6, "Practice
noise recipe (public)").

perturb_pseudo_relevant_set() takes a clean pseudo-relevant doc_id list
and swaps a stated fraction of it for OTHER documents from the same
candidate pool -- i.e. documents the first-pass retriever also considered
plausible enough to make the top-K, but that didn't make your seed's
top-PRF_DEPTH -- and returns the perturbed list, same length, same
doc_ids still all valid candidates. Drawing noise from within the
candidate pool rather than from the whole corpus is deliberate: it models
"plausible but wrong" contamination (a document that scored reasonably
under the first-pass retriever but isn't actually relevant), which is a
harder and more realistic case for a relevance model to reject than
wildly unrelated noise would be. Use this to test your own
relevance_model_feedback() against contaminated input before you submit.

THIS IS NOT THE GRADING RECIPE. The real Track C noise levels, sampling
strategy, and number of perturbed conditions are undisclosed (spec
Section 6, "Grading noise recipe (undisclosed)") for exactly the reason
the earlier project's held-out topics and baseline parameters were undisclosed:
a submission tuned against a known perturbation recipe is measuring its
ability to guess the recipe, not its robustness to realistic drift. Build
a feedback function that is robust across a *range* of noise conditions
and fractions, not one that happens to do well at the specific fractions
below.
"""
import random
import zlib
from typing import List, Tuple


def stable_seed(qid: str, base_seed: int) -> int:
    """A reproducible per-query seed: base_seed + a deterministic hash of
    qid. Deliberately NOT Python's builtin hash(qid) -- as of Python 3.3,
    str hashing is randomised per-process (PYTHONHASHSEED) for security
    reasons, so hash(qid) gives a DIFFERENT value every time you start a
    new Python process. Using it to seed noise injection would mean the
    same submission gets scored against a different noise draw on every
    grading run, and a student's local practice run would never match
    another local practice run either -- silently breaking the
    reproducibility this whole recipe depends on. zlib.crc32 is stable
    across processes (and platforms), which is what a reproducible seed
    actually requires."""
    return base_seed + (zlib.crc32(qid.encode("utf-8")) & 0xFFFF)

# Practice noise levels you can sweep locally (see project Section 8,
# "a plot or table ... at the highest noise level your local
# noise_injection.py practice recipe supports"). The real grading recipe
# is not required to use these same levels.
PRACTICE_NOISE_LEVELS = [0.0, 0.25, 0.5]


def perturb_pseudo_relevant_set(
    pseudo_relevant_doc_ids: List[str],
    candidate_pool_doc_ids: List[str],
    noise_fraction: float,
    rng: random.Random,
) -> List[str]:
    """Return a new list, same length as `pseudo_relevant_doc_ids`, with
    `noise_fraction` of its entries (rounded to the nearest whole document,
    at least 0) replaced by documents sampled uniformly at random from
    `candidate_pool_doc_ids` (excluding documents already in the set, so a "noise"
    document is never accidentally one you already had). At
    noise_fraction=0.0 this returns the input unchanged (as a new list,
    not perturbed at all); at noise_fraction=1.0 every entry is replaced.

    This is a simple, transparent, uniform-random contamination model --
    real query drift in the wild is not always this random (topically
    *adjacent* contamination, e.g. a different sense of an ambiguous
    query term, tends to be worse for relevance models than pure random
    noise, precisely because it is harder for term-distribution methods
    to distinguish from genuine signal). Treat this as a floor your
    feedback function should clear easily, not a ceiling for how robust
    it needs to be.
    """
    if not pseudo_relevant_doc_ids:
        return []
    if not (0.0 <= noise_fraction <= 1.0):
        raise ValueError(f"noise_fraction must be in [0.0, 1.0], got {noise_fraction!r}")

    n = len(pseudo_relevant_doc_ids)
    n_to_replace = round(n * noise_fraction)
    if n_to_replace == 0:
        return list(pseudo_relevant_doc_ids)

    kept = list(pseudo_relevant_doc_ids)
    replace_positions = rng.sample(range(n), n_to_replace)

    pool = [d for d in candidate_pool_doc_ids if d not in set(pseudo_relevant_doc_ids)]
    if not pool:
        # Degenerate case (tiny toy corpus): nothing outside the
        # pseudo-relevant set to draw noise from. Leave unperturbed
        # rather than error -- this only happens on toy-sized data.
        return kept

    n_draw = min(n_to_replace, len(pool))
    noise_docs = rng.sample(pool, n_draw)
    for pos, noise_doc in zip(replace_positions, noise_docs):
        kept[pos] = noise_doc
    return kept


def sweep(
    pseudo_relevant_doc_ids: List[str],
    candidate_pool_doc_ids: List[str],
    seed: int = 0,
    levels: List[float] = None,
) -> List[Tuple[float, List[str]]]:
    """Convenience wrapper: returns [(noise_fraction, perturbed_list), ...]
    across `levels` (defaults to PRACTICE_NOISE_LEVELS), all drawn from a
    single seeded Random instance so the sweep is reproducible."""
    rng = random.Random(seed)
    use_levels = levels if levels is not None else PRACTICE_NOISE_LEVELS
    return [
        (level, perturb_pseudo_relevant_set(pseudo_relevant_doc_ids, candidate_pool_doc_ids, level, rng))
        for level in use_levels
    ]
