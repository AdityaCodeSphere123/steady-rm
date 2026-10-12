# Submission Interface

Your entire submission is `submission/feedback.py`, exposing exactly
these three functions (see the file's module docstring for full detail):

```python
def prepare(corpus_path: str) -> None: ...

def score_candidates(
    query: str, candidate_doc_ids: List[str], k: int = 10
) -> List[Tuple[str, float]]: ...

def relevance_model_feedback(
    query: str, pseudo_relevant_doc_ids: List[str], candidate_doc_ids: List[str], k: int = 10
) -> List[Tuple[str, float]]: ...
```

Do not rename these, change their signatures, or move them out of
`submission/feedback.py` — the harness imports and calls exactly these
three names.

## You never rank the whole corpus

Every retrieval call you implement is scoped to a PROVIDED candidate
pool — `candidate_doc_ids`, typically the top-100 documents from an
undisclosed reference retriever (spec Section 6) — not the full
corpus. `score_candidates()` reranks that pool with your own unigram LM;
`relevance_model_feedback()` reranks the same pool using a relevance
model estimated from `pseudo_relevant_doc_ids`. Both functions must
return doc_ids drawn ONLY from `candidate_doc_ids` — this is
conformance-checked (see "Anti-gaming checks" below). `prepare()` still
reads the whole corpus once, to build collection-wide background
statistics (Section 3.1) — that is a legitimate, one-time linear pass,
not the thing being removed here. What's removed is ever having to
score, rank, or index the corpus at large per query.

## No build/load process split this time

the earlier project ran `build_index()` and `load_index()`/`retrieve()` as two
separate processes specifically to verify your index persistence was
real. This project does not have that requirement — `prepare()` and
both retrieval functions run **in the same process**, called directly by
`harness/run_harness.py`. Module-level state set in `prepare()` is
expected to still be there when the retrieval functions run. This is a
deliberate simplification for this project's compressed pre-midterm
timeline, not a signal that persistence stopped mattering in general.

## Call-order independence

`relevance_model_feedback(query, pseudo_relevant_doc_ids, candidate_doc_ids, k)`
must behave correctly no matter what `pseudo_relevant_doc_ids` it is
given, and no matter whether `score_candidates()` has ever been called
for that exact query text in this process. Concretely:

- **Do not** cache "the real top-k for this query" inside
  `score_candidates()` and read that cache back inside
  `relevance_model_feedback()` instead of trusting the
  `pseudo_relevant_doc_ids` argument you were actually passed. The
  grading harness sometimes calls `relevance_model_feedback()` with a
  noise-perturbed set (spec Section 7, Track C) specifically to see
  whether your feedback estimation is sensitive to *that exact input* —
  silently substituting your own clean top-k defeats the entire point of
  the noise-injection stress test and will visibly show up as suspiciously
  perfect Track C retention, which the understanding check (Section 7.1)
  is designed to catch.
- **Do not** assume `relevance_model_feedback()` is only ever called
  after `score_candidates()` for the same query in the same run. It is
  conformance-checked directly (see
  `tests/test_interface_conformance.py::test_relevance_model_feedback_does_not_require_prior_score_candidates_call`).

## Anti-gaming checks

Beyond call-order independence, the harness enforces:

- **No duplicate doc_ids** in a single returned list (see the earlier project's
  `harness/metrics.py` for why — nDCG@10/MAP@10 are unbounded above by
  construction).
- **Every returned doc_id must be in the candidate pool you were given**
  (`candidate_doc_ids` for `score_candidates()`; `candidate_doc_ids` again
  — not `pseudo_relevant_doc_ids` — for `relevance_model_feedback()`, since
  that is the pool being reranked, per the note above). Returning a
  doc_id from outside that pool, or from `pseudo_relevant_doc_ids` when it
  isn't also in `candidate_doc_ids`, is a conformance failure.

## Wall-clock budget

Because grading is in-process rather than containerised, there is no
per-team resource isolation from a submission that runs unreasonably
long. `prepare()` and each individual retrieval call are subject to a
generous wall-clock cap (stated precisely in `docs/GRADING.md`); exceeding
it is scored as a miss on that call, not a harness crash that blocks the
rest of the class's grading run.

## Continuous conformance checking

Every push to your submission repository triggers the public harness on
the toy set and reports pass/fail within minutes — use this early, this
project's timeline leaves little room to discover interface breakage
at the deadline.

## Conformance freeze

48 hours before the deadline, the CI check must be green. Interface
failures reported for the first time after the freeze are graded as a
submission error (spec Section 9), not treated as a harness bug.
