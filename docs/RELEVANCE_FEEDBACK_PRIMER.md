# Relevance Feedback Primer

A compact reference for the math behind `submission/feedback.py`. This is
a supplement to lecture, not a replacement for it — if your instructor
presented RM1/RM2 with different notation or a different derivation, use
theirs; the grading interface is behavioural (it only cares what
`relevance_model_feedback()` returns, not which formula you used to get
there).

## Query likelihood with smoothing

Rank documents by `P(Q|D) = ∏_{w∈Q} P(w|D)` (in practice, sum of
log-probabilities, to avoid underflow). The maximum-likelihood estimate
`c(w,D)/|D|` assigns zero probability to any query term absent from `D`,
so every practical implementation smooths against a collection model
`P(w|C) = collection_count(w) / collection_length`:

- **Jelinek-Mercer:** `P(w|D) = (1-λ)·c(w,D)/|D| + λ·P(w|C)`
- **Dirichlet prior:** `P(w|D) = (c(w,D) + μ·P(w|C)) / (|D| + μ)`

Both are implemented in `submission/lm_utils.py`
(`jelinek_mercer_smoothed_log_prob` / `dirichlet_smoothed_log_prob`); the
shipped baseline uses Dirichlet with `μ = 1500`. Neither value is
"correct" — tuning them, and understanding *why* your choice behaves the
way it does on this corpus, is part of the project (Section 8).

## Relevance models (RM1 / RM2 / RM3)

Given a pseudo-relevant set `F = {D_1, ..., D_k}` (here: the
`pseudo_relevant_doc_ids` argument), estimate a distribution over the
vocabulary approximating "the language of documents relevant to this
query", **without using any relevance judgments** — only the documents
themselves and the original query:

```
P(w | R) ≈ Σ_{D ∈ F} P(w | D) · P(D | Q)
```

where `P(D | Q)` is `D`'s (normalized) query-likelihood score under the
*original* query — documents that already matched the query well get
more influence over what the relevance model looks like. This is the core
of what's usually called **RM1**: used directly to rank documents (e.g.
via KL-divergence between `P(·|R)` and each candidate document's own
language model — lower divergence ranks higher).

**RM2** is commonly presented as an alternate derivation of the same
quantity under a different conditional-independence assumption between
query terms and the sampled document; some courses skip straight from RM1
to RM3 without separately implementing RM2. If your lecture covered a
specific RM2 formulation, use it — there is no single canonical form
enforced by this project.

**RM3** — the version that matters most in practice — linearly
interpolates the relevance model back with the original query's own
(unsmoothed or lightly-smoothed) model:

```
P_RM3(w) = λ · P(w | Q) + (1 - λ) · P(w | R)
```

`λ` is the single most important knob in this project. `λ = 1` ignores
feedback entirely (identical, in effect, to the shipped trivial baseline).
`λ = 0` trusts the relevance model completely, with no anchor back to
what the user actually typed — maximally powerful when the pseudo-relevant
set is clean, and maximally exposed to query drift when it isn't. Somewhere
between is where a robust submission lives; finding where, and explaining
why, is most of what Section 8's report asks for.

## Suggested implementation shape

```python
def relevance_model_feedback(query, pseudo_relevant_doc_ids, candidate_doc_ids, k=10):
    # Estimate P(w|R) from the (possibly noise-perturbed) SEED set only --
    # this is the input the harness is actually testing your robustness against.
    relevance_model = estimate_relevance_model(query, pseudo_relevant_doc_ids, _STATS)
    interpolated = interpolate_with_query_model(relevance_model, query, lam=YOUR_LAMBDA)
    # Rerank the full CANDIDATE POOL (not just the seed) using the estimated
    # model -- the seed only ever informs what P_RM3 looks like; the pool
    # being ranked is candidate_doc_ids, same as score_candidates() saw.
    return rank_by_relevance_model(interpolated, candidate_doc_ids, _STATS, k)
```

None of `estimate_relevance_model`, `interpolate_with_query_model`, or
`rank_by_relevance_model` are provided — implementing them, correctly and
robustly, is the project. Note the two different doc-id lists doing
two different jobs: `pseudo_relevant_doc_ids` only ever feeds the
*estimation* step; `candidate_doc_ids` is the pool your function actually
reranks and returns from (see `docs/SUBMISSION_INTERFACE.md`).

## Optional: beyond unigram

Everything above is unigram (term counts, order-independent), and that is
what's required — it's what `docs/GRADING.md`'s "Correctness of required
components" line checks, and what makes every submission comparable on
the same known small examples. Nothing about the harness enforces it,
though: `score_candidates()` and `relevance_model_feedback()` are graded
purely on the ranked list they return (see `docs/SUBMISSION_INTERFACE.md`)
— never on what statistics you computed to get there. If you want to
extend a *correctly-implemented* unigram baseline with a bigram or
positional component for competitive edge (spec Section 4.1), go
ahead; `lm_utils.py` is optional convenience, not a contract, so nothing
stops you from building your own n-gram statistics alongside or instead
of `CollectionStats`.

One caution: don't assume this makes you *more* robust to noise by
default. A pseudo-relevant set has, at most, a handful of documents —
bigram co-occurrence counts within that set are sparser than unigram
counts over the same documents, and sparser statistics can be *more*
sensitive to a contaminated document sneaking in, not less. If you go
this route, measure it (Section 8's report asks for exactly this kind of
evidence) rather than assuming the more sophisticated model wins.

## Further reading

- Victor Lavrenko and W. Bruce Croft. Relevance Based Language Models.
  *SIGIR 2001*, pp. 120–127. (The original RM1-family construction.)
- Chengxiang Zhai and John Lafferty. A Study of Smoothing Methods for
  Language Models Applied to Ad Hoc Information Retrieval. *SIGIR 2001*,
  pp. 334–342. (Dirichlet vs. Jelinek-Mercer, and when each tends to win.)
- Mandar Mitra, Amit Singhal, and Chris Buckley. Improving Automatic Query
  Expansion. *SIGIR 1998*, pp. 206–214. (The original query-drift
  discussion — read this before you tune λ.)
