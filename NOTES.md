# Working notes

Not part of the submission. Delete before pushing if you want a clean tree.

## Where the score comes from

| Track | Weight | What drives it |
|---|---|---|
| A | 30% | `score_candidates()` raw nDCG@10 (24%) + MAP@10 (6%) |
| B | 25% | `relevance_model_feedback()` raw nDCG@10 at 0% noise |
| C | 35% | **retention ratio** = mean(noisy nDCG) / your own clean nDCG |
| bonus | +10% | round-robin attack + defense |

Track C being a *ratio* is the whole strategic shape of this assignment.
Turning expansion off entirely gives retention = 1.0 and the top Track C
percentile — `docs/GRADING.md` says so in as many words. So the only way
to beat that is expansion that genuinely does not move when the seed is
contaminated: gain on B without paying for it on C. Every robustness
device in `feedback.py` exists for that one trade.

## Toy numbers (20 docs, 6 queries — sanity only, not tuning signal)

```
score_candidates()        nDCG@10 0.9036   MAP@10 0.9024
feedback, clean           nDCG@10 0.9300
feedback, 25% noise       nDCG@10 0.9296
feedback, 50% noise       nDCG@10 0.9294
retention                 0.9995
```

Out to 100% contamination, and under the harder *adjacent* noise model in
`tools/sweep.py`, retention stays ≈0.98. Feedback beats plain QL on the
clean condition, which is the point.

Timing on a synthetic 171k-document / 170-avgdl corpus (trec-covid shape):
`prepare()` 12s against a 600s budget, 0.06 GB peak RSS, `score_candidates()`
0.5 ms/call, `relevance_model_feedback()` 1.7 ms/call against a 5s budget.
The byte-offset lazy document reader is what keeps memory flat.

## Two judgement calls you should make yourself

**1. `RANK_PRIOR_WEIGHT = 0.30`.** `score_candidates()` fuses the LM score
with a reciprocal-rank prior taken from the order of `candidate_doc_ids`.
That order is a real input, arrives best-first, and comes from a source
independent of our LM — textbook conditions for rank fusion to help. On
the toy set it is worth about +0.02 nDCG and +0.045 MAP.

The argument against: Track A is labelled "base LM accuracy", and a grader
could reasonably say part of that score is then the reference retriever's,
not your language model's. `docs/GRADING.md` warns against building around
the generator. I do not think a weight-0.30 z-score fusion crosses that
line, but it is your submission and your viva. Set `RANK_PRIOR_WEIGHT = 0.0`
for a pure LM ranking; nothing else changes.

**2. Adaptive lambda.** `_adaptive_lambda()` moves λ using coherence and
clarity computed *from the seed set alone*. It never caches
`score_candidates()` output and cannot tell a clean call from a perturbed
one, so it is on the right side of the call-order-independence rule — but
be ready to explain exactly why, because it is the part of the design
closest to the line. Pin it with `LAMBDA_FIXED = 0.55` if you would rather
not defend it.

## What to run once you have `data/full/`

```bash
python scripts/download_full_corpus.py
# then, with the staff-released candidates file in place:
python -m tools.sweep --corpus data/full/corpus.jsonl \
  --queries data/full/queries_dev.tsv --qrels data/full/qrels_dev.txt \
  --candidates data/full/candidates_dev.jsonl --ablations
python -m tools.sweep ... --grid mu
python -m tools.sweep ... --grid lambda
python -m tools.sweep ... --grid beta
python -m tools.sweep ... --ablations --noise-model adjacent
```

Order of likely payoff on the real corpus:

1. `MU_AVGDL_FACTOR` / the `MU_MIN`–`MU_MAX` clamp. Currently μ = 12·avgdl
   clamped to [250, 2500], which lands at ~2000 on a trec-covid-shaped
   collection. Worth confirming against the `--grid mu` sweep.
2. `LAMBDA_MIN` / `LAMBDA_MAX`. The single most important knob per the
   primer. Widen or narrow the band based on where B stops gaining and C
   starts losing.
3. `MAX_EXPANSION_TERMS` and `MAX_FEEDBACK_DOCS`. 60/10 are conventional;
   fewer terms is usually more robust, more terms usually better on clean.
4. `WEIGHT_BETA`. Higher = sharper concentration on the best seed docs =
   more robust, up to the point where the feedback set collapses to one
   document and B falls.
5. `USE_STEMMER`. Worth +0.04 nDCG and +0.18 MAP on toy. Re-check on the
   real corpus — if it is neutral there, consider turning it off, since a
   simpler tokeniser is easier to defend.

Do not tune on the toy set. `data/README.md` says this explicitly and it
is right: six queries over twenty documents will happily reward noise.

## Report (Section 8) — the evidence is already generated

- `--ablations` gives the per-device table, uniform and adjacent noise.
- `--grid lambda` gives the λ-vs-drift curve the primer asks you to
  discuss after reading Mitra, Singhal & Buckley (1998).
- `explain_feedback(query, seed, pool)` in `feedback.py` dumps the seed
  weights, coherence, clarity, chosen λ and the top expansion terms for a
  single query — use it for the worked example of drift being suppressed.
- `tools/make_adversarial.py` prints the mean surrogate drop per query,
  which is the evidence for the bonus-track write-up.
