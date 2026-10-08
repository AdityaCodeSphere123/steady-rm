# Assignment 2: Query Drift Arena

Rerank a provided top-K candidate list with your own unigram language
model, then build an RM1/RM2/RM3 relevance-feedback layer on top of it —
and survive the harness quietly feeding that feedback function a
noise-contaminated pseudo-relevant seed. See the assignment spec
(`assignment2_query_drift_arena.docx`/`.pdf` in the course materials) for
the full writeup; this README is quickstart + repo map only.

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -v
python -m harness.run_harness \
  --corpus data/toy/corpus.jsonl \
  --queries data/toy/queries_dev.tsv \
  --qrels data/toy/qrels_dev.txt \
  --candidates data/toy/candidates_dev.jsonl \
  --run-out runs/dev_run.trec \
  --report-out runs/dev_report.json
```

Or just run `bash scripts/smoke_test.sh`, which does all three.

## What you edit

Only `submission/feedback.py` (and, optionally,
`submission/adversarial_set.json` for the bonus track — see
`docs/GRADING.md`). Everything under `harness/` is read-only reference
code shared by every submission; it is what actually scores you, so
reading it (especially `harness/leaderboard.py` and
`harness/noise_injection.py`) is worth your time.

## Models and relevance models

The graded functions run the submitted model (`competition`: SPUD + lead
field + proximity, with RM3). The required baselines and RM1/RM2 are
selectable without editing code; the full list of calls is at the top of
`submission/feedback.py`.

| `FEEDBACK_MODEL` | document model |
|---|---|
| `competition` (default) | SPUD query likelihood + lead field + proximity |
| `dirichlet` | multinomial unigram QL, Dirichlet mu 500 |
| `jm` | multinomial unigram QL, Jelinek-Mercer 0.3 |

`FEEDBACK_RM` = 1 (RM1), 2 (RM2) or 3 (RM3, default); `FEEDBACK_LAMBDA`
overrides the RM3 query weight. Example, plain Dirichlet QL + RM1:

```bash
FEEDBACK_MODEL=dirichlet FEEDBACK_RM=1 python -m harness.run_harness \
  --corpus data/full/corpus.jsonl --queries data/full/queries_dev.tsv \
  --qrels data/full/qrels_dev.txt --candidates data/full/candidates_dev.jsonl
```

From Python: `fb.rerank(q, pool, 10, model="dirichlet")`,
`fb.rm_feedback(q, seed, pool, 10, model="dirichlet", rm=2)`,
`fb.relevance_model(q, seed, variant=1, model="jm")`, or
`fb.use_model("jm", rm=3, lam=0.7)` to switch the graded functions.

## Repo map

```
submission/
  feedback.py          <- YOUR CODE GOES HERE (the required entrypoint)
  cpp/                  C++ engine used by feedback.py (see README_A2_ENGINE.md)
  adversarial_set.json  <- optional, bonus track (see docs/GRADING.md)

harness/
  run_harness.py         runs your submission end-to-end, prints a local sanity report
  leaderboard.py          Track A/B/C + bonus percentile scoring (read this)
  noise_injection.py       PUBLIC practice noise recipe (NOT the grading recipe)
  candidates_io.py          reads the provided top-K candidate list
  metrics.py               nDCG@10 / MAP@10 / MRR / P@k, TREC-eval-compatible
  trec_io.py                corpus/queries/qrels/run file I/O

docs/
  SUBMISSION_INTERFACE.md   exact function signatures + conformance rules
  GRADING.md                 what's public vs. undisclosed, wall-clock budget
  RELEVANCE_FEEDBACK_PRIMER.md   compact RM1/RM2/RM3 formula reference

data/toy/       20-doc hand-built set + provided candidates for fast local iteration (see data/README.md)
scripts/        download_full_corpus.py, smoke_test.sh
tests/          the exact tests CI runs on every push
```

## Differences from Assignment 1, at a glance

- **You never build an index or rank the whole corpus.** Every retrieval
  call is scoped to a PROVIDED candidate pool (`candidate_doc_ids`,
  typically ~100 documents from an undisclosed reference retriever — see
  `docs/GRADING.md`) — not `data/toy/corpus.jsonl` or `data/full/corpus.jsonl`
  at large. `prepare()` still reads the whole corpus once for cheap,
  one-time collection-wide statistics; nothing after that is O(corpus)
  per query. This is deliberate: Assignment 1 already covered
  "efficiently rank a whole corpus" — this assignment is about the
  language model and the relevance-feedback layer, not indexing.
- No Docker image, no build/load process split — `prepare()` and both
  retrieval functions run in-process, called directly by
  `run_harness.py`. See `docs/SUBMISSION_INTERFACE.md`.
- Two required retrieval functions instead of one (`score_candidates()`
  and `relevance_model_feedback()`), scored on three tracks instead of a
  single accuracy metric plus efficiency/size modifiers — see the
  assignment spec, Section 7.
- A public, unit-tested "practice" noise-injection recipe you can sweep
  locally (`harness/noise_injection.py`) — the real grading recipe is
  undisclosed, same secrecy principle as Assignment 1's held-out topics.
- An optional bonus track (`submission/adversarial_set.json`) that is
  genuinely head-to-head against the rest of the class, not just a shared
  leaderboard climb.
