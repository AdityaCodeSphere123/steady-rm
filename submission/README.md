# COL764 Assignment 2 – Aditya Anand

This folder is my submission: a QL reranker (Track A) and
relevance-model feedback (RM1, RM2, RM3) for Tracks B and C.The scoring code is a small C++ engine in `cpp/`, and`feedback.py` loads it with `ctypes`. Only the Python standard library and the C++ standard library are used, so it runs offline. The engine compiles itself on first import into `submission/_build/`; you need g++ or clang++, and nothing
has to be installed.

## What is implemented

- **Unigram query likelihood** (`score_candidates`) with three smoothing
  choices: Dirichlet, Jelinek-Mercer and SPUD. The smoothing parameter is tunable. Only the given candidate pool is scored and returned.
- **RM1**: P(w|R) = Σ_D P(w|D) P(D|Q) over the pseudo-relevant documents, with
  P(D|Q) ∝ P(Q|D).
- **RM2**: Lavrenko and Croft's conditional-sampling estimate,
  P(w|R) ∝ P(w) Π_q Σ_D P(q|D) P(D|w).
- **RM3**: RM1 interpolated with the original query, using weight λ on the
  query side. With λ = 1 it returns exactly what `score_candidates` returns.
- **Extensions** on top of the unigram model, used only by my competition
  model:
  - a lead-field model over the first 20 tokens of each document;
  - sequential-dependence proximity: ordered bigrams and unordered windows;
  - a few robustness controls on the feedback side: sharpening P(D|Q),
    dropping seed documents that barely match the query, and mixing the
    expansion score into the Track A score.

The pseudo-relevant ids are used only to estimate P(w|R). They don't have to
be in the pool, and unknown ids are ignored. If no usable seed document is
left, the function returns the Track A ranking unchanged.

## Three models, one switch

There are three ready-made parameter sets in `MODELS` (in `feedback.py`).
Each can be combined with RM1, RM2 or RM3.

| Name | Document model | Default feedback |
|---|---|---|
| `competition` (submitted) | SPUD (spud_mu 800) + lead field + proximity | RM3: 0.9 · S_A + 0.1 · expansion, top 5 seed docs, 80 terms |
| `dirichlet` (required baseline) | plain multinomial unigram QL, Dirichlet mu = 360 | RM3: λ 0.45, top 2 seed docs, 30 terms |
| `jm` | plain multinomial unigram QL, Jelinek-Mercer λ = 0.35 | RM3: λ 0.3, top 7 seed docs, 100 terms |

The two baselines contain no SPUD, lead field or proximity, and they use the
plain textbook P(D|Q). Their parameters were tuned for nDCG@10 on `data/full`.

### From the harness (no code changes)

The graded functions read three environment variables at import. If none is
set, you get the submitted model.

```bash
D=data/full
ARGS="--corpus $D/corpus.jsonl --queries $D/queries_dev.tsv --qrels $D/qrels_dev.txt --candidates $D/candidates_dev.jsonl"

python -m harness.run_harness $ARGS                                     # submitted: competition + RM3
FEEDBACK_RM=1 python -m harness.run_harness $ARGS                       # competition + RM1
FEEDBACK_RM=2 python -m harness.run_harness $ARGS                       # competition + RM2
FEEDBACK_MODEL=dirichlet FEEDBACK_RM=1 python -m harness.run_harness $ARGS   # Dirichlet QL + RM1
FEEDBACK_MODEL=dirichlet FEEDBACK_RM=2 python -m harness.run_harness $ARGS   # Dirichlet QL + RM2
FEEDBACK_MODEL=dirichlet FEEDBACK_RM=3 python -m harness.run_harness $ARGS   # Dirichlet QL + RM3
FEEDBACK_MODEL=jm FEEDBACK_RM=3 FEEDBACK_LAMBDA=0.7 python -m harness.run_harness $ARGS   # JM + RM3, λ = 0.7
```

`FEEDBACK_MODEL` is one of `competition`, `dirichlet` or `jm`. `FEEDBACK_RM`
is 1, 2 or 3, and `FEEDBACK_LAMBDA` overrides λ for RM3.

### From Python

```python
from submission import feedback as fb
fb.prepare("data/full/corpus.jsonl")

fb.score_candidates(q, pool, 10)                    # graded Track A (active model)
fb.relevance_model_feedback(q, seed, pool, 10)      # graded Track B/C (active model)

fb.rerank(q, pool, 10, model="dirichlet")           # plain Dirichlet QL
fb.rm_feedback(q, seed, pool, 10, rm=1)             # competition + RM1
fb.rm_feedback(q, seed, pool, 10, model="dirichlet", rm=2)            # Dirichlet + RM2
fb.rm_feedback(q, seed, pool, 10, model="jm", rm=3, lam=0.7)          # JM + RM3, λ = 0.7
fb.rm_feedback(q, seed, pool, 10, model="dirichlet", rm=3, mu=1000)   # any parameter can be overridden

fb.relevance_model(q, seed, variant=1, model="dirichlet")  # P(w|R) itself as [(term, p)]; variant 2 = RM2

fb.use_model("dirichlet", rm=1)   # make the graded functions use Dirichlet + RM1
fb.use_model()                    # back to the submitted model
```

`rerank` and `rm_feedback` switch parameters only for that one call; afterwards
the active model is restored. All three models share the same `prepare()` (same
tokeniser and stemmer), so you can switch between them freely.

## Checking correctness by hand

```bash
python -m tools.verify_rm
python -m tools.verify_rm --query "apple juice" --seed d2 d3 --lam 0.7 --mu 5
```

This uses a five-document toy corpus. For each step it prints my engine's value
next to a direct Python calculation of the textbook formula:
- the Dirichlet log P(Q|D);
- P(D|Q);
- the RM1 and RM2 term distributions;
- the RM1, RM2 and RM3 ranking scores.

Every row matches to about 1e-12. The same checks, plus the conformance rules
(only pool ids, no duplicates, k respected, empty or unknown seed, λ = 1 equals
Track A), run in the test suite:

```bash
pytest -q tests
python -m tests.test_engine
```

## Results on the full dev set

All numbers are nDCG@10 from `harness/run_harness.py` on `data/full` (the
CORD-19 / TREC-COVID dev queries). The noise columns are the harness's public
practice noise.

### Every model with every relevance model

| Model | Track A | RM | Clean | 25% noise | 50% noise |
|---|---|---|---|---|---|
| competition | 0.6521 | RM1 | 0.6703 | 0.6592 | 0.6327 |
| | | RM2 | 0.6492 | 0.6344 | 0.6266 |
| | | **RM3 (submitted)** | **0.6534** | **0.6533** | **0.6546** |
| dirichlet | 0.6076 | RM1 | 0.6262 | 0.6139 | 0.6131 |
| | | RM2 | 0.6166 | 0.6154 | 0.5943 |
| | | RM3 | 0.6582 | 0.6473 | 0.6407 |
| jm | 0.5434 | RM1 | 0.5925 | 0.5823 | 0.5669 |
| | | RM2 | 0.5505 | 0.5487 | 0.5351 |
| | | RM3 | 0.6030 | 0.5925 | 0.5890 |

### Effect of λ in RM3

λ is the weight on the original query: 0 means pure expansion, and 1 means no
feedback.

| λ | competition clean | competition 50% noise | dirichlet clean | dirichlet 50% noise |
|---|---|---|---|---|
| 0.0 | 0.6703 | 0.6327 | 0.6262 | 0.6131 |
| 0.2 | 0.6457 | 0.6430 | 0.6411 | 0.6258 |
| 0.4 | 0.6489 | 0.6497 | 0.6526 | 0.6368 |
| 0.6 | 0.6524 | 0.6528 | 0.6398 | 0.6244 |
| 0.8 | 0.6535 | 0.6548 | 0.6240 | 0.6185 |
| 0.9 (submitted) | 0.6534 | 0.6546 | 0.6150 | 0.6139 |
| 1.0 | 0.6521 | 0.6521 | 0.6076 | 0.6076 |

### What I take from this

- **Smoothing.** Dirichlet beats Jelinek-Mercer by about 0.06 on Track A.
  Documents in this corpus vary a lot in length, and Dirichlet smoothing
  adapts to document length while JM doesn't. SPUD plus the lead field and
  proximity adds another 0.045 over Dirichlet.
- **Feedback gains depend on the first stage.** On the plain baselines RM3
  helps a lot (+0.05 for Dirichlet, +0.06 for JM). On the competition model
  the gain is small, because the first-stage ranking is already good and
  there is less left to fix.
- **Pure RM1 is the most fragile.** It has no query term at all. On the
  competition model it gives the best clean score (0.6703) but loses about
  0.04 at 50% noise, because a noisy seed pulls the expansion terms off
  topic. This is query drift.
- **RM2 scores below RM1 in every model.** Here it tends to put too much
  weight on terms that occur in only one seed document.
- **Why λ is high in the submission.** At λ 0.8–0.9 the competition model is
  as good with 50% noise as without it, so its retention ratio stays at
  1.0. That matters more for Tracks B and C than the extra clean score from
  a lower λ, because Track C is 35% of the grade and the hidden seeds are
  perturbed.

### Hidden leaderboard (best submitted run)

| Track A nDCG | Track A MAP | Track B nDCG | C1 retention | C2 retention |
|---|---|---|---|---|
| 0.53809 | 0.46063 | 0.53814 | 1.00000 | 0.99854 |

## Files

- `feedback.py`: entry points, the `MODELS` table, and the build and loading
  of the C++ engine.
- `cpp/rm_engine.h`: every parameter, each with a one-line description.
- `cpp/rm_engine.cpp`: indexing, the document models, RM1, RM2, RM3 and the
  scoring.
- `cpp/analysis.cpp`: the tokeniser, stopword list and Porter stemmer.
- `cpp/c_api.cpp`: the C interface used by `ctypes`.
- `cpp/util.h`: a small thread pool and helpers.

`RM_THREADS=n` limits the number of threads the engine uses (at most 4).
