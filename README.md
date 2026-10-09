# Assignment 2: Query Drift Arena - Aditya Anand (2023CS50284)

Rerank a provided candidate pool with a query-likelihood model, then apply
relevance-model feedback (RM1, RM2, RM3). The scoring code is a small C++
engine in `submission/cpp/`, loaded by `submission/feedback.py` with `ctypes`.
Only the Python and C++ standard libraries are used. The engine compiles
itself on first import into `submission/_build/`; you need g++ or clang++.

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

## What is implemented

- **Unigram query likelihood** (`score_candidates`) with Dirichlet, Jelinek-Mercer and SPUD smoothing. The smoothing parameter is tunable. Only the given candidate pool is scored and returned.
- **RM1**: P(w|R) = Σ_D P(w|D) P(D|Q), with P(D|Q) ∝ P(Q|D).
- **RM2**: P(w|R) ∝ P(w) Π_q Σ_D P(q|D) P(D|w).
- **RM3**: RM1 interpolated with the original query, weight λ on the query. λ = 1 returns exactly `score_candidates`.
- **Extensions** used only by the competition model: a lead-field model over the first 20 tokens, ordered and unordered proximity, and feedback controls that sharpen P(D|Q), drop seed documents that barely match the query, and mix the expansion into the Track A score.

The pseudo-relevant ids are used only to estimate P(w|R). They need not be in the pool. Unknown ids are ignored. With no usable seed, the function returns the Track A ranking.

## Models

Three parameter sets live in `MODELS` in `submission/feedback.py`. Each works with RM1, RM2 and RM3.

| Name | Document model | Default feedback |
|---|---|---|
| `competition` (submitted) | SPUD (spud_mu 800) + lead field + proximity | RM3: 0.9 · S_A + 0.1 · expansion, top 5 seed docs, 80 terms |
| `dirichlet` | plain unigram QL, Dirichlet mu = 360 | RM3: λ 0.45, top 2 seed docs, 30 terms |
| `jm` | plain unigram QL, Jelinek-Mercer λ = 0.35 | RM3: λ 0.3, top 7 seed docs, 100 terms |

The two baselines have no SPUD, lead field or proximity, and they use the textbook P(D|Q). They were tuned for nDCG@10 on `data/full` (TREC-COVID).

The graded functions read `FEEDBACK_MODEL`, `FEEDBACK_RM` (1, 2 or 3) and `FEEDBACK_LAMBDA` at import. Unset, they run the submitted model. In zsh, set the harness arguments as an array so they are passed separately:

```bash
args=(--corpus data/full/corpus.jsonl --queries data/full/queries_dev.tsv --qrels data/full/qrels_dev.txt --candidates data/full/candidates_dev.jsonl)

python -m harness.run_harness $args
FEEDBACK_RM=1 python -m harness.run_harness $args
FEEDBACK_RM=2 python -m harness.run_harness $args
FEEDBACK_MODEL=dirichlet FEEDBACK_RM=1 python -m harness.run_harness $args
FEEDBACK_MODEL=dirichlet FEEDBACK_RM=2 python -m harness.run_harness $args
FEEDBACK_MODEL=dirichlet FEEDBACK_RM=3 python -m harness.run_harness $args
FEEDBACK_MODEL=jm FEEDBACK_RM=3 FEEDBACK_LAMBDA=0.7 python -m harness.run_harness $args
```

From Python, after `fb.prepare(...)`:

```python
fb.rerank(q, pool, 10, model="dirichlet")
fb.rm_feedback(q, seed, pool, 10, model="dirichlet", rm=2)
fb.relevance_model(q, seed, variant=1, model="jm")
fb.use_model("jm", rm=3, lam=0.7)   # graded functions use this until use_model()
```

## Checking correctness by hand

```bash
python -m tools.verify_rm
python -m tools.verify_rm --query "apple juice" --seed d2 d3 --lam 0.7 --mu 5
pytest -q tests
python -m tests.test_engine
```

`verify_rm` uses a five-document corpus and prints the engine next to a direct calculation of Dirichlet log P(Q|D), P(D|Q), the RM1 and RM2 term distributions, and the RM1, RM2 and RM3 ranking scores. The rows match to about 1e-12. The test suite also checks pool-only ids, no duplicates, k, an empty or unknown seed, and λ = 1 equalling Track A.

## Results on the full dev set

nDCG@10 from `harness/run_harness.py` on `data/full`. The noise columns use the harness's public practice noise.

| Model | Track A | RM | Clean | 25% noise | 50% noise |
|---|---|---|---|---|---|
| competition | 0.6521 | RM1 | 0.6703 | 0.6592 | 0.6327 |
| | | RM2 | 0.6492 | 0.6344 | 0.6266 |
| | | RM3 (submitted) | 0.6534 | 0.6533 | 0.6546 |
| dirichlet | 0.6076 | RM1 | 0.6262 | 0.6139 | 0.6131 |
| | | RM2 | 0.6166 | 0.6154 | 0.5943 |
| | | RM3 | 0.6582 | 0.6473 | 0.6407 |
| jm | 0.5434 | RM1 | 0.5925 | 0.5823 | 0.5669 |
| | | RM2 | 0.5505 | 0.5487 | 0.5351 |
| | | RM3 | 0.6030 | 0.5925 | 0.5890 |

λ is the weight on the original query. 0 is pure expansion, 1 is no feedback.

| λ | competition clean | competition 50% noise | dirichlet clean | dirichlet 50% noise |
|---|---|---|---|---|
| 0.0 | 0.6703 | 0.6327 | 0.6262 | 0.6131 |
| 0.2 | 0.6457 | 0.6430 | 0.6411 | 0.6258 |
| 0.4 | 0.6489 | 0.6497 | 0.6526 | 0.6368 |
| 0.6 | 0.6524 | 0.6528 | 0.6398 | 0.6244 |
| 0.8 | 0.6535 | 0.6548 | 0.6240 | 0.6185 |
| 0.9 (submitted) | 0.6534 | 0.6546 | 0.6150 | 0.6139 |
| 1.0 | 0.6521 | 0.6521 | 0.6076 | 0.6076 |

Dirichlet beats Jelinek-Mercer by about 0.06 on Track A. SPUD plus the lead field and proximity adds about 0.045 over Dirichlet. RM3 helps the plain baselines by 0.05 to 0.06, and the competition model by about 0.001, because that first stage is already strong. Pure RM1 has the highest clean score on the competition model (0.6703) and loses about 0.04 at 50% noise. RM2 is below RM1 in every model. λ 0.9 is the submitted setting because the score at 50% noise stays level with the clean score.

Hidden leaderboard, best submitted run:

| Track A nDCG | Track A MAP | Track B nDCG | C1 retention | C2 retention |
|---|---|---|---|---|
| 0.53809 | 0.46063 | 0.53814 | 1.00000 | 0.99854 |

## Adversarial set

`submission/adversarial_set.json` lists dev queries 48, 29, 44, 25 and 22, each with documents from that query's own TREC-COVID candidate pool. None is judged relevant. They score high on query likelihood, so a feedback function that trusts its seed promotes them into the top 10.

The lists were built by adding documents one at a time, then swapping or dropping one while that raised the average nDCG@10 drop. The panel was 10 feedback setups (the submitted model with RM3 and RM1, the tuned baselines, typical RM1/RM2/RM3, uniform P(D), and λ 0.7), each run with Porter stemming and with raw tokens. The simulated drop averages about 0.57 on a separate group of strong RM3 systems. The submitted model itself does not move: its top 10 is unchanged on all five queries.

## Files

- `submission/feedback.py`: entry points, `MODELS`, and loading the engine.
- `submission/cpp/rm_engine.h`: the parameters.
- `submission/cpp/rm_engine.cpp`: indexing, the document models, RM1, RM2, RM3, scoring.
- `submission/cpp/analysis.cpp`: tokeniser, stopwords, Porter stemmer.
- `submission/cpp/c_api.cpp`: the C interface used by `ctypes`.
- `submission/cpp/util.h`: the string table, thread pool, and JSONL reader.

`RM_THREADS=n` caps the engine at 4 threads.
