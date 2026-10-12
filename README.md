# steady-rm

Relevance-model feedback that holds steady under query drift.

Rerank a given candidate pool with a unigram language model, then expand the query with a relevance model estimated from a pseudo-relevant seed. The submitted system is built so that a clean seed can still help, while a seed that has been quietly filled with off-topic documents does not drag the ranking away from the original query.

The longer write-up, with the λ curves, is in [`report/report.pdf`](report/report.pdf).

**Author:** Aditya Anand

## Setup

You need Python 3.12 or newer, and `g++` or `clang++`. The C++ engine compiles itself the first time it is imported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead of `source`.

Then run the tests and the harness on the toy set that ships with the repo:

```bash
pytest tests/ -v
bash scripts/smoke_test.sh
```

`scripts/smoke_test.sh` runs the conformance tests and writes `runs/dev_run.trec` and `runs/dev_report.json` for `data/toy/` (20 documents, 6 queries). Those toy scores catch interface bugs. They are not the experiment numbers later in this file.

Leave the virtualenv with `deactivate`.

## The problem

Pseudo-relevance feedback assumes the top documents for a query are relevant and mines expansion terms from them. When some of those documents are not relevant, the expansion moves toward the wrong topic. That failure is query drift (Mitra, Singhal, and Buckley, 1998).

Here the retriever never searches the whole collection per query. A first-pass system has already proposed a candidate pool, typically about 100 documents. Two functions rerank that pool:

- `score_candidates` ranks by query likelihood. This is Track A.
- `relevance_model_feedback` estimates P(w | R) from a list of pseudo-relevant document ids and reranks the same pool. This is Track B when the seed is clean, and Track C when the seed has been perturbed.

The seed is an argument. It is not recovered by caching the Track A ranking. Unknown ids are ignored. If nothing usable remains, the function returns the Track A ranking. Returned ids always come from the pool that was passed in.

## What was submitted

The graded run is one model for every track: SPUD query likelihood, a light lead-field and proximity model, and RM3 with λ = 0.9 on the original score. Feedback is a tenth of the final score. On the public TREC-COVID dev set that raises nDCG@10 from 0.6521 to 0.6534, and the score stays flat when half the seed is replaced. On the hidden leaderboard, Track B sits 0.00005 above Track A.

| Piece | Setting |
|---|---|
| Document model | SPUD, ω = 800, document-frequency background |
| Lead field | first 20 tokens, weight 0.1, Dirichlet μ = 60 |
| Proximity | ordered bigrams 0.05, unordered pairs in a window of 9 at weight 0.1, μ = 450 |
| Relevance model | Dirichlet μ = 100 inside the estimate only, at most 5 seed documents, 80 terms |
| Seed filter | drop a document covering under a quarter of the query; sharpen P(D \| Q) with β = 40; keep a term only if it occurs in at least a fifth of the seed |
| RM3 | λ = 0.9 on the Track A score, expansion centered over the pool |

Two plain baselines live next to it in `MODELS` in `submission/feedback.py`: Dirichlet (μ = 360) and Jelinek–Mercer (λ_JM = 0.35). They have no SPUD, lead field, or proximity, and they use the textbook interpolation. All three models can run RM1, RM2, or RM3.

## Models

Tokens are lowercase alphanumeric words. Dotted single letters are joined into acronyms. Query stopwords (the NLTK English list) are removed and tokens are Porter-stemmed.

Three unigram document models:

```
P_Dir(w|D)  = (tf(w,D) + μ P(w|C)) / (|D| + μ)
P_JM(w|D)   = (1 − λ_JM) tf(w,D)/|D| + λ_JM P(w|C)
P_SPUD(w|D) = (|V_D| · tf(w,D)/|D| + ω P_df(w)) / (|V_D| + ω)
```

P(w | C) is the collection frequency. P_df is the document frequency, normalised over the vocabulary. |V_D| is the number of distinct terms in D. Dirichlet and SPUD both grow the smoothing mass with a document property (length, or vocabulary size). Jelinek–Mercer uses one mixture weight for every document.

RM1 and RM2 follow Lavrenko and Croft (SIGIR 2001). Inside the estimate, P(w | D) is Dirichlet with μ = 100. Candidates are still scored with the Track A model.

```
P_RM1(w|R) = Σ_{D in seed} P(w|D) P(D|Q)
P_RM2(w|R) ∝ P(w) Π_{q in Q} Σ_D P(q|D) P(D|w)
```

P(D | Q) is proportional to P(Q | D). On the submitted model that posterior is sharpened with β = 40, and a seed document that covers less than a quarter of the query is dropped. Inverse document frequency only decides which non-query terms survive the support cutoff. The distribution is cut to 80 terms and renormalised.

RM1 ranks by Σ_w P(w | R) log P(w | D). RM2 uses the same score with P_RM2. RM3 puts weight λ back on the query. λ = 0 is pure RM1. λ = 1 returns the Track A ranking and the Track A scores.

The baselines use the usual mixture of query and expansion. The submitted model keeps the Track A score and centers the expansion over the pool, so the two scores can be added:

```
S(D) = λ S_A(D) + (1 − λ) (E(D) − mean E) / 12
E(D) = Σ_w P_RM1(w|R) log P_SPUD(w|D)
```

## How the code is split

`submission/feedback.py` is the entry point. On import it compiles `submission/cpp/` with g++ or clang++ into `submission/_build/` and loads the shared library with `ctypes`. Scoring uses the C++ standard library only. `RM_THREADS` caps the engine; the default is 4.

| File | Role |
|---|---|
| `submission/feedback.py` | `prepare`, `score_candidates`, `relevance_model_feedback`, and the named models |
| `submission/cpp/rm_engine.cpp` | Index, document models, RM1, RM2, RM3, scoring |
| `submission/cpp/analysis.cpp` | Tokeniser, stopwords, Porter stemmer |
| `submission/cpp/c_api.cpp` | C interface used by ctypes |
| `submission/cpp/util.h` | String table, thread pool, JSONL reader |
| `tools/verify_rm.py` | Hand calculation next to the engine on a five-document corpus |
| `harness/run_harness.py` | End-to-end run: Track A, clean feedback, and practice noise |
| `report.pdf` | Full report |

## Run other models

With the virtualenv from [Setup](#setup) still active, the graded functions read `FEEDBACK_MODEL`, `FEEDBACK_RM` (1, 2, or 3), and `FEEDBACK_LAMBDA` at import. Unset, they run the submitted model. In zsh, pass the harness arguments as an array:

```bash
args=(--corpus data/full/corpus.jsonl \
      --queries data/full/queries_dev.tsv \
      --qrels data/full/qrels_dev.txt \
      --candidates data/full/candidates_dev.jsonl)

python -m harness.run_harness "${args[@]}" \
  --run-out runs/dev_run.trec --report-out runs/dev_report.json

FEEDBACK_RM=1 python -m harness.run_harness "${args[@]}"
FEEDBACK_MODEL=dirichlet FEEDBACK_RM=3 python -m harness.run_harness "${args[@]}"
FEEDBACK_MODEL=jm FEEDBACK_RM=3 FEEDBACK_LAMBDA=0.7 python -m harness.run_harness "${args[@]}"
```

From Python, after `prepare`:

```python
import submission.feedback as fb
fb.prepare("data/toy/corpus.jsonl")
fb.rerank(q, pool, 10, model="dirichlet")
fb.rm_feedback(q, seed, pool, 10, model="dirichlet", rm=2)
fb.relevance_model(q, seed, variant=1, model="jm")
fb.use_model("jm", rm=3, lam=0.7)  # graded functions use this until the next use_model()
```

`data/full/` is the TREC-COVID collection (`beir/trec-covid` via `ir_datasets`). It is not committed.

```bash
python scripts/download_full_corpus.py
```

`ir_datasets` is already installed by `requirements.txt`.

That writes the corpus, queries, and qrels. The candidate pools used for the tables below were the ones released with the original task. `download_full_corpus.py` does not rebuild them.

## Checking the math

```bash
python -m tools.verify_rm
python -m tools.verify_rm --query "apple juice" --seed d2 d3 --lam 0.7 --mu 5
```

`verify_rm` prints the engine beside a direct calculation of Dirichlet log P(Q | D), P(D | Q), the RM1 and RM2 term distributions, and the RM1, RM2, and RM3 ranking scores. The rows match to about 1e-12. The tests also check that returned ids stay inside the pool, that there are no duplicates, that k is respected, that an empty or unknown seed falls back to Track A, and that λ = 1 reproduces `score_candidates`.

## Experiments

Dev numbers are nDCG@10 from `harness/run_harness.py`. Unless a table says otherwise, the collection is the released TREC-COVID dev set (50 queries) and the noise columns are one deterministic draw per query from the harness practice recipe (25% and 50% of the seed replaced). Retention is the 50% score divided by that row's clean score.

### Track A, before any feedback

| Document model | nDCG@10 |
|---|---|
| Jelinek–Mercer, λ_JM = 0.35 | 0.5434 |
| Dirichlet, μ = 360 | 0.6076 |
| SPUD ω = 800, lead field, proximity | 0.6521 |

Document lengths vary a lot, and Dirichlet scales the smoothing mass with length, which is why it is about 0.06 above Jelinek–Mercer. SPUD keeps a length-related term (the document's own vocabulary size) and uses a document-frequency background. The lead weight is 0.1 and the ordered-bigram weight is 0.05, so neither can override the unigram score. Together they add about 0.045 over plain Dirichlet.

### Feedback under a clean seed and under noise

| Model | Track A | Feedback | Clean | 25% | 50% | Retention at 50% |
|---|---|---|---|---|---|---|
| Competition | 0.6521 | RM1 | 0.6703 | 0.6592 | 0.6327 | 0.944 |
| | | RM2 | 0.6492 | 0.6344 | 0.6266 | 0.965 |
| | | RM3, λ = 0.9 | **0.6534** | **0.6533** | **0.6546** | **1.002** |
| Dirichlet, μ = 360, 2 docs, 30 terms | 0.6076 | RM1 | 0.6262 | 0.6139 | 0.6131 | 0.979 |
| | | RM2 | 0.6166 | 0.6154 | 0.5943 | 0.964 |
| | | RM3, λ = 0.45 | 0.6582 | 0.6473 | 0.6407 | 0.973 |
| Jelinek–Mercer, λ_JM = 0.35, 7 docs, 100 terms | 0.5434 | RM1 | 0.5925 | 0.5823 | 0.5669 | 0.957 |
| | | RM2 | 0.5505 | 0.5487 | 0.5351 | 0.972 |
| | | RM3, λ = 0.3 | 0.6030 | 0.5925 | 0.5890 | 0.977 |

RM1 is the best clean score on the competition model (0.6703) and the one that loses the most when the seed is corrupted (0.6327, retention 0.944). RM2 sits below RM1 on every model. Its retention is still higher than RM1 on the competition model (0.965 vs 0.944) and on Jelinek–Mercer (0.972 vs 0.957), so the gap is the estimate itself. On the competition model, RM3 is the only row above Track A at 0%, 25%, and 50% noise. Dirichlet RM3 gains more in absolute nDCG (0.6076 to 0.6582) because that first stage still has more to fix. On the submitted model the clean move is 0.0013.

### λ, the weight on the original query

λ = 0 is pure RM1. λ = 1 is Track A. The same noise draw is used at every λ.

| λ | Competition, clean | Competition, 50% | Dirichlet, clean | Dirichlet, 50% |
|---|---|---|---|---|
| 0.0 | 0.6703 | 0.6327 | 0.6262 | 0.6131 |
| 0.1 | 0.6442 | 0.6391 | 0.6321 | 0.6252 |
| 0.2 | 0.6457 | 0.6430 | 0.6411 | 0.6258 |
| 0.3 | 0.6500 | 0.6493 | 0.6439 | 0.6274 |
| 0.4 | 0.6489 | 0.6497 | 0.6526 | 0.6368 |
| 0.5 | 0.6496 | 0.6498 | 0.6521 | 0.6347 |
| 0.6 | 0.6524 | 0.6528 | 0.6398 | 0.6244 |
| 0.7 | 0.6536 | 0.6549 | 0.6262 | 0.6242 |
| 0.8 | 0.6535 | 0.6548 | 0.6240 | 0.6185 |
| **0.9** | **0.6534** | **0.6546** | 0.6150 | 0.6139 |
| 1.0 | 0.6521 | 0.6521 | 0.6076 | 0.6076 |

Below about λ = 0.4 the clean competition curve is above the noisy one: expansion helps a clean seed and hurts a noisy one. From 0.6 to 1.0 the two competition curves stay within 0.003. On this grid the flat part peaks at λ = 0.7 (0.6536 clean, 0.6549 at 50%), which is 0.0002 above λ = 0.9. Dirichlet's 50% curve peaks near 0.4 and then falls. Its tuned point, λ = 0.45, is the 0.6582 row in the table above. That clean score beats the submitted RM3, and Track A is 0.6076 against 0.6521, with retention 0.973 against 1.002.

The 50% column is one noise seed. Over ten seeds on the competition model:

| λ | Mean nDCG@10 | Std | Mean retention | Std |
|---|---|---|---|---|
| 0.0 | 0.6466 | 0.014 | 0.965 | 0.020 |
| 0.9 | 0.6537 | 0.0006 | 1.000 | 0.0009 |

λ = 0 is 0.017 higher on the single clean run and about 0.007 lower under the ten noise seeds. I kept 0.9 because retention stays near 1 and, on the other collections below, nDCG moves by at most 0.0007.

### The same RM3 on every local collection

Competition RM3, λ = 0.9, on every collection that had a released candidate file. Noise is the same practice recipe, one draw per query.

| Collection | Queries | Track A | Clean | 25% | 50% | Retention |
|---|---|---|---|---|---|---|
| Toy | 6 | 0.8716 | 0.8716 | 0.8900 | 0.8716 | 1.000 |
| Cranfield | 225 | 0.2984 | 0.2982 | 0.2982 | 0.2977 | 0.998 |
| CACM | 52 | 0.5018 | 0.5018 | 0.5018 | 0.5018 | 1.000 |
| Vaswani | 93 | 0.4065 | 0.4065 | 0.4058 | 0.4058 | 0.998 |
| SciFact | 300 | 0.7001 | 0.7001 | 0.7001 | 0.7000 | 1.000 |
| NFCorpus | 323 | 0.3195 | 0.3196 | 0.3197 | 0.3198 | 1.001 |
| FiQA | 648 | 0.2673 | 0.2676 | 0.2673 | 0.2676 | 1.000 |
| SciDocs | 1000 | 0.1660 | 0.1658 | 0.1659 | 0.1659 | 1.001 |
| ArguAna | 1406 | 0.3811 | 0.3811 | 0.3811 | 0.3811 | 1.000 |
| Touché-2020 | 49 | 0.4323 | 0.4322 | 0.4322 | 0.4323 | 1.000 |
| Antique | 200 | 0.3837 | 0.3839 | 0.3839 | 0.3839 | 1.000 |
| TREC-COVID dev | 50 | 0.6521 | 0.6534 | 0.6533 | 0.6546 | 1.002 |

On the ten non-toy collections other than TREC-COVID, nDCG@10 moves by at most 0.0007 (Cranfield 0.2984 to 0.2977, Vaswani 0.4065 to 0.4058). The toy set is the exception: six queries make one swapped document a large swing, so the 25% score moves from 0.8716 to 0.8900.

### Hidden leaderboard

Each row is one upload. The five numbers are what the leaderboard printed. C1 and C2 are retention of the noisy conditions against the clean feedback run. A separate parameter set for Track A and Track B raised Track A and dropped C2 to 0.78241; that split was removed. A translation model on the query terms cut Track A from 0.53809 to 0.53558 and was taken out. The submitted run is one shared model, with Track B 0.00005 above Track A. The dev noise standard deviation at λ = 0.9 is 0.0006, so that hidden gap sits inside the noise.

| Upload | A nDCG | A MAP | B nDCG | C1 | C2 |
|---|---|---|---|---|---|
| Early single model | 0.53295 | 0.45387 | 0.53373 | 1.00000 | — |
| Separate parameters for A and B | 0.53821 | 0.45985 | 0.53289 | 1.00000 | 0.78241 |
| One model, Track B still below Track A | 0.53820 | 0.46070 | 0.53783 | 1.00000 | 0.99927 |
| Translation model on query terms | 0.53558 | — | — | — | — |
| Submitted | 0.53809 | 0.46063 | 0.53814 | 1.00000 | 0.99854 |

## Adversarial seed

`submission/adversarial_set.json` lists dev queries 48, 29, 44, 25, and 22. Each list is drawn from that query's own top-100 pool. Every document is non-relevant or unjudged, and each has a high query likelihood, so a feedback function that trusts P(D | Q) treats it as a good seed and promotes it into the top 10.

Each list was grown one document at a time, then one swap or deletion was tried when it raised the average nDCG@10 drop. The panel was ten of my own setups, each with Porter stemming and with raw tokens: submitted RM3 and RM1, the tuned Dirichlet and Jelinek–Mercer RM3, textbook RM1, RM2, and RM3, RM3 with a uniform document prior, λ = 0.7, and Dirichlet RM3 with μ = 2000.

On that panel the file drops mean nDCG@10 by 0.649:

| Query | Documents | Mean drop |
|---|---|---|
| 48 | 8 | 0.702 |
| 44 | 5 | 0.671 |
| 29 | 6 | 0.657 |
| 22 | 6 | 0.625 |
| 25 | 7 | 0.589 |

The submitted model does not move. Its top 10 is the same on all five queries, because λ = 0.9 leaves little weight on the seed and the coverage filter drops documents that miss most of the query. Typical RM1 and RM3 setups on the panel drop by about 0.6 to 0.9.
