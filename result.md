# Local + hidden results

`lambda = 1` unless noted. The live leaderboard is an undisclosed held-out collection (not TREC-COVID). Local tables are whatever corpus is named in that section.

## Dataset identity

The released staff `candidates_dev.jsonl` (50 queries, 100-doc pools, 4408 unique IDs, qids `1`–`50`) matches **`beir/trec-covid` in `data/full` at 100%**. Coverage on every other local corpus (FiQA, SciFact, NFCorpus, Antique, …) is **0%**. That file is the **local** TREC-COVID pool. The live leaderboard uses a **different, undisclosed** collection — not TREC-COVID.

Local TREC-COVID uses that staff pool. Other collections use a BM25-generated pool.

Do not treat a local peak as a submit signal by itself. COVID-max local **0.6871** dropped hidden A **0.5250 → 0.4627**. Quora-max local **0.7223** dropped hidden A **0.5265 → 0.4999**. Touche2020-max local **0.4987** **raised** hidden A **0.4999 → 0.5284**. ArguAna-max local **0.4177** dropped hidden A **0.5284 → 0.5149**.

---

## Why Track A and Track B nDCG are the same

Track A is `score_candidates` (query-likelihood + SDM + lead + priors).

Track B is `relevance_model_feedback` (RM3): mix the original query with a language model estimated from pseudo-relevant docs.

The mix weight is `lambda`:

- `lambda = 1` → 100% original query, 0% expansion
- In `rm_engine.cpp` this short-circuits: `if (P.lambda >= 1) return score_candidates(...)`

So the ranked list is identical, nDCG matches, and noisy seeds cannot change the ranking either (retention 1.0). That is intentional: expansion was hurting hidden B (0.5058 vs A 0.5250). Turning RM3 back on (`lambda < 1`) is the only way A and B can diverge.

---

## Hidden / live leaderboard (undisclosed collection)

Same hidden topics every submit. Percentile final score, not raw average. Not TREC-COVID.

| submit | PARAMS | A nDCG@10 | A MAP@10 | B nDCG@10 | C retention | final | rank | time |
|---|---|---|---|---|---|---|---|---|
| pre-COVID-max | `lead_w=0.35`, SDM 0.05, `mu_factor=10`, `prior_w=0.60`, `lambda=1` | 0.5250 | — | 0.5250 | 1.0000 | 0.6625 | 1 | earlier |
| RM3 on | `lambda` 0.80–0.95 | — | — | 0.5199 | — | — | — | earlier |
| COVID-max | unigram, `lead_w=0.60`, `mu_factor=12`, `len_prior=0.20` | 0.4627 | 0.3784 | 0.4627 | 1.0000 | 0.2625 | 5 / 6 | 21:44 |
| **FiQA-max** | `w_od=w_uw=0.10`, `lead_w=0.40`, `len_prior=0`, `prior_w=0.30`, `prior_k=10`, `mu_factor=1`, `mu_min=150`, `lambda=1` | **0.5242** | **0.4467** | **0.5242** | **1.0000** | **0.6600** | **1 / 7** | 22:00 |
| **NFCorpus-max** | `doc_stop=1`, `lead_mu=20`, `w_od=0.2`, `w_uw=0`, `lead_w=0.2`, `prior_w=0.15`, `mu_factor=4`, `lambda=0.5` | **0.5265** | **0.4467** | **0.4992** | **0.9712** | **0.4180** | **4 / 7** | 22:18 |
| **Quora-max** | `mu_min=50`, `w_od=0.10`, `lead_w=0.65`, `lead_len=6`, `lead_mu=8`, `prior_w=0.40`, `prior_k=5`, `lambda=1` | **0.4999** | **0.4210** | **0.4999** | **1.0000** | **0.4880** | — | 22:44 |
| **Touche2020-max** | unigram, `lead_w=0`, `mu_factor=8`, `prior_w=0.35`, `prior_k=12`, `len_prior=0.15`, `lambda=1` | **0.5284** | **0.4489** | **0.5284** | **1.0000** | **0.7700** | **1 / 7** | 22:50 |
| **ArguAna-max** | `query_stop=0`, `mu_min=50`, `mu_factor=0.5`, `w_od=0.35`, `w_uw=0.10`, `lead_w=0`, `prior_w=0.08`, `lambda=1` | **0.5149** | **0.4403** | **0.5149** | **1.0000** | **0.6100** | **2 / 7** | 23:08 |

Snapshot 23:10 field (raw): Bansal 0.5315 / 0.4544 / 0.5241 / 0.9817 → 0.6400; boldfit 0.5035 / 0.4278 / 0.5035 / 1.0 → 0.5000; P. Kumar 0.5244 / 0.4484 / 0.5246 / 0.9040 → 0.4900; Akhtar 0.4975 / 0.4212 / 0.4975 / 1.0 → 0.3400; A. Kumar 0.4960 / 0.4201 / 0.4998 / 0.9313 → 0.1200.

FiQA-max (`lambda=1`) won on B=A and C=1. NFCorpus-max slightly **raised hidden A** (0.5242 → 0.5265) but RM3 **hurt B** (0.5242 → 0.4992) and cut C (1.0 → 0.9712), so percentile final fell 0.6600 → 0.4180. Quora-max restored C=1 and B=A but **overfit the short-doc knobs**: hidden A **0.5265 → 0.4999**, MAP **0.4467 → 0.4210**, final **0.4180 → 0.4880**. Touche2020-max (`lead_w=0`, unigram, `mu_factor=8`) **undid that**: hidden A **0.4999 → 0.5284**, MAP **0.4210 → 0.4489**, B=A, C=1, final **0.4880 → 0.7700**, rank **1 / 7**. Bansal still has higher raw A/MAP (0.5315 / 0.4544) but C 0.9817. ArguAna-max (`query_stop=0`, light mu, SDM) **gave back A**: **0.5284 → 0.5149**, MAP **0.4489 → 0.4403**, final **0.7700 → 0.6100**, rank **1 → 2**. C stayed 1. Bansal’s final rose 0.5900 → 0.6400 from the percentile shift. This is **not** the same as COVID-max (that used `lead_w=0.60` and `mu_factor=12` and tanked A to 0.4627).

---

## TREC-COVID (`beir/trec-covid`, `data/full`)

| | |
|---|---|
| docs | 171332 |
| avgdl | ~161.5 |
| queries | 50 |
| query length | ~6.1 tokens |

### Local (staff candidate pool), `lambda=1` so B=A and C=1

| PARAMS | A nDCG@10 | A MAP@10 | B nDCG@10 | C retention |
|---|---|---|---|---|
| Board / pre-max: `mu_factor=10`, `w_od=w_uw=0.05`, `lead_w=0.35`, `lead_len=12`, `len_prior=0.15`, `prior_w=0.60` | 0.6696 | — | 0.6696 | 1.0000 |
| Unigram + stronger lead: `w_od=w_uw=0`, `lead_w=0.55`, `mu_factor=14` | 0.6870 | — | 0.6870 | 1.0000 |
| COVID-max: `lead_w=0.60`, `mu_factor=12`, `mu_max=8000` | 0.6871 | — | 0.6871 | 1.0000 |
| FiQA-max PARAMS, **new** staff pool, `lambda=1` | 0.6416 | — | 0.6416 | 1.0000 |

### Local RM3 (new staff pool, FiQA-max Track A knobs)

A nDCG stays **0.6416**. Track B is clean RM3. Practice C is `min(n25,n50)/clean` clamped at 1.

| lambda | fb_mu | fb_docs | fb_terms | rm2_mix | A nDCG | B nDCG | C retention | B−A |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 200 | 10 | 20 | 0 | 0.6416 | 0.6416 | 1.0000 | 0 |
| 0.90 | 200 | 10 | 20 | 0 | 0.6416 | 0.6471 | 1.0000 | +0.0055 |
| 0.70 | 400 | 10 | 20 | 0 | 0.6416 | 0.6549 | 1.0000 | +0.0133 |
| 0.65 | 400 | 10 | 20 | 0 | 0.6416 | **0.6568** | **0.9992** | +0.0152 |
| 0.60 | 400 | 10 | 20 | 0 | 0.6416 | 0.6600 | 0.9944 | +0.0184 |
| 0.55 | 400 | 10 | 20 | 0 | 0.6416 | **0.6614** | 0.9926 | +0.0197 |

RM2 mix did not help. Live PARAMS: **`lambda=0.65`**, `fb_mu=400`. Hidden RM3 previously *hurt* (B 0.5199 vs A 0.5250) on older Track A knobs.

---

## FiQA (`beir/fiqa`, `data/fiqa`)

Finance forum answers. BM25 candidate pool. 648 queries.

| | |
|---|---|
| docs | 57638 |
| avgdl | ~130.2 |
| queries | 648 |
| query length | ~5.5 tokens |

### Local (BM25 pool), `lambda=1` so B=A and C=1

| PARAMS | A nDCG@10 | A MAP@10 | B nDCG@10 | C retention |
|---|---|---|---|---|
| COVID-max | 0.2481 | — | 0.2481 | 1.0000 |
| Coarse SDM / `mu_factor=4` | 0.2690 | — | 0.2690 | 1.0000 |
| FiQA-max (live PARAMS at FiQA submit) | **0.2764** | — | **0.2764** | **1.0000** |

### Hidden (after submitting FiQA-max) — 22:00, 26 Sep

| A nDCG@10 | A MAP@10 | B nDCG@10 | C retention | final | rank |
|---|---|---|---|---|---|
| 0.5242 | 0.4467 | 0.5242 | 1.0000 | 0.6600 | 1 / 7 |

---

## NFCorpus (`beir/nfcorpus`, `data/nfcorpus`)

Short medical title queries (~3.3 raw words, ~2.4 after stops). 3633 docs, 323 queries. BM25 pool recall of **all** relevants is only 15%, but each query has ~38 qrels so nDCG@10 is not that recall-bound: scoring the **entire corpus** only reaches A **0.3274** / RM3 B **0.3393**. Pool rerank is already ~98% of that A ceiling. Unigram+SDM params cannot jump much further; a different scoring family would.

| PARAMS | A nDCG@10 | B nDCG@10 | C retention |
|---|---|---|---|
| FiQA-max | 0.3194 | 0.3194 | 1.0000 |
| Earlier peak (`prior_w=0.2`, `len_prior=0.1`) | 0.3216 | 0.3216 | 1.0000 |
| Track A max: `doc_stop=1`, `lead_mu=20`, `w_od=0.2`, `w_uw=0`, `lead_w=0.2`, `prior_w=0.15`, `mu_factor=4`, `len_prior=0` | **0.3229** | 0.3229 | 1.0000 |
| + RM3 `lambda=0.65` | 0.3229 | 0.3245 | 0.9974 |
| + RM3 `lambda=0.5`, `fb_mu=400`, `fb_terms=20` (previous live) | 0.3229 | **0.3258** | 0.9874 |
| Full-corpus QL (no pool) | 0.3274 | 0.3274 | 1.0000 |
| Full-corpus RM3 `lambda=0.5` | 0.3274 | 0.3393 | 0.9841 |

Live `feedback.py` was this NFCorpus cell until Quora-max (`lambda=1`). `doc_stop=1` still changes indexing vs FiQA-max 1st-place.

### Hidden (after submitting NFCorpus-max) — 22:18, 26 Sep

Undisclosed live collection (not NFCorpus, not TREC-COVID).

| A nDCG@10 | A MAP@10 | B nDCG@10 | C retention | final | rank |
|---|---|---|---|---|---|
| 0.5265 | 0.4467 | 0.4992 | 0.9712 | 0.4180 | 4 / 7 |

---

## SciFact (`beir/scifact`, `data/scifact`)

Scientific claims as queries (~8.4 tokens), 5183 docs, avgdl ~214. BM25 pool. `lambda=1` so B=A, C=1.

| PARAMS | A nDCG@10 | A MAP@10 | B nDCG@10 | C retention |
|---|---|---|---|---|
| FiQA-max | 0.6823 | — | 0.6823 | 1.0000 |
| Peak: `w_od=0.2`, `w_uw=0.1`, `lead_w=0.2`, `prior_w=0.2`, `mu_factor=1`, `mu_min=250`, `len_prior=0`, `lead_len=16`, `prior_k=10` | **0.6898** | — | **0.6898** | **1.0000** |

SciFact peak is a local-only cell. Live PARAMS are now **ArguAna-max**.

---

## Quora (`beir/quora`, `data/quora` + `data/quora_tune`)

Duplicate-question retrieval. Docs and queries are short questions (raw avgdl ~10.9, ~5.7 tokens after stops). Full set: 522931 docs, 10000 queries, 15675 qrels. `data/quora/candidates_dev.jsonl` is empty; BM25 top-100 pools were built for an 800-query tune split (`data/quora_tune`, recall@100 **0.8052**) and a disjoint 400-query holdout (`data/quora_hold`, recall@100 **0.8103**).

### Local Track A (`lambda=1` so B=A and C=1)

NFCorpus-max RM3 (`lambda=0.5`) **hurts** Quora B (tune 0.6476 vs A 0.6875). Keep expansion off.

| PARAMS | split | A nDCG@10 | A MAP@10 | B nDCG@10 | C |
|---|---|---|---|---|---|
| NFCorpus-max (`lambda=0.5`) | tune-800 | 0.6875 | 0.6179 | 0.6476 | 1.0000 |
| FiQA-max (`lambda=1`) | tune-800 | 0.7012 | 0.6329 | 0.7012 | 1.0000 |
| **Quora-max** | tune-800 | **0.7223** | **0.6563** | **0.7223** | **1.0000** |
| NFCorpus-max (`lambda=0.5`) | hold-400 | 0.6980 | 0.6268 | 0.6498 | 1.0000 |
| FiQA-max (`lambda=1`) | hold-400 | 0.7121 | 0.6405 | 0.7121 | 1.0000 |
| **Quora-max** | hold-400 | **0.7364** | **0.6679** | **0.7364** | **1.0000** |

Quora-max knobs (vs NFCorpus-max): `mu_min=50` (was 150; Dirichlet was oversmoothing 6-token docs), `w_od=0.10`, `w_uw=0`, `lead_w=0.65`, `lead_len=6`, `lead_mu=8`, `prior_w=0.40`, `prior_k=5`, `lambda=1`. Stemmer 2 + doc/query stops stay; turning stops off lost ~0.007–0.015.

RM3 (`lambda` 0.5–0.95) never beat `lambda=1` on tune. Live `feedback.py` was this cell until Touche2020-max.

### Hidden (after submitting Quora-max) — 22:44, 26 Sep

Undisclosed live collection (not Quora, not TREC-COVID). Same pattern as COVID-max: local nDCG jumped, hidden A/MAP fell.

| A nDCG@10 | A MAP@10 | B nDCG@10 | C retention | final | rank |
|---|---|---|---|---|---|
| 0.4999 | 0.4210 | 0.4999 | 1.0000 | 0.4880 | — |

---

## Touché 2020 (`beir/webis-touche2020`, `data/touche2020`)

Argument retrieval: short yes/no debate queries, long web arguments. 382545 docs, 49 queries, 2962 qrel lines (2413 positive, ~49 relevant/query). Raw avgdl ~294 words, ~154 tokens after stops. Existing BM25 top-100 pool, recall@100 **0.4356**. MAP@10 is small because AP divides by all relevants (same effect as TREC-COVID).

### Local (BM25 pool), `lambda=1` so B=A and C=1

The lead field **hurts**: debate titles restated the motion, so `lead_w=0.65` (Quora-max) scored **0.4111**. Turning lead off jumped to ~0.48.

| PARAMS | A nDCG@10 | A MAP@10 | B nDCG@10 | C |
|---|---|---|---|---|
| Quora-max | 0.4111 | 0.1289 | 0.4111 | 1.0000 |
| FiQA-like scoring | 0.4206 | 0.1395 | 0.4206 | 1.0000 |
| COVID-like scoring | 0.4298 | 0.1376 | 0.4298 | 1.0000 |
| NF-like scoring | 0.4398 | 0.1409 | 0.4398 | 1.0000 |
| **Touche2020-max**: unigram, `lead_w=0`, `mu_factor=8`, `prior_w=0.35`, `prior_k=12`, `len_prior=0.15` | **0.4987** | **0.1638** | **0.4987** | **1.0000** |

RM3 (`lambda` 0.5–0.95) never beat `lambda=1` (best expansion B ~0.4957 at 0.95). `query_stop=0` was +0.0013 nDCG and −MAP; left off. Live `feedback.py` was this cell until ArguAna-max. Unlike COVID-max, dropping the lead (not cranking it) transferred to the hidden set.

### Hidden (after submitting Touche2020-max) — 22:50, 26 Sep

Undisclosed live collection (not Touché, not TREC-COVID). Best hidden A/MAP so far among our submits. Rank **1 / 7**, final **0.7700**.

| A nDCG@10 | A MAP@10 | B nDCG@10 | C retention | final | rank |
|---|---|---|---|---|---|
| 0.5284 | 0.4489 | 0.5284 | 1.0000 | 0.7700 | 1 / 7 |

---

## ArguAna (`beir/arguana`, `data/arguana`)

Counterargument retrieval: the query is a full debate argument (~194 raw words, ~91 tokens after analysis), the document is the opposing argument. 8674 docs, 1406 queries, **one** positive qrel each. BM25 pool recall@100 **0.9196**. avgdl ~94 tokens after stops.

### Local (BM25 pool), `lambda=1` so B=A and C=1

Touche2020-max is a bad fit here (short yes/no queries vs document-as-query): high `mu_factor` and a strong BM25 prior both hurt. Descent used 700 tune / 706 hold; numbers below are **full 1406**.

| PARAMS | A nDCG@10 | A MAP@10 | B nDCG@10 | C |
|---|---|---|---|---|
| Touche2020-max | 0.3234 | 0.2089 | 0.3234 | 1.0000 |
| Quora-like | 0.3133 | 0.2003 | 0.3133 | 1.0000 |
| FiQA-like | 0.3648 | 0.2417 | 0.3648 | 1.0000 |
| NF-like | 0.3698 | 0.2436 | 0.3698 | 1.0000 |
| scoring-only peak (`query_stop=1`) | 0.4106 | 0.2789 | 0.4106 | 1.0000 |
| **ArguAna-max**: `query_stop=0`, `mu_min=50`, `mu_factor=0.5`, `w_od=0.35`, `w_uw=0.10`, `lead_w=0`, `prior_w=0.08`, `prior_k=20`, `len_prior=0.20`, `longq_n=32` | **0.4177** | **0.2853** | **0.4177** | **1.0000** |

Tune/hold for the `query_stop=1` scoring peak: tune-700 A **0.4075**, hold-706 A **0.4136**. RM3 at `lambda=0.55` was B 0.4116 vs A 0.4106 with C 0.9979; left off. Live `feedback.py` is ArguAna-max.

### Hidden (after submitting ArguAna-max) — 23:08, 26 Sep

Undisclosed live collection (not ArguAna, not TREC-COVID). `query_stop=0` + light mu + SDM **hurt** vs Touche2020-max. Rank **2 / 7**.

| A nDCG@10 | A MAP@10 | B nDCG@10 | C retention | final | rank |
|---|---|---|---|---|---|
| 0.5149 | 0.4403 | 0.5149 | 1.0000 | 0.6100 | 2 / 7 |
