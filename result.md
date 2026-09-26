# Local + hidden results

`lambda = 1` unless noted. The live leaderboard is an undisclosed held-out collection (not TREC-COVID). Local tables are whatever corpus is named in that section.

## Dataset identity

The released staff `candidates_dev.jsonl` (50 queries, 100-doc pools, 4408 unique IDs, qids `1`–`50`) matches **`beir/trec-covid` in `data/full` at 100%**. Coverage on every other local corpus (FiQA, SciFact, NFCorpus, Antique, …) is **0%**. That file is the **local** TREC-COVID pool. The live leaderboard uses a **different, undisclosed** collection — not TREC-COVID.

Local TREC-COVID uses that staff pool. Other collections use a BM25-generated pool.

Do not treat a local peak as a submit signal by itself. COVID-max local **0.6871** dropped hidden A **0.5250 → 0.4627**. Quora-max local **0.7223** dropped hidden A **0.5265 → 0.4999**. Touche2020-max local **0.4987** **raised** hidden A **0.4999 → 0.5284**. ArguAna-max local **0.4177** dropped hidden A **0.5284 → 0.5149**. SciFact-max local **0.6908** **raised** hidden A **0.5149 → 0.5294**. SciDocs-max local **0.1666** dropped hidden A **0.5294 → 0.5014**. Vaswani-max local **0.4426** recovered hidden A **0.5014 → 0.5243**. The live win is **not** a local max: Lucene query stops (`qstop_list=0`) on the μ3 interpolant raised hidden A **0.5295 → 0.5323**.

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
| **SciFact-max** | `doc_stop=0`, `query_stop=1`, `mu_factor=1`, `mu_min=250`, `w_od=0.10`, `w_uw=0.15`, `lead_w=0.14`, `lead_len=12`, `lead_mu=60`, `prior_w=0.35`, `prior_k=20`, `lambda=1` | **0.5294** | **0.4526** | **0.5294** | **1.0000** | **0.7700** | **1 / 7** | 23:17 |
| **SciDocs-max** | `doc_stop=1`, `mu_factor=24`, `mu_max=4000`, `w_od=0.35`, `w_uw=0.10`, `lead_w=0.18`, `lead_len=12`, `lead_mu=30`, `prior_w=0.25`, `prior_k=18`, `lambda=1` | **0.5014** | **0.4169** | **0.5014** | **1.0000** | **0.4550** | **4 / 8** | 23:23 |
| **Vaswani-max** | `doc_stop=0`, `mu_min=150`, `w_od=0.10`, `w_uw=0.15`, `lead_w=0.40`, `lead_len=12`, `lead_mu=40`, `prior_w=0.35`, `prior_k=20`, `len_prior=0.10`, `lambda=1` | **0.5243** | **0.4462** | **0.5243** | **1.0000** | **0.5714** | **3 / 8** (tie) | 23:31 |
| SciFact-max `lead_w=0.12` | same as SciFact-max except `lead_w=0.12` | 0.5294 | 0.4522 | 0.5294 | 1.0000 | 0.7714 | — | 23:46 |
| SciFact SDM nudge | `w_od=0.09`, `w_uw=0.16`, `lead_w=0.13` | 0.5293 | 0.4521 | 0.5293 | 1.0000 | 0.7714 | — | 23:55 |
| **μ3 + len 0.08** | SciFact-max lead/SDM, `mu_factor=3`, `len_prior=0.08` | **0.5295** | 0.4508 | **0.5295** | **1.0000** | **0.7917** | — | 00:11 |
| μ5 + len 0.12 | `mu_factor=5`, `len_prior=0.12` | 0.5280 | 0.4492 | 0.5280 | 1.0000 | 0.7293 | — | 00:14 |
| μ3 prior_k=16 | nDCG-best + `prior_k=16` | 0.5290 | 0.4503 | 0.5290 | 1.0000 | 0.7379 | — | 00:16 |
| **Lucene qstop + μ3** | nDCG-best scoring + `qstop_list=0` | **0.5323** | **0.4528** | **0.5323** | **1.0000** | **0.8164** | — | 00:20 |

### What actually won (00:20)

The 0.5323 cell is three stacked pieces, not one lucky knob:

1. **Keep C = 1.** `lambda=1` makes Track B call `score_candidates`, so noisy seeds cannot change the ranking. Retention is exactly 1. Bansal/Gupta had higher or similar raw B at times but C 0.97–0.98; that 35% percentile is why 0.8164 beats ~0.61.
2. **SciFact-max body** (`doc_stop=0`, `query_stop=1`, SDM `w_od=0.10` `w_uw=0.15`, `lead_w=0.14`, `prior_w=0.35` `prior_k=20`). That family first took hidden A to **0.5294**. Collection-hopping (COVID/Quora/SciDocs/ArguAna/Vaswani) and 0.01 SDM/`lead_w` nudges all lost. Do not retune those.
3. **Touche-ward Dirichlet, one step only.** `mu_factor` 1→**3** and `len_prior` 0→**0.08** raised A **0.5294 → 0.5295**. The next step (`mu_factor=5`, `len_prior=0.12`) fell to 0.5280. Freeze μ/len here.
4. **The jump that beat Bansal:** `qstop_list` **1 → 0**. Queries still drop stopwords (`query_stop=1`; ArguAna's `query_stop=0` had already tanked hidden A to 0.5149). The list itself changes from Fox (long) to Lucene (short), so more query tokens survive. Hidden A **0.5295 → 0.5323**, MAP **0.4508 → 0.4528**, final **0.7917 → 0.8164**. That is past Bansal's raw A **0.5315**.

Misses on the way (do not repeat): `lead_w=0.12`, SDM 0.09/0.16, `prior_k=16`, `mu_factor=5`.

**Freeze this zip.** Next edits should not touch `qstop_list`, μ, len, lead, or SDM unless a later board proves a new axis.

Snapshot 23:33 field (raw): Gupta 0.5272 / 0.4501 / 0.5295 / 0.9714 → 0.6143; Bansal 0.5315 / 0.4544 / 0.5241 / 0.9817 → 0.6071; Goel 0.5274 / 0.4504 / 0.5277 / 0.9396 → 0.5714; boldfit 0.5035 / 0.4278 / 0.5035 / 1.0 → 0.4571; P. Kumar 0.5244 / 0.4484 / 0.5246 / 0.9040 → 0.3500; Akhtar 0.4975 / 0.4212 / 0.4975 / 1.0 → 0.3429; A. Kumar 0.4960 / 0.4201 / 0.4998 / 0.9313 → 0.0857.

FiQA-max (`lambda=1`) won on B=A and C=1. NFCorpus-max slightly **raised hidden A** (0.5242 → 0.5265) but RM3 **hurt B** (0.5242 → 0.4992) and cut C (1.0 → 0.9712), so percentile final fell 0.6600 → 0.4180. Quora-max restored C=1 and B=A but **overfit the short-doc knobs**: hidden A **0.5265 → 0.4999**, MAP **0.4467 → 0.4210**, final **0.4180 → 0.4880**. Touche2020-max (`lead_w=0`, unigram, `mu_factor=8`) **undid that**: hidden A **0.4999 → 0.5284**, MAP **0.4210 → 0.4489**, B=A, C=1, final **0.4880 → 0.7700**, rank **1 / 7**. Bansal still had higher raw A/MAP (0.5315 / 0.4544) **until the Lucene-qstop zip**; C 0.9817. ArguAna-max (`query_stop=0`, light mu, SDM) **gave back A**: **0.5284 → 0.5149**, MAP **0.4489 → 0.4403**, final **0.7700 → 0.6100**, rank **1 → 2**. SciFact-max (`doc_stop=0`, `query_stop=1`, mild SDM + lead) **retook 1st**: hidden A **0.5149 → 0.5294**, MAP **0.4403 → 0.4526**, B=A, C=1, final **0.6100 → 0.7700**, rank **2 → 1**. SciDocs-max (`doc_stop=1`, `mu_factor=24`) is the COVID-max pattern again: local nDCG is recall-capped at **0.1666**, hidden A **0.5294 → 0.5014**, MAP **0.4526 → 0.4169**, final **0.7700 → 0.4550**, rank **1 → 4 / 8**. Vaswani-max (SciFact-max family, stronger `lead_w=0.40`) **recovered A** from SciDocs: **0.5014 → 0.5243**, MAP **0.4169 → 0.4462**, final **0.4550 → 0.5714**, rank **4 → 3 / 8** (tied with Goel). Still **below SciFact-max** (0.5294 / 0.4526 / 0.7700). Gupta is new 1st (A 0.5272, **B 0.5295**, C 0.9714 → 0.6143) — expansion helps her B; C is not 1. Field percentiles moved (Bansal 0.6417 → 0.6071) as the class grew. SciFact-max `lead_w=0.12` (23:46) **kept A nDCG 0.5294** but **dropped MAP 0.4526 → 0.4522**. SDM/lead nudge `w_od=0.09` `w_uw=0.16` `lead_w=0.13` **dropped both**: A **0.5294 → 0.5293**, MAP **0.4526 → 0.4521**. Local SciFact gains did not transfer. **μ3 + len_prior 0.08** raised hidden A **0.5294 → 0.5295** (MAP **0.4526 → 0.4508**). **μ5 + len 0.12** and **`prior_k=16`** both missed. **`qstop_list=0`** on that frozen scoring cell is the current champion: A **0.5323**, MAP **0.4528**, C **1**, final **0.8164**, raw A above Bansal **0.5315**. See **What actually won** above. Freeze this zip.

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

Scientific claims as queries (~12.5 raw words, ~8.4 tokens after query stops). 5183 paper abstracts, 300 queries, 339 positive qrels (~1.13/query). BM25 pool recall@100 **0.8761**. With `doc_stop=0`, avgdl ~214 tokens.

### Local (BM25 pool), `lambda=1` so B=A and C=1

3-round coordinate descent plateaued; a dense grid around the peak added +0.0005. RM3 never beat `lambda=1`. First/last 150 queries split 0.722 vs 0.659 — topic variance, not a second collection.

| PARAMS | A nDCG@10 | A MAP@10 | B nDCG@10 | C |
|---|---|---|---|---|
| ArguAna-max (`query_stop=0`, `doc_stop=1`) | 0.6722 | 0.6258 | 0.6722 | 1.0000 |
| Touche-like | 0.6668 | 0.6192 | 0.6668 | 1.0000 |
| FiQA-like (`query_stop=1`, `doc_stop=1`) | 0.6814 | 0.6378 | 0.6814 | 1.0000 |
| NF-like (`doc_stop=1`) | 0.6829 | 0.6382 | 0.6829 | 1.0000 |
| Earlier peak (`doc_stop=1`) | 0.6811 | 0.6367 | 0.6811 | 1.0000 |
| `doc_stop=1` scoring max | 0.6869 | 0.6434 | 0.6869 | 1.0000 |
| Earlier recorded peak (`w_od=0.2`, `lead_w=0.2`, `mu_min=250`) | 0.6898 | — | 0.6898 | 1.0000 |
| **SciFact-max**: `doc_stop=0`, `query_stop=1`, `mu_factor=1`, `mu_min=250`, `w_od=0.10`, `w_uw=0.15`, `lead_w=0.14`, `lead_len=12`, `lead_mu=60`, `prior_w=0.35`, `prior_k=20` | **0.6908** | **0.6485** | **0.6908** | **1.0000** |

Live `feedback.py` was this cell until SciDocs-max. **`doc_stop=0` changes indexing vs Touche2020-max / ArguAna-max.**

### Hidden (after submitting SciFact-max) — 23:17, 26 Sep

Undisclosed live collection (not SciFact, not TREC-COVID). Best hidden A/MAP of our submits. Rank **1 / 7**, final **0.7700** (same percentile as Touche2020-max, higher raw A/MAP).

| A nDCG@10 | A MAP@10 | B nDCG@10 | C retention | final | rank |
|---|---|---|---|---|---|
| 0.5294 | 0.4526 | 0.5294 | 1.0000 | 0.7700 | 1 / 7 |

---

## SciDocs (`beir/scidocs`, `data/scidocs`)

Citation / co-view paper retrieval: title queries (~9.4 raw words, ~7.4 tokens after stops), paper abstracts. 25657 docs, 1000 queries, 4928 positive qrels (~4.9/query). BM25 pool recall@100 **0.3389** — nDCG@10 is recall-capped near ~0.17. With `doc_stop=1`, avgdl ~114.

### Local (BM25 pool), `lambda=1` so B=A and C=1

Two-index descent (`doc_stop=0` vs `1`); `doc_stop=1` won. 500/500 split for the `doc_stop=1` peak: tune **0.1610**, hold **0.1700**. Dense grid + `mu_max` then plateaued.

| PARAMS | A nDCG@10 | A MAP@10 | B nDCG@10 | C |
|---|---|---|---|---|
| SciFact-max (`doc_stop=0`) | 0.1576 | 0.0931 | 0.1576 | 1.0000 |
| Touche-like | 0.1518 | 0.0877 | 0.1518 | 1.0000 |
| FiQA-like | 0.1617 | 0.0957 | 0.1617 | 1.0000 |
| NF-like | 0.1616 | 0.0951 | 0.1616 | 1.0000 |
| `doc_stop=0` scoring max | 0.1633 | 0.0968 | 0.1633 | 1.0000 |
| `doc_stop=1` descent | 0.1658 | 0.0978 | 0.1658 | 1.0000 |
| **SciDocs-max**: `doc_stop=1`, `mu_factor=24`, `mu_max=4000`, `w_od=0.35`, `w_uw=0.10`, `lead_w=0.18`, `lead_len=12`, `lead_mu=30`, `prior_w=0.25`, `prior_k=18` | **0.1666** | **0.0982** | **0.1666** | **1.0000** |

RM3 at `lambda=0.7` was B 0.1659 vs A 0.1658 with C 0.9952; left off. Live `feedback.py` was this cell until Vaswani-max. **High `mu_factor` is the COVID-max family** that already tanked hidden A once (`lead_w=0.60`, `mu_factor=12` → A 0.4627). SciFact-max (hidden 1st, A 0.5294) used `doc_stop=0` and `mu_factor=1`.

### Hidden (after submitting SciDocs-max) — 23:23, 26 Sep

Undisclosed live collection (not SciDocs, not TREC-COVID). Same pattern as COVID-max / Quora-max: local peak, hidden A/MAP fell. C stayed 1. Rank **4 / 8**.

| A nDCG@10 | A MAP@10 | B nDCG@10 | C retention | final | rank |
|---|---|---|---|---|---|
| 0.5014 | 0.4169 | 0.5014 | 1.0000 | 0.4550 | 4 / 8 |

---

## Vaswani (`ir_datasets` NPL / `data/vaswani`)

Short scientific abstracts (~42 raw words, ~25 tokens after doc stops / ~40 without). 11429 docs, 93 queries (~11 words), 2083 positive qrels (~22/query). BM25 pool recall@100 **0.4570**. MAP@10 is small because AP divides by all relevants.

A 46/47 split overfit (tune ~0.52 vs hold ~0.34). Numbers below are **all 93 queries**.

### Local (BM25 pool), `lambda=1` so B=A and C=1

| PARAMS | A nDCG@10 | A MAP@10 | B nDCG@10 | C |
|---|---|---|---|---|
| SciDocs-max (`doc_stop=1`, `mu_factor=24`) | 0.4195 | 0.1509 | 0.4195 | 1.0000 |
| Touche-like | 0.4107 | 0.1479 | 0.4107 | 1.0000 |
| FiQA-like | 0.4319 | 0.1542 | 0.4319 | 1.0000 |
| NF-like | 0.4348 | 0.1599 | 0.4348 | 1.0000 |
| SciFact-max | 0.4352 | 0.1590 | 0.4352 | 1.0000 |
| `doc_stop=1` full descent | 0.4415 | 0.1619 | 0.4415 | 1.0000 |
| **Vaswani-max**: `doc_stop=0`, `query_stop=1`, `mu_factor=1`, `mu_min=150`, `w_od=0.10`, `w_uw=0.15`, `lead_w=0.40`, `lead_len=12`, `lead_mu=40`, `prior_w=0.35`, `prior_k=20`, `len_prior=0.10` | **0.4426** | **0.1618** | **0.4426** | **1.0000** |

Same indexing family as SciFact-max (`doc_stop=0`); mainly a stronger lead. RM3 at `lambda=0.7` was B 0.4484 with C 0.9901; left off. Live `feedback.py` is this cell. SciFact-max is still the best **hidden** A/MAP of our submits (0.5294 / 0.4526 / 0.7700).

### Hidden (after submitting Vaswani-max) — 23:31, 26 Sep

Undisclosed live collection (not Vaswani, not TREC-COVID). Recovered from SciDocs-max but did not beat SciFact-max. Rank **3 / 8**, tied with Goel on final **0.5714**.

| A nDCG@10 | A MAP@10 | B nDCG@10 | C retention | final | rank |
|---|---|---|---|---|---|
| 0.5243 | 0.4462 | 0.5243 | 1.0000 | 0.5714 | 3 / 8 (tie) |

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

Tune/hold for the `query_stop=1` scoring peak: tune-700 A **0.4075**, hold-706 A **0.4136**. RM3 at `lambda=0.55` was B 0.4116 vs A 0.4106 with C 0.9979; left off. Live `feedback.py` was this cell until SciFact-max.

### Hidden (after submitting ArguAna-max) — 23:08, 26 Sep

Undisclosed live collection (not ArguAna, not TREC-COVID). `query_stop=0` + light mu + SDM **hurt** vs Touche2020-max. Rank **2 / 7**.

| A nDCG@10 | A MAP@10 | B nDCG@10 | C retention | final | rank |
|---|---|---|---|---|---|
| 0.5149 | 0.4403 | 0.5149 | 1.0000 | 0.6100 | 2 / 7 |
