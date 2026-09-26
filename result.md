# Local + hidden results

`lambda = 1` unless noted. Hidden eval is always TREC-COVID held-out topics; local NFCorpus/FiQA numbers are a different collection.

## Dataset identity

The staff `candidates_dev.jsonl` (50 queries, 100-doc pools, 4408 unique IDs, qids `1`–`50`) matches **`beir/trec-covid` in `data/full` at 100%**. Coverage on every other local corpus (FiQA, SciFact, NFCorpus, Antique, …) is **0%**. Dev topics/qrels are the public TREC-COVID set; the live leaderboard is held-out topics on that same collection, not FiQA.

Local TREC-COVID uses that staff pool. Other collections use a BM25-generated pool.

Do not treat a local peak as a submit signal by itself. COVID-max local **0.6871** dropped hidden A **0.5250 → 0.4627**.

---

## Why Track A and Track B nDCG are the same

Track A is `score_candidates` (query-likelihood + SDM + lead + priors).

Track B is `relevance_model_feedback` (RM3): mix the original query with a language model estimated from pseudo-relevant docs.

The mix weight is `lambda`:

- `lambda = 1` → 100% original query, 0% expansion
- In `rm_engine.cpp` this short-circuits: `if (P.lambda >= 1) return score_candidates(...)`

So the ranked list is identical, nDCG matches, and noisy seeds cannot change the ranking either (retention 1.0). That is intentional: expansion was hurting hidden B (0.5058 vs A 0.5250). Turning RM3 back on (`lambda < 1`) is the only way A and B can diverge.

---

## Hidden / live leaderboard (graded TREC-COVID held-out)

Same topics every submit. Percentile final score, not raw average.

| submit | PARAMS | A nDCG@10 | A MAP@10 | B nDCG@10 | C retention | final | rank | time |
|---|---|---|---|---|---|---|---|---|
| pre-COVID-max | `lead_w=0.35`, SDM 0.05, `mu_factor=10`, `prior_w=0.60`, `lambda=1` | 0.5250 | — | 0.5250 | 1.0000 | 0.6625 | 1 | earlier |
| RM3 on | `lambda` 0.80–0.95 | — | — | 0.5199 | — | — | — | earlier |
| COVID-max | unigram, `lead_w=0.60`, `mu_factor=12`, `len_prior=0.20` | 0.4627 | 0.3784 | 0.4627 | 1.0000 | 0.2625 | 5 / 6 | 21:44 |
| **FiQA-max** | `w_od=w_uw=0.10`, `lead_w=0.40`, `len_prior=0`, `prior_w=0.30`, `prior_k=10`, `mu_factor=1`, `mu_min=150`, `lambda=1` | **0.5242** | **0.4467** | **0.5242** | **1.0000** | **0.6600** | **1 / 7** | 22:00 |
| **NFCorpus-max** | `doc_stop=1`, `lead_mu=20`, `w_od=0.2`, `w_uw=0`, `lead_w=0.2`, `prior_w=0.15`, `mu_factor=4`, `lambda=0.5` | **0.5265** | **0.4467** | **0.4992** | **0.9712** | **0.4180** | **4 / 7** | 22:18 |

Snapshot 22:20 field (raw): Bansal 0.5315 / 0.4544 / 0.5241 / 0.9817 → 0.7100; boldfit 0.5035 / 0.4278 / 0.5035 / 1.0 → 0.5850; Kumar 0.5244 / 0.4484 / 0.5246 / 0.9040 → 0.4420.

FiQA-max (`lambda=1`) won on B=A and C=1. NFCorpus-max slightly **raised hidden A** (0.5242 → 0.5265) but RM3 **hurt B** (0.5242 → 0.4992) and cut C (1.0 → 0.9712), so percentile final fell 0.6600 → 0.4180.

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
| + RM3 `lambda=0.5`, `fb_mu=400`, `fb_terms=20` (live PARAMS) | 0.3229 | **0.3258** | 0.9874 |
| Full-corpus QL (no pool) | 0.3274 | 0.3274 | 1.0000 |
| Full-corpus RM3 `lambda=0.5` | 0.3274 | 0.3393 | 0.9841 |

Live `feedback.py` is this NFCorpus cell (`doc_stop=1` changes indexing). That is **not** the FiQA-max 1st-place submit.

### Hidden (after submitting NFCorpus-max) — 22:18, 26 Sep

Graded set is still held-out TREC-COVID, not NFCorpus.

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

Live PARAMS stay on FiQA-max. SciFact peak is a local-only cell; do not submit it without a reason to leave 1st place.
