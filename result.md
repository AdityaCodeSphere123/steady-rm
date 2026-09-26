# Local + hidden results

`lambda = 1` unless noted (Track B ranking = Track A). nDCG is @10. Local pools are staff `candidates_dev.jsonl` on TREC-COVID (`data/full`); other collections use a BM25-generated pool.

Do not treat a local peak as a submit signal by itself. The COVID-max PARAMS that hit **0.6871** local nDCG dropped hidden Track A from **0.5250 → 0.4627**.

---

## TREC-COVID (`beir/trec-covid`, `data/full`)

| | |
|---|---|
| docs | 171332 |
| avgdl | ~161.5 |
| queries | 50 |
| query length | ~6.1 tokens |

### Local (staff candidate pool)

| PARAMS | QL nDCG@10 | notes |
|---|---|---|
| Board / pre-max: `mu_factor=10`, `w_od=w_uw=0.05`, `lead_w=0.35`, `lead_len=12`, `len_prior=0.15`, `prior_w=0.60`, `prior_k=20`, `mu_max=2500` | 0.6696 | after first COVID knobs; earlier board cell was ~0.6523 |
| Unigram + stronger lead: `w_od=w_uw=0`, `lead_w=0.55`, `lead_len=16`, `len_prior=0.20`, `prior_w=0.50`, `mu_factor=14`, `mu_max=8000` | 0.6870 | SDM off helps this collection |
| Peak: `w_od=w_uw=0`, `lead_w=0.60`, `lead_len=16`, `len_prior=0.20`, `prior_w=0.50`, `prior_k=20`, `mu_factor=12`, `mu_max=8000`, `lambda=1` | **0.6871** | `lead_len` 20/24 loses ~0.01 |

Proximity (`w_od`/`w_uw` > 0) and longer lead windows were worse on this set.

### Hidden / live leaderboard (same held-out topics every submit)

| when | Track A nDCG | Track A MAP | Track B nDCG | Track C retention | final (percentile) | rank |
|---|---|---|---|---|---|---|
| `lambda=1`, pre-COVID-max PARAMS (`lead_w=0.35`, SDM 0.05, `mu_factor=10`, `prior_w=0.60`) | **0.5250** | — | 0.5250 | 1.0000 | 0.6625 | 1 (then) |
| `lambda` 0.80–0.95 (RM3 on) | — | — | 0.5199 | — | — | B still under 0.5241 |
| COVID-max PARAMS submitted ~21:44, 26 Sep | **0.4627** | 0.3784 | 0.4627 | 1.0000 | 0.2625 | 5 / 6 |

Snapshot 21:47: Bansal 0.5315 / 0.4544 / 0.5241 / 0.9817 → 0.5750; boldfit 0.5035 / 0.4278 / 0.5035 / 1.0 → 0.5375; Kumar 0.5244 / 0.4484 / 0.5246 / 0.9040 → 0.4750.

Local COVID nDCG is **not** the hidden scale (0.69 vs ~0.53): staff pool and held-out topics differ. Overfitting the 50 local topics hurt Track A by ~0.06.

---

## FiQA (`beir/fiqa`, `data/fiqa`)

Finance forum answers. Different length, vocab, and query style from TREC-COVID. Candidate pool is BM25-generated (not staff). 648 queries.

| | |
|---|---|
| docs | 57638 |
| avgdl | ~130.2 |
| queries | 648 |
| query length | ~5.5 tokens |

### Local (BM25 candidate pool)

| PARAMS | QL nDCG@10 | notes |
|---|---|---|
| COVID-max (unigram, `lead_w=0.60`, `mu_factor=12`, `len_prior=0.20`) | 0.2481 | starting point after COVID tune |
| Coarse: SDM on, `mu_factor=4`, moderate lead | 0.2690 | `w_od=w_uw=0.1`, `lead_w=0.2`, `prior_w=0.3` |
| Peak: `w_od=w_uw=0.10`, `uw_window=8`, `lead_w=0.40`, `lead_len=16`, `len_prior=0`, `prior_w=0.30`, `prior_k=10`, `mu_factor=1`, `mu_min=150`, `lambda=1` | **0.2764** | live in `submission/feedback.py` |

FiQA wants the opposite of COVID: **SDM on**, **small μ** (`μ ≈ avgdl`, clamped at 150), **no length prior**, weaker rank prior. Same PARAMS on local TREC-COVID: **0.6451** (vs COVID-max 0.6871). That is closer to the ~0.65 local cell that had hidden A **0.5250** than the 0.687 overfit that had hidden A **0.4627**.

These knobs are a FiQA local max, not a proven hidden win. Zip yourself if you submit.
