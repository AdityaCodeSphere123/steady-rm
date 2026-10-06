# A2 C++ engine — quick reference

* `submission/feedback.py` is the entrypoint; the engine is `submission/cpp/`.
  The model and its parameters are described in the `feedback.py` docstring
  and in `cpp/rm_engine.h` (one commented field per parameter).
* First import compiles `cpp/*.cpp` with g++ into `submission/_build/` (about
  10 s, once; hash-named so edits rebuild). Never commit `_build/`.
* If g++ or loading fails, importing `submission.feedback` raises.
* `fb.set_params(name=value, ...)` overrides any parameter;
  `fb.set_params(**fb.PARAMS)` restores the submitted configuration.
* Required baseline (multinomial Dirichlet QL + RM3 in the LM):
  `smoothing=0, mu=1000, w_od=0, w_uw=0, lead_w=0, fb_mode=0, lambda=0.5`.

## Tests
* `python -m pytest -q` — QL, RM1, RM2, RM3 checked against the textbook
  formulas on a 5-doc corpus, plus conformance, call-order independence and
  `lambda = 1` reproducing `score_candidates` exactly.
* `python -m tests.test_fuzz` — hostile inputs (bad JSON, duplicate ids,
  unicode, empty/300-term queries, unknown ids, k = -1/0/1000). The docstring
  gives the AddressSanitizer / ThreadSanitizer command.

## Tools (development only, not shipped)
* `tools/sweep.py` — readers for queries/qrels/candidates, nDCG@10, and a
  CLI grid over pool / out-of-pool / hard noise conditions.
* `tools/track_a_sweep.py` — Track A grid over the local collections.
* `tools/gain_sweep.py` — B − A gain for a grid of feedback settings.
* `tools/c1_check.py`, `tools/c2_check.py` — retention under random noise
  and under hard non-relevant seed swaps.
* `tools/make_adversarial.py --root data/full --n 5` — writes
  `submission/adversarial_set.json` for the bonus track.
* `tools/fetch_datasets.py` — downloads the BEIR collections used above.
