# A2 C++ engine — quick reference

Drop-in: copy `submission/feedback.py`, `submission/_feedback_py.py`,
`submission/cpp/`, `submission/Makefile`, `tools/sweep.py`, `tests/test_engine.py`
into the repo (keep the repo's own `submission/__init__.py` etc.).

* First import compiles `cpp/*.cpp` with g++ into `submission/_build/` (≈10 s,
  once; hash-named so edits rebuild). Never commit `_build/`.
* If g++ or loading fails, `_feedback_py.py` (your original Python) is used.
* `python -m tests.test_engine` — QL, RM1, RM2, RM3 checked against textbook
  formulas on a 5-doc corpus + conformance + call-order independence.
* `python -m tools.sweep --corpus ... --queries ... --qrels ... --candidates ... --folds 2 --grid lambda=0.3,0.5,0.7`
  — tuning / report plots (nDCG@10 at 0%, 25%, 50%, 75%, out-of-pool, hard noise).
* `fb.explain_feedback(q, seed, pool)` — seed weights, expansion terms, λ.

## Tuning / bonus tools
* `python -m tools.tune --root data --rounds 2 --workers 4` — parallel coordinate
  ascent over ~20 parameters on every collection; prints PARAMS to paste.
* `python -m tools.make_adversarial --root data/full --n 5` — writes
  `submission/adversarial_set.json` for the round-robin bonus.
* `python -m tools.bench ...` with `RM_TIMING=1` — speed breakdown.

## Safety checks (all pass)
* `python -m tests.test_fuzz` — 1200 hostile calls (bad JSON, duplicate ids,
  unicode, empty/300-term queries, unknown ids, k = -1/0/1000).
* Same tests under AddressSanitizer + UBSan and ThreadSanitizer (4 threads):
  see the docstring of tests/test_fuzz.py for the command.
* Rankings are byte-identical with 1, 2 and 4 threads.
