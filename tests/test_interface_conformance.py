"""
tests/test_interface_conformance.py -- the exact check the CI conformance
job runs on every push (spec Section 5, "Submission Interface &
Conformance Checking"). If this file passes, your submission has the
right shape for the grading harness to run it; it says nothing about
ranking quality.

Run it yourself any time with:
    pytest tests/test_interface_conformance.py -v
"""
import os
import time

import pytest

from harness.candidates_io import read_candidates
from harness.run_harness import _validate_and_sort_results, check_conformance
from harness.trec_io import read_queries
from submission import feedback as submission_module

TOY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "toy")
CORPUS_PATH = os.path.join(TOY_DIR, "corpus.jsonl")
QUERIES_PATH = os.path.join(TOY_DIR, "queries_dev.tsv")
CANDIDATES_PATH = os.path.join(TOY_DIR, "candidates_dev.jsonl")


def test_required_functions_exist_with_correct_signature():
    problems = check_conformance(submission_module)
    assert not problems, f"Interface conformance problems: {problems}"


def test_score_candidates_returns_well_formed_results_from_the_given_pool():
    submission_module.prepare(CORPUS_PATH)
    queries = read_queries(QUERIES_PATH)
    candidates_by_qid = read_candidates(CANDIDATES_PATH)

    for qid, text in queries:
        candidate_doc_ids = [doc_id for doc_id, _score in candidates_by_qid[qid]]
        valid = set(candidate_doc_ids)

        results = submission_module.score_candidates(text, candidate_doc_ids, 5)
        assert len(results) <= 5, f"qid={qid} returned more than k results"
        scores = [score for _doc_id, score in results]
        assert scores == sorted(scores, reverse=True), f"qid={qid} results not sorted descending"
        for doc_id, score in results:
            assert doc_id in valid, f"qid={qid} returned doc_id {doc_id!r} not in the candidate pool it was given"
            assert isinstance(score, (int, float)), f"qid={qid} score {score!r} is not numeric"


def test_relevance_model_feedback_returns_well_formed_results_from_the_candidate_pool():
    submission_module.prepare(CORPUS_PATH)
    queries = read_queries(QUERIES_PATH)
    candidates_by_qid = read_candidates(CANDIDATES_PATH)

    for qid, text in queries:
        candidate_doc_ids = [doc_id for doc_id, _score in candidates_by_qid[qid]]
        valid = set(candidate_doc_ids)
        # A harness-constructed pseudo-relevant seed -- deliberately NOT
        # this submission's own score_candidates() output, to mirror how
        # the real harness sometimes calls this function (see the
        # stronger call-order-independence check below).
        pseudo_relevant = candidate_doc_ids[:5]

        results = submission_module.relevance_model_feedback(text, pseudo_relevant, candidate_doc_ids, 5)
        assert len(results) <= 5, f"qid={qid} returned more than k results"
        for doc_id, score in results:
            assert doc_id in valid, f"qid={qid} returned doc_id {doc_id!r} not in the candidate pool it was given"
            assert isinstance(score, (int, float)), f"qid={qid} score {score!r} is not numeric"


def test_relevance_model_feedback_may_not_return_a_doc_id_outside_the_candidate_pool():
    """Anti-gaming check specific to this project's design (spec
    Section 5): a reranker must only ever return documents from the
    candidate_doc_ids it was given -- never a doc_id from
    pseudo_relevant_doc_ids that happens to fall outside candidate_doc_ids,
    and never a doc_id invented or recalled from elsewhere. The harness
    enforces this via _validate_and_sort_results()'s valid_doc_ids check;
    this test exercises that check directly with a deliberately-invalid
    result set."""
    with pytest.raises(ValueError, match="not present in the candidate"):
        _validate_and_sort_results(
            [("outside_the_pool", 5.0)], qid="q1", k=10, valid_doc_ids={"d1", "d2"}
        )


def test_relevance_model_feedback_does_not_require_prior_score_candidates_call():
    """The conformance check for project Section 5, "call-order
    independence": relevance_model_feedback() must work correctly when
    called directly for a query this process has never passed to
    score_candidates() at all -- the real harness sometimes does exactly
    this (module docstring of submission/feedback.py: "you have no
    legitimate way to tell" whether pseudo_relevant_doc_ids is your own
    clean top-k or a perturbed version). A submission that secretly
    caches "the real top-k" keyed by query text inside score_candidates()
    and reads that cache back inside relevance_model_feedback() (instead
    of trusting the pseudo_relevant_doc_ids argument it was actually
    given) would misbehave here.
    """
    submission_module.prepare(CORPUS_PATH)
    candidates_by_qid = read_candidates(CANDIDATES_PATH)
    candidate_doc_ids = [doc_id for doc_id, _score in next(iter(candidates_by_qid.values()))]
    pseudo_relevant = candidate_doc_ids[:5]

    # Note: no score_candidates() call for this query anywhere above.
    results = submission_module.relevance_model_feedback(
        "a query never seen before", pseudo_relevant, candidate_doc_ids, 5
    )
    assert len(results) <= 5
    for doc_id, _score in results:
        assert doc_id in candidate_doc_ids


def test_retrieve_shaped_call_may_not_return_a_duplicate_doc_id():
    # Anti-gaming check: see the earlier project's harness/metrics.py module
    # docstring, "A note on duplicate doc_ids" -- the same exploit
    # applies here (nDCG@10/MAP@10 are unbounded above by construction).
    with pytest.raises(ValueError, match="duplicate doc_id"):
        _validate_and_sort_results([("d1", 5.0), ("d1", 4.0)], qid="q1", k=10)


def test_prepare_and_retrieval_are_reasonably_fast_on_the_toy_set():
    """Not a correctness check -- just catches an accidentally slow
    per-candidate implementation before it becomes a problem on the real
    corpus's larger candidate pools. See docs/GRADING.md for the actual
    wall-clock budget grading enforces; this is a much looser local
    sanity threshold."""
    t0 = time.perf_counter()
    submission_module.prepare(CORPUS_PATH)
    prepare_time = time.perf_counter() - t0

    queries = read_queries(QUERIES_PATH)
    candidates_by_qid = read_candidates(CANDIDATES_PATH)
    latencies = []
    for qid, text in queries:
        candidate_doc_ids = [doc_id for doc_id, _score in candidates_by_qid[qid]]
        t0 = time.perf_counter()
        submission_module.score_candidates(text, candidate_doc_ids, 10)
        latencies.append(time.perf_counter() - t0)

    assert prepare_time < 30, "prepare() on a 20-document toy corpus took >30s -- something is off."
    assert max(latencies) < 5, "A single score_candidates() call on the toy candidate pool took >5s."
