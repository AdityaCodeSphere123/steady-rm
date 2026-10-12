"""
harness/run_harness.py -- runs your submission end-to-end against a
corpus/queries/qrels/candidates set and reports metrics, exactly like the
grading harness (minus the undisclosed Track C noise recipe -- see
harness/noise_injection.py -- and minus the undisclosed reference
retriever that generated the candidates file -- see the spec,
Section 6).

Unlike the earlier project, this is a SINGLE IN-PROCESS SCRIPT -- prepare(),
score_candidates(), and relevance_model_feedback() are all called
directly in this process, with no build/load subprocess split and no
Docker image (spec Section 4.2). You are also never asked to rank
the whole corpus: every retrieval call here is scoped to the provided
candidate pool for that query (typically ~12-100 documents), never all of
`data/toy/corpus.jsonl` or `data/full/corpus.jsonl`.

What this script does, per query:
  1. Loads that query's candidate pool from the candidates file.
  2. Calls score_candidates(query, candidate_doc_ids, k) -- scored
     directly as Track A.
  3. Takes score_candidates(query, candidate_doc_ids, PRF_DEPTH)'s top
     doc_ids as the "clean" pseudo-relevant seed and calls
     relevance_model_feedback(query, clean_ids, candidate_doc_ids, k) --
     scored as Track B (at 0% noise).
  4. For each PUBLIC practice noise level in harness.noise_injection
     (0.25, 0.5), perturbs the clean SEED (never the candidate pool
     itself) and calls relevance_model_feedback() again -- gives you a
     *practice* Track C signal locally. The real grading noise recipe is
     undisclosed; do not assume it matches these levels or this
     perturbation strategy.

Usage:
    python -m harness.run_harness \\
        --corpus data/toy/corpus.jsonl \\
        --queries data/toy/queries_dev.tsv \\
        --qrels data/toy/qrels_dev.txt \\
        --candidates data/toy/candidates_dev.jsonl \\
        --run-out runs/dev_run.trec \\
        --report-out runs/dev_report.json
"""
import argparse
import inspect
import json
import os
import time
from typing import List, Optional, Set, Tuple

from harness import leaderboard, noise_injection
from harness.candidates_io import read_candidates
from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries, write_run

PRF_DEPTH = 10  # how many docs from score_candidates() seed the pseudo-relevant set


def check_conformance(submission_module) -> List[str]:
    """Returns a list of human-readable problems (empty list = conformant)."""
    problems = []
    for name, expected_params in [
        ("prepare", ["corpus_path"]),
        ("score_candidates", ["query", "candidate_doc_ids", "k"]),
        ("relevance_model_feedback", ["query", "pseudo_relevant_doc_ids", "candidate_doc_ids", "k"]),
    ]:
        fn = getattr(submission_module, name, None)
        if fn is None or not callable(fn):
            problems.append(f"submission.feedback.{name} is missing or not callable")
            continue
        params = list(inspect.signature(fn).parameters.keys())
        if params != expected_params:
            problems.append(
                f"submission.feedback.{name} has parameters {params}, expected {expected_params}"
            )
    return problems


def _validate_and_sort_results(
    results: List[Tuple[str, float]],
    qid: str,
    k: int,
    valid_doc_ids: Optional[Set[str]] = None,
) -> List[Tuple[str, float]]:
    """Anti-gaming + sanity check applied to every retrieve-shaped call:
    rejects duplicate doc_ids (see the earlier project's harness/metrics.py for
    why this matters -- nDCG@10/MAP@10 are both unbounded above by
    construction, so a repeated known-relevant doc_id is a real exploit,
    not a cosmetic bug), enforces the k cap, and -- new this project --
    rejects any doc_id not present in `valid_doc_ids` when given: a
    reranker must only ever return documents it was actually handed as
    candidates, never invent or recall documents from outside the
    provided pool. Returns the list re-sorted by score descending
    (defense-in-depth against a submission that returns an unsorted but
    otherwise valid list)."""
    if len(results) > k:
        raise ValueError(f"qid={qid}: returned {len(results)} results, more than k={k}")
    doc_ids = [doc_id for doc_id, _score in results]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError(f"qid={qid}: returned a duplicate doc_id in {doc_ids!r}")
    if valid_doc_ids is not None:
        invalid = [d for d in doc_ids if d not in valid_doc_ids]
        if invalid:
            raise ValueError(
                f"qid={qid}: returned doc_id(s) {invalid!r} not present in the candidate "
                f"pool it was given -- a reranker may only return documents from its input."
            )
    return sorted(results, key=lambda pair: pair[1], reverse=True)


def run(
    corpus_path: str,
    queries_path: str,
    qrels_path: str,
    candidates_path: str,
    run_out: str = None,
    report_out: str = None,
    practice_noise: bool = True,
) -> dict:
    import submission.feedback as submission_module

    problems = check_conformance(submission_module)
    if problems:
        raise SystemExit("Conformance problems:\n  " + "\n  ".join(problems))

    queries = read_queries(queries_path)
    qrels = read_qrels(qrels_path)
    candidates_by_qid = read_candidates(candidates_path)

    t0 = time.perf_counter()
    submission_module.prepare(corpus_path)
    prepare_seconds = time.perf_counter() - t0

    ql_run, clean_fb_run = {}, {}
    noisy_fb_runs = {level: {} for level in noise_injection.PRACTICE_NOISE_LEVELS if level > 0.0}

    for qid, text in queries:
        if qid not in candidates_by_qid:
            continue  # no provided candidates for this query -- nothing to score
        candidate_doc_ids = [doc_id for doc_id, _score in candidates_by_qid[qid]]
        valid = set(candidate_doc_ids)

        ql_results = _validate_and_sort_results(
            submission_module.score_candidates(text, candidate_doc_ids, 10), qid, 10, valid
        )
        ql_run[qid] = ql_results

        seed_results = _validate_and_sort_results(
            submission_module.score_candidates(text, candidate_doc_ids, PRF_DEPTH), qid, PRF_DEPTH, valid
        )
        clean_ids = [doc_id for doc_id, _score in seed_results]

        clean_fb_results = _validate_and_sort_results(
            submission_module.relevance_model_feedback(text, clean_ids, candidate_doc_ids, 10), qid, 10, valid
        )
        clean_fb_run[qid] = clean_fb_results

        if practice_noise:
            seed = noise_injection.stable_seed(qid, base_seed=0)
            for level, perturbed_ids in noise_injection.sweep(clean_ids, candidate_doc_ids, seed=seed):
                if level == 0.0:
                    continue
                noisy_results = _validate_and_sort_results(
                    submission_module.relevance_model_feedback(text, perturbed_ids, candidate_doc_ids, 10),
                    qid, 10, valid,
                )
                noisy_fb_runs[level][qid] = noisy_results

    ql_metrics = evaluate_run(ql_run, qrels)
    clean_fb_metrics = evaluate_run(clean_fb_run, qrels)
    noisy_fb_ndcg_by_level = {
        level: evaluate_run(run_, qrels)["aggregate"]["ndcg@10"] for level, run_ in noisy_fb_runs.items()
    }

    summary = leaderboard.summarize(
        ql_metrics["aggregate"], clean_fb_metrics["aggregate"], noisy_fb_ndcg_by_level
    )
    summary["timing"] = {"prepare_seconds": prepare_seconds}

    if run_out:
        os.makedirs(os.path.dirname(run_out) or ".", exist_ok=True)
        write_run(run_out, ql_run, run_tag="score_candidates")
    if report_out:
        os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)
        with open(report_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--run-out", default=None)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--no-practice-noise", action="store_true")
    args = parser.parse_args()

    summary = run(
        args.corpus, args.queries, args.qrels, args.candidates,
        run_out=args.run_out, report_out=args.report_out,
        practice_noise=not args.no_practice_noise,
    )

    print("== Local sanity-check only -- NOT your real leaderboard score ==")
    print(f"score_candidates()   nDCG@10: {summary['score_candidates_ndcg@10']:.4f}   MAP@10: {summary['score_candidates_map@10']:.4f}")
    print(f"relevance_model_feedback() nDCG@10 @ 0% noise (clean): {summary['clean_feedback_ndcg@10']:.4f}")
    for level, ndcg in sorted(summary["noisy_feedback_ndcg@10_by_level"].items()):
        print(f"relevance_model_feedback() nDCG@10 @ {level:.0%} practice noise: {ndcg:.4f}")
    if summary["practice_retention_ratio"] is not None:
        print(f"Practice retention ratio (noisy / clean): {summary['practice_retention_ratio']:.4f}")
    print()
    print(summary["note"])


if __name__ == "__main__":
    main()
