"""
harness/candidates_io.py -- reading the provided top-K candidate list
every submission reranks (spec Section 6). Deliberately
dependency-free, same conventions as harness/trec_io.py.

Format (candidates_dev.jsonl): one JSON object per line,
    {"qid": "q1", "candidates": [["d18", 12.34], ["d16", 9.87], ...]}
already sorted best-first by whatever first-pass retriever produced them.
The exact retriever and its parameters are NOT disclosed (spec
Section 6, "Where the provided candidates come from") -- only its output
for the released dev/toy topics is. The held-out set's candidates file is
never released at all; course staff apply the same harness against it
internally at grading time.
"""
import json
from typing import Dict, List, Tuple


def read_candidates(path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Returns {qid: [(doc_id, score), ...]}, in file order (already
    best-first; not re-sorted here, unlike read_run() in trec_io.py,
    since this is a read-only input file you never write yourself)."""
    candidates: Dict[str, List[Tuple[str, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            candidates[obj["qid"]] = [(doc_id, float(score)) for doc_id, score in obj["candidates"]]
    return candidates


def write_candidates(path: str, candidates: Dict[str, List[Tuple[str, float]]]) -> None:
    """Instructor-side convenience (not needed by students -- they only
    ever read this file). Kept here rather than instructor-tools-only so
    the format is defined in exactly one place."""
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in candidates.items():
            f.write(json.dumps({"qid": qid, "candidates": [[doc_id, score] for doc_id, score in ranked]}) + "\n")
