"""
tests/test_fuzz.py -- hostile inputs; the engine must never crash and must
always return a conformant ranking.

  python -m tests.test_fuzz
  # memory/UB check (dev machine):
  RM_EXTRA_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  ASAN_OPTIONS=detect_leaks=0 LD_PRELOAD=$(g++ -print-file-name=libasan.so) python -m tests.test_fuzz
"""
import json, os, random, string, tempfile

from submission import feedback as fb

WORDS = ["cat", "dog", "virus", "covid", "mask", "the", "of", "U.S.", "e-mail", "naïve", "café",
         "日本", "x", "a", "123", "4g", "α-helix", "don't", "C++", "\\u00e9t\\u00e9"]


def rand_text(r):
    n = r.choice([0, 1, 3, 20, 200])
    parts = [r.choice(WORDS) if r.random() < 0.8 else "".join(r.choice(string.printable) for _ in range(r.randint(1, 12)))
             for _ in range(n)]
    if r.random() < 0.05:
        parts.append("z" * 1000)            # absurdly long token
    return " ".join(parts)


def main():
    r = random.Random(7)
    d = tempfile.mkdtemp()
    path = os.path.join(d, "corpus.jsonl")
    ids = []
    with open(path, "w", encoding="utf-8") as f:
        for i in range(400):
            did = f"doc{i}" if r.random() < 0.95 else r.choice(["dup", "doc1"])   # duplicate ids
            ids.append(did)
            obj = {"text": rand_text(r), "doc_id": did} if r.random() < 0.5 else {"doc_id": did, "title": "t", "text": rand_text(r)}
            if r.random() < 0.02:
                obj.pop("text")                  # missing text field
            line = json.dumps(obj, ensure_ascii=r.random() < 0.5)
            f.write(line + ("\r\n" if r.random() < 0.1 else "\n"))
            if r.random() < 0.03:
                f.write("\n   \n")                 # blank lines
        f.write('{"doc_id": 999, "text": "numeric id cat"}\n')
        f.write('{"broken json\n')
    fb.prepare(d)           # directory form
    fb.prepare(path)        # file form, idempotent
    universe = list(set(ids)) + ["999", "unknown1", "", "doc1"]
    queries = ["", "   ", "!!!", "the of a", "cat", "cat dog cat dog", " ".join(r.choice(WORDS) for _ in range(300)),
               "zzzzz qqqqq", "U.S. covid-19 mask", "日本 café"]
    n = 0
    for _ in range(600):
        q = r.choice(queries) if r.random() < 0.7 else rand_text(r)
        pool = [r.choice(universe) for _ in range(r.choice([0, 1, 5, 100, 300]))]
        seed = [r.choice(universe) for _ in range(r.choice([0, 1, 3, 10, 50]))]
        k = r.choice([-1, 0, 1, 10, 1000])
        valid = set(pool)
        for res in (fb.score_candidates(q, pool, k), fb.relevance_model_feedback(q, seed, pool, k)):
            got = [x for x, _ in res]
            assert len(got) == len(set(got)), "duplicate ids"
            assert set(got) <= valid, "id outside the pool"
            assert len(got) == max(0, min(k, len(valid))), (len(got), k, len(valid))
            sc = [s for _, s in res]
            assert all(a >= b for a, b in zip(sc, sc[1:])), "not sorted"
            assert all(s == s for s in sc), "NaN score"
            n += 1
    # determinism
    a = fb.relevance_model_feedback("cat dog", ["doc1", "doc2"], ids[:100], 10)
    assert a == fb.relevance_model_feedback("cat dog", ["doc1", "doc2"], ids[:100], 10)
    print(f"fuzz ok: {n} calls, stats={fb.stats()}")


if __name__ == "__main__":
    main()
