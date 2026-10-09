#!/usr/bin/env bash
# Build the student submission zip (repo layout, no corpora / build junk).
#
#   bash scripts/pack_submission.sh
#   bash scripts/pack_submission.sh /path/to/2023CSXXXXX.zip
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/COL764-Assignment2-submission.zip}"
OUT="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$OUT")"
cd "$ROOT"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

copy_tree() {
  local src="$1"
  if [[ -d "$src" ]]; then
    mkdir -p "$STAGE/$src"
    rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
          --exclude '_build' --exclude '*.so' "$src/" "$STAGE/$src/"
  elif [[ -f "$src" ]]; then
    mkdir -p "$STAGE/$(dirname "$src")"
    cp "$src" "$STAGE/$src"
  fi
}

# Required entrypoint + C++ engine + README + optional bonus set
copy_tree submission
# Hand-check demo referenced from README.md
copy_tree tools/__init__.py
copy_tree tools/verify_rm.py

# Starter repo the grader expects (same layout as GitHub)
copy_tree harness
copy_tree tests
copy_tree docs
copy_tree scripts
copy_tree data/README.md
copy_tree data/toy
copy_tree README.md
copy_tree requirements.txt
copy_tree conftest.py
copy_tree .gitignore
copy_tree .github

# Do not ship: data/full and other corpora, .venv, .git, _build, *.so,
# the tuning scripts in tools/, tune logs, assignment PDF/tex, NOTES, runs/

rm -f "$STAGE/scripts/pack_submission.sh"
for f in submission/feedback.py README.md; do
  if [[ ! -f "$STAGE/$f" ]]; then
    echo "error: $f missing" >&2
    exit 1
  fi
done

rm -f "$OUT"
( cd "$STAGE" && zip -r -q "$OUT" . )
echo "Wrote $OUT ($(du -h "$OUT" | awk '{print $1}'))"
unzip -l "$OUT" | grep -E 'submission/(feedback\.py|adversarial_set\.json|cpp/rm_engine\.cpp)|(^|/)README\.md|tools/verify_rm\.py' 
echo "files: $(unzip -l "$OUT" | tail -1)"
