#!/bin/bash
# Build an open-source-friendly zip: clone to a temp repo, strip sensitive/large paths from the index, then pack.
# Strip set matches scripts/build_open_source_archive.py (venv, .env, DB, workspace, models, etc.).
# docs/ is kept so README doc links still work after unzip.
# Output: s-open-source-YYYYMMDD.zip in repo root (unpacks as s-main/), with .env.example when possible.
#
# Usage:
#   ./scripts/build_open_source_archive.sh              # default: git archive, no .git (good for public release)
#   ./scripts/build_open_source_archive.sh --with-git   # full clone with .git (branch-ready after unzip)
#   ./scripts/build_open_source_archive.sh --help
#
# About "no .git": git archive exports the tree only; the result is not a git repo after unzip.
# For a working git checkout after unzip, use --with-git; output name ends with ...-with-git.zip.
#
# Security (--with-git):
#   The zip contains the full .git history. If secrets were ever committed, do not publish this zip;
#   use it only for trusted internal sandboxes. For public release, use the default (no .git).
#
# Important: archive content is the committed snapshot at HEAD, not uncommitted or untracked files.
# To ship latest working tree, commit first, or use the disk-copy script build_open_source_archive.py.
# Repo root is derived from this script path: $(dirname "$0")/.. (no hard-coded absolute paths).

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
SUFFIX=$(date +%Y%m%d)

WITH_GIT=0
for arg in "$@"; do
  case "$arg" in
    --with-git) WITH_GIT=1 ;;
    -h|--help)
      grep '^#' "$0" | head -n 42 | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg (use --help)" >&2
      exit 1
      ;;
  esac
done

ZIP_BASE="s-open-source-${SUFFIX}"
if [[ "$WITH_GIT" -eq 1 ]]; then
  ZIP_NAME="${ZIP_BASE}-with-git.zip"
else
  ZIP_NAME="${ZIP_BASE}.zip"
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

CLONE="$TMP_DIR/s-clean"
echo "Cloning repo to temp directory..."
git clone --no-hardlinks --single-branch "$ROOT" "$CLONE"
cd "$CLONE"
git config user.email "export@local"
git config user.name "Export"

echo "Removing sensitive/local paths from export index (temp clone only)..."

# Secrets / env (must not ship in open-source zip)
git rm --cached -f .env .env.local 2>/dev/null || true

# Virtualenvs
git rm -r --cached .venv venv ENV env 2>/dev/null || true

# Databases
git rm --cached -f data.db data.db-shm data.db-wal 2>/dev/null || true
shopt -s nullglob
for f in data.db.*; do
  git rm --cached -f "$f" 2>/dev/null || true
done
shopt -u nullglob

# Other tracked .db / .venv (grep+awk is fragile with spaces; wrap grep so pipefail does not abort)
git status -s | (grep -E '\.db|\.venv' || true) | awk '{print $2}' | while read -r p; do
  [ -n "$p" ] && git rm --cached -f -- "$p" 2>/dev/null || true
done

# Dirs/files aligned with the Python exporter (split loops: one missing path must not skip the rest)
for _strip in workspace models backups archive reports; do
  git rm -r --cached --ignore-unmatch -- "$_strip" 2>/dev/null || true
done
git rm --cached -f endogenous_state.json 2>/dev/null || true
git ls-files -z '*.jsonl' | xargs -0r git rm --cached -f 2>/dev/null || true

git commit -m "chore: strip local/sensitive paths for open-source export" --allow-empty --no-verify

if [[ "$WITH_GIT" -eq 1 ]]; then
  echo "Building full-repo zip with .git (see header: history leak risk)..."
  cd "$TMP_DIR"
  mv s-clean s-main
  if [ -f "$ROOT/.env.example" ]; then
    cp "$ROOT/.env.example" s-main/
  fi
  (cd "$TMP_DIR" && zip -rq "$ROOT/$ZIP_NAME" s-main)
  cd "$ROOT"
  echo "Wrote: $ROOT/$ZIP_NAME"
  echo "Unpack directory: s-main/ (includes .git; git status / branches work)"
  exit 0
fi

echo "Building archive (git archive, no .git)..."
git archive --format=zip --prefix=s-main/ -o "$TMP_DIR/archive.zip" HEAD
cd "$ROOT"

# If unzip exists and we have .env.example, merge it into the tree and re-zip
if command -v unzip >/dev/null 2>&1 && [ -f "$ROOT/.env.example" ]; then
  echo "Adding .env.example and repackaging..."
  mkdir -p "$TMP_DIR/extract"
  unzip -q "$TMP_DIR/archive.zip" -d "$TMP_DIR/extract"
  cp "$ROOT/.env.example" "$TMP_DIR/extract/s-main/"
  (cd "$TMP_DIR/extract" && zip -rq "$ROOT/$ZIP_NAME" s-main)
else
  # No unzip: start from git archive output. If .env.example is already in the repo, the archive already has it;
  # do not ZipFile('a') a duplicate name (some unpackers warn).
  if [ -f "$ROOT/.env.example" ]; then
    python3 -c "
import os
import shutil
import zipfile
from pathlib import Path
root = Path('$ROOT')
tmp = Path('$TMP_DIR')
out = root / '$ZIP_NAME'
key = 's-main/.env.example'
env_f = root / '.env.example'
shutil.copy2(tmp / 'archive.zip', out)
with zipfile.ZipFile(out, 'r') as zin:
    has_key = key in zin.namelist()
if not has_key:
    with zipfile.ZipFile(out, 'a', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(str(env_f), key)
    print('Added .env.example')
else:
    tmp_out = out.with_name(out.name + '.new')
    with zipfile.ZipFile(tmp / 'archive.zip', 'r') as zin:
        with zipfile.ZipFile(tmp_out, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == key:
                    continue
                zout.writestr(item, zin.read(item.filename))
            zout.write(str(env_f), key)
    os.replace(tmp_out, out)
    print('Replaced s-main/.env.example from repo root template')"
  else
    cp "$TMP_DIR/archive.zip" "$ROOT/$ZIP_NAME"
  fi
fi
echo "Wrote: $ROOT/$ZIP_NAME"
echo "Unpack directory: s-main/ (default has no .git; use --with-git for a git repo)"
