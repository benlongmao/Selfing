#!/usr/bin/env bash
# Deprecated path: forwards to the canonical installer at the repo root.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/install_s_project.sh" "$@"
