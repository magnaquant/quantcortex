#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${POETRY_BIN:-}" ]]; then
  poetry_bin="${POETRY_BIN}"
elif command -v poetry >/dev/null 2>&1; then
  poetry_bin="$(command -v poetry)"
elif [[ -x "${repo_root}/.venv/bin/poetry" ]]; then
  poetry_bin="${repo_root}/.venv/bin/poetry"
else
  printf '%s\n' "Poetry is required; install it or set POETRY_BIN" >&2
  exit 1
fi

cd "${repo_root}"
"${poetry_bin}" lock
"${poetry_bin}" export --only main --without-hashes --output requirements/core.lock
"${poetry_bin}" export --only main,test --without-hashes --output requirements/test.lock
"${poetry_bin}" export --only main,dev --without-hashes --output requirements/notebooks.lock
"${poetry_bin}" export --only main,test,dev --without-hashes --output requirements/dev.lock
"${poetry_bin}" export --only main -E brokers --without-hashes --output requirements/brokers.lock
"${poetry_bin}" export --only main -E brokers -E providers -E storage -E regime \
  --without-hashes --output requirements/runtime.lock
"${poetry_bin}" export --only build --without-hashes --output requirements/build.lock
"${poetry_bin}" check
