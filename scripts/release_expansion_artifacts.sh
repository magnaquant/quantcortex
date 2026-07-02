#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
panel_argument="${1:-${repo_root}/local_data/expansion}"

if [[ ! -x "${python_bin}" ]]; then
  printf '%s\n' "Python environment not found: ${python_bin}" >&2
  exit 1
fi
# Generation runs from a detached worktree at the source commit, so only
# uncommitted changes to release-critical source can corrupt a release.
# Scoping the cleanliness check to those paths keeps the wrapper rerunnable
# while regenerated artifacts sit uncommitted in the working tree.
release_source_paths=(
  quantcortex
  schemas/canonical_target_tape.schema.json
  pyproject.toml
  poetry.lock
  paper/preregistration.md
  paper/expansion/protocol.json
  scripts/fetch_expansion_data.py
  scripts/release_expansion_artifacts.sh
  scripts/run_expansion_experiments.py
)
if ! git -C "${repo_root}" diff --quiet -- "${release_source_paths[@]}" || \
   ! git -C "${repo_root}" diff --cached --quiet -- "${release_source_paths[@]}"; then
  printf '%s\n' \
    "commit release-critical source changes before releasing expansion artifacts" >&2
  exit 1
fi

panel_dir="$(
  "${python_bin}" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
    "${panel_argument}"
)"
if [[ ! -d "${panel_dir}" ]]; then
  printf '%s\n' "expansion panel directory not found: ${panel_dir}" >&2
  exit 1
fi
for panel in us_sector_etfs country_equity_etfs; do
  for suffix in csv metadata.json; do
    if [[ ! -f "${panel_dir}/${panel}.${suffix}" ]]; then
      printf '%s\n' "missing expansion input: ${panel_dir}/${panel}.${suffix}" >&2
      exit 1
    fi
  done
done

current_commit="$(git -C "${repo_root}" rev-parse HEAD)"
reviewed_manifest="${repo_root}/paper/expansion/results/manifest.json"

if [[ -n "${QUANTCORTEX_EXPANSION_GENERATED_AT:-}" ]]; then
  source_commit="${current_commit}"
  generated_at="${QUANTCORTEX_EXPANSION_GENERATED_AT}"
elif [[ -f "${reviewed_manifest}" ]]; then
  reviewed_source_commit="$(
    "${python_bin}" -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["git"]["source_commit"])' \
      "${reviewed_manifest}"
  )"
  reviewed_generated_at="$(
    "${python_bin}" -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["generated_at"])' \
      "${reviewed_manifest}"
  )"
  if ! git -C "${repo_root}" cat-file -e "${reviewed_source_commit}^{commit}" \
    >/dev/null 2>&1; then
    printf '%s\n' "reviewed expansion source commit is unavailable" >&2
    exit 1
  fi
  if ! git -C "${repo_root}" diff --quiet \
    "${reviewed_source_commit}" "${current_commit}" -- \
    "${release_source_paths[@]}"; then
    printf '%s\n' \
      "QUANTCORTEX_EXPANSION_GENERATED_AT is required for changed release source" >&2
    exit 1
  fi
  source_commit="${reviewed_source_commit}"
  generated_at="${reviewed_generated_at}"
else
  printf '%s\n' \
    "QUANTCORTEX_EXPANSION_GENERATED_AT is required for the first release" >&2
  exit 1
fi

temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/quantcortex-expansion-release.XXXXXX")"
source_worktree="${temporary_root}/source"
generated_output="${temporary_root}/generated"

cleanup() {
  git -C "${repo_root}" worktree remove --force "${source_worktree}" \
    >/dev/null 2>&1 || true
  rm -rf "${temporary_root}"
}
trap cleanup EXIT

git -C "${repo_root}" worktree add --detach "${source_worktree}" "${source_commit}" \
  >/dev/null

(
  cd "${source_worktree}"
  MPLCONFIGDIR="${temporary_root}/matplotlib" \
  PYTHONPATH="${source_worktree}" \
  "${python_bin}" scripts/run_expansion_experiments.py \
    --protocol "${source_worktree}/paper/expansion/protocol.json" \
    --panel-dir "${panel_dir}" \
    --output-dir "${generated_output}" \
    --generated-at "${generated_at}"
)

"${python_bin}" - \
  "${generated_output}/results/manifest.json" \
  "${source_commit}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_commit = sys.argv[2]
git = manifest["git"]
if git["source_commit"] != expected_commit:
    raise SystemExit("manifest source commit does not match the release commit")
if git["tracked_worktree_clean_at_start"] is not True:
    raise SystemExit("manifest did not record a clean source worktree")
root = manifest_path.parents[1]
for relative, expected in manifest["artifacts"].items():
    artifact = root / relative
    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"artifact digest mismatch: {relative}")
PY

rm -rf "${repo_root}/paper/expansion/results" \
       "${repo_root}/paper/expansion/figures"
cp -R "${generated_output}/results" "${repo_root}/paper/expansion/results"
cp -R "${generated_output}/figures" "${repo_root}/paper/expansion/figures"

printf '%s\n' "released expansion artifacts from source commit ${source_commit}"
printf '%s\n' "manifest timestamp: ${generated_at}"
