#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
prices_argument="${1:-${repo_root}/local_data/published_rotation_prices.csv}"

if [[ ! -x "${python_bin}" ]]; then
  printf '%s\n' "Python environment not found: ${python_bin}" >&2
  exit 1
fi
if ! git -C "${repo_root}" diff --quiet || \
   ! git -C "${repo_root}" diff --cached --quiet; then
  printf '%s\n' \
    "commit tracked source changes before releasing paper artifacts" >&2
  exit 1
fi

prices_csv="$(
  "${python_bin}" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
    "${prices_argument}"
)"
if [[ ! -f "${prices_csv}" ]]; then
  printf '%s\n' "price matrix not found: ${prices_csv}" >&2
  exit 1
fi

reviewed_manifest="${repo_root}/paper/results/manifest.json"
if [[ ! -f "${reviewed_manifest}" ]]; then
  printf '%s\n' \
    "reviewed paper manifest not found: ${reviewed_manifest}" >&2
  exit 1
fi
expected_input_digest="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["source"]["input_sha256"])' \
    "${reviewed_manifest}"
)"
actual_input_digest="$(
  "${python_bin}" -c \
    'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "${prices_csv}"
)"
if [[ "${actual_input_digest}" != "${expected_input_digest}" ]]; then
  printf '%s\n' \
    "price matrix digest does not match the reviewed experiment" \
    "expected: ${expected_input_digest}" \
    "actual:   ${actual_input_digest}" >&2
  exit 1
fi
case "${prices_csv}" in
  "${repo_root}"/*)
    relative_prices="${prices_csv#${repo_root}/}"
    if git -C "${repo_root}" ls-files --error-unmatch -- "${relative_prices}" \
      >/dev/null 2>&1; then
      printf '%s\n' \
        "reviewed raw price input must not be tracked by Git: ${relative_prices}" >&2
      exit 1
    else
      tracking_status=$?
      if [[ "${tracking_status}" -ne 1 ]]; then
        printf '%s\n' \
          "could not determine whether the raw price input is tracked" >&2
        exit 1
      fi
    fi
    ;;
esac

current_commit="$(git -C "${repo_root}" rev-parse HEAD)"
reviewed_source_commit="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["generator"]["git"]["source_commit"])' \
    "${reviewed_manifest}"
)"
reviewed_generated_at="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["generated_at"])' \
    "${reviewed_manifest}"
)"
if [[ -n "${QUANTCORTEX_GENERATED_AT:-}" ]]; then
  source_commit="${current_commit}"
  generated_at="${QUANTCORTEX_GENERATED_AT}"
else
  if ! git -C "${repo_root}" cat-file -e "${reviewed_source_commit}^{commit}" \
    >/dev/null 2>&1; then
    printf '%s\n' "reviewed source commit is unavailable" >&2
    exit 1
  fi
  release_source_paths=(
    quantcortex
    schemas
    pyproject.toml
    poetry.lock
    scripts/build_paper.sh
    scripts/generate_report.py
    scripts/fetch_expansion_data.py
    scripts/release_expansion_artifacts.sh
    scripts/release_paper_artifacts.sh
    scripts/run_expansion_experiments.py
    scripts/run_paper_experiments.py
    paper/main.tex
    paper/anonymous.tex
    paper/checklist.tex
    paper/references.bib
    paper/neurips_2026.sty
    paper/preregistration.md
    paper/expansion/protocol.json
  )
  if ! git -C "${repo_root}" diff --quiet \
    "${reviewed_source_commit}" "${current_commit}" -- \
    "${release_source_paths[@]}"; then
    printf '%s\n' \
      "QUANTCORTEX_GENERATED_AT is required for changed release source" >&2
    exit 1
  fi
  source_commit="${reviewed_source_commit}"
  generated_at="${reviewed_generated_at}"
fi
source_date_epoch="${SOURCE_DATE_EPOCH:-$(git -C "${repo_root}" show -s --format=%ct "${source_commit}")}"
temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/quantcortex-paper-release.XXXXXX")"
source_worktree="${temporary_root}/source"
generated_output="${temporary_root}/generated"
performance_output="${temporary_root}/performance"

cleanup() {
  git -C "${repo_root}" worktree remove --force "${source_worktree}" \
    >/dev/null 2>&1 || true
  rm -rf "${temporary_root}"
}
trap cleanup EXIT

git -C "${repo_root}" worktree add --detach "${source_worktree}" "${source_commit}" \
  >/dev/null

provider="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["source"]["provider"])' \
    "${reviewed_manifest}"
)"
permission="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["source"]["permission_basis"])' \
    "${reviewed_manifest}"
)"
retrieved_at="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["source"]["retrieved_at"])' \
    "${reviewed_manifest}"
)"
adjustment="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["source"]["adjustment_method"])' \
    "${reviewed_manifest}"
)"

(
  cd "${source_worktree}"
  MPLCONFIGDIR="${temporary_root}/matplotlib" \
  PYTHONPATH="${source_worktree}" \
  "${python_bin}" scripts/run_paper_experiments.py \
    --prices-csv "${prices_csv}" \
    --cash-proxy-symbol SHV \
    --output-dir "${generated_output}" \
    --bootstrap-replications 5000 \
    --data-provider "${provider}" \
    --permission-basis "${permission}" \
    --retrieved-at "${retrieved_at}" \
    --adjustment-method "${adjustment}" \
    --generated-at "${generated_at}" \
    --require-clean-source
)

(
  cd "${source_worktree}"
  MPLCONFIGDIR="${temporary_root}/matplotlib-report" \
  PYTHONPATH="${source_worktree}" \
  "${python_bin}" scripts/generate_report.py \
    --prices-csv "${prices_csv}" \
    --cash-proxy-symbol SHV \
    --imgdir "${performance_output}/img" \
    --report-out "${performance_output}/report.md" \
    --manifest-out "${performance_output}/img/performance_manifest.json" \
    --data-provider "${provider}" \
    --permission-basis "${permission}" \
    --retrieved-at "${retrieved_at}" \
    --adjustment-method "${adjustment}" \
    --generated-at "${generated_at}" \
    --require-clean-source \
    >/dev/null
)

"${python_bin}" - \
  "${generated_output}/results/manifest.json" \
  "${source_commit}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_commit = sys.argv[2]
git = manifest["generator"]["git"]
if git["source_commit"] != expected_commit:
    raise SystemExit("manifest source commit does not match the release commit")
if git["worktree_clean_at_start"] is not True:
    raise SystemExit("manifest did not record a clean source worktree")
PY

"${python_bin}" - \
  "${performance_output}/img/performance_manifest.json" \
  "${source_commit}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_commit = sys.argv[2]
git = manifest["generator"]["git"]
if git["source_commit"] != expected_commit:
    raise SystemExit("performance manifest source commit does not match")
if git["worktree_clean_at_start"] is not True:
    raise SystemExit("performance manifest did not record a clean source worktree")
PY

rm -rf "${source_worktree}/paper/results" "${source_worktree}/paper/figures"
cp -R "${generated_output}/results" "${source_worktree}/paper/results"
cp -R "${generated_output}/figures" "${source_worktree}/paper/figures"

expansion_manifest="${repo_root}/paper/expansion/results/manifest.json"
if [[ ! -f "${expansion_manifest}" ]]; then
  printf '%s\n' "reviewed expansion manifest not found: ${expansion_manifest}" >&2
  exit 1
fi
expansion_source_commit="$(
  "${python_bin}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["git"]["source_commit"])' \
    "${expansion_manifest}"
)"
if [[ "${expansion_source_commit}" != "${source_commit}" ]]; then
  printf '%s\n' \
    "expansion artifacts do not match the paper source commit" \
    "paper source:     ${source_commit}" \
    "expansion source: ${expansion_source_commit}" >&2
  exit 1
fi
rm -rf "${source_worktree}/paper/expansion/results" \
       "${source_worktree}/paper/expansion/figures"
cp -R "${repo_root}/paper/expansion/results" \
  "${source_worktree}/paper/expansion/results"
cp -R "${repo_root}/paper/expansion/figures" \
  "${source_worktree}/paper/expansion/figures"

(
  cd "${source_worktree}"
  PYTHON_BIN="${python_bin}" \
  PAPER_SOURCE_COMMIT="${source_commit}" \
  SOURCE_DATE_EPOCH="${source_date_epoch}" \
  TECTONIC_EXPECTED_VERSION=0.16.9 \
    scripts/build_paper.sh
)

rm -rf "${repo_root}/paper/results" "${repo_root}/paper/figures"
cp -R "${source_worktree}/paper/results" "${repo_root}/paper/results"
cp -R "${source_worktree}/paper/figures" "${repo_root}/paper/figures"
cp "${source_worktree}/paper/quantcortex_audit_neurips2026.pdf" \
  "${repo_root}/paper/quantcortex_audit_neurips2026.pdf"
cp "${source_worktree}/paper/quantcortex_audit_neurips2026.sha256" \
  "${repo_root}/paper/quantcortex_audit_neurips2026.sha256"
cp "${source_worktree}/paper/quantcortex_audit_neurips2026.sources.sha256" \
  "${repo_root}/paper/quantcortex_audit_neurips2026.sources.sha256"
cp "${source_worktree}/paper/quantcortex_audit_anonymous.pdf" \
  "${repo_root}/paper/quantcortex_audit_anonymous.pdf"
cp "${source_worktree}/paper/quantcortex_audit_anonymous.sha256" \
  "${repo_root}/paper/quantcortex_audit_anonymous.sha256"
cp "${source_worktree}/paper/build_manifest.json" \
  "${repo_root}/paper/build_manifest.json"

rm -rf "${repo_root}/docs/img"
cp -R "${performance_output}/img" "${repo_root}/docs/img"

mkdir -p "${repo_root}/output/pdf"
cp "${source_worktree}/paper/quantcortex_audit_neurips2026.pdf" \
  "${repo_root}/output/pdf/quantcortex_audit_neurips2026.pdf"
cp "${source_worktree}/paper/quantcortex_audit_anonymous.pdf" \
  "${repo_root}/output/pdf/quantcortex_audit_anonymous.pdf"

printf '%s\n' "released paper artifacts from source commit ${source_commit}"
printf '%s\n' "manifest timestamp: ${generated_at}"
