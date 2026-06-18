#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/tmp/pdfs/build"
output_dir="${repo_root}/output/pdf"
paper_dir="${repo_root}/paper"
python_bin="${PYTHON_BIN:-python3}"
expected_tectonic_version="${TECTONIC_EXPECTED_VERSION:-0.16.9}"
expected_bundle_sha256="${TECTONIC_BUNDLE_SHA256:-6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c}"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

command -v tectonic >/dev/null 2>&1 || {
  printf '%s\n' "tectonic is required to build the paper" >&2
  exit 1
}
command -v "${python_bin}" >/dev/null 2>&1 || {
  printf '%s\n' "${python_bin} is required to write the build manifest" >&2
  exit 1
}

tectonic_version="$(tectonic --version | awk '{print $2}')"
if [[ "${tectonic_version}" != "${expected_tectonic_version}" ]]; then
  printf '%s\n' \
    "expected Tectonic ${expected_tectonic_version}, found ${tectonic_version}" >&2
  exit 1
fi

# Fix PDF timestamps to the experiment vintage so repeated builds with the
# same Tectonic engine and package bundle are byte-identical.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1781568000}"

rm -rf "${build_dir}"
mkdir -p "${build_dir}" "${output_dir}"
(
  cd "${paper_dir}"
  tectonic --keep-logs --keep-intermediates \
    --outdir "${build_dir}" main.tex
  tectonic --keep-logs --keep-intermediates \
    --outdir "${build_dir}" anonymous.tex
)
bundle_cache_dir="$(tectonic -X show user-cache-dir 2>/dev/null | tail -n 1)"
bundle_hash_file="$(
  find "${bundle_cache_dir}/hashes" -type f \
    -name '*default_bundle_v33.tar' -print -quit
)"
if [[ -z "${bundle_hash_file}" || ! -f "${bundle_hash_file}" ]]; then
  printf '%s\n' "could not locate the Tectonic default bundle digest" >&2
  exit 1
fi
bundle_sha256="$(tr -d '[:space:]' < "${bundle_hash_file}")"
if [[ "${bundle_sha256}" != "${expected_bundle_sha256}" ]]; then
  printf '%s\n' \
    "expected Tectonic bundle ${expected_bundle_sha256}, found ${bundle_sha256}" >&2
  exit 1
fi
cp "${build_dir}/main.pdf" \
  "${output_dir}/quantcortex_audit_neurips2026.pdf"
cp "${build_dir}/main.pdf" \
  "${repo_root}/paper/quantcortex_audit_neurips2026.pdf"
cp "${build_dir}/anonymous.pdf" \
  "${output_dir}/quantcortex_audit_anonymous.pdf"
cp "${build_dir}/anonymous.pdf" \
  "${repo_root}/paper/quantcortex_audit_anonymous.pdf"

check_body_limit() {
  local pdf_path="$1"
  local label="$2"
  local first_nine
  local page_ten
  local first_page_ten_text
  first_nine="$(pdftotext -f 1 -l 9 -layout "${pdf_path}" -)"
  page_ten="$(pdftotext -f 10 -l 10 -layout "${pdf_path}" -)"
  if printf '%s\n' "${first_nine}" | grep -Eq \
    '^[[:space:]]*([0-9]+[[:space:]]+)?References[[:space:]]*$'; then
    return
  elif printf '%s\n' "${page_ten}" | grep -Eq \
    '^[[:space:]]*([0-9]+[[:space:]]+)?References[[:space:]]*$'; then
    first_page_ten_text="$({
      printf '%s\n' "${page_ten}" | awk '
        {
          gsub(/\f/, "")
          if ($0 ~ /^[[:space:]]*$/) next
          if ($0 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/) next
          sub(/^[[:space:]]+/, "")
          sub(/[[:space:]]+$/, "")
          sub(/^[0-9]+[[:space:]]+/, "")
          print
          exit
        }
      '
    })"
    if [[ "${first_page_ten_text}" != "References" ]]; then
      printf '%s\n' \
        "${label} body exceeds the nine-page NeurIPS content limit" >&2
      exit 1
    fi
  else
    printf '%s\n' \
      "${label} references begin after page 10; body exceeds nine pages" >&2
    exit 1
  fi
}

if command -v pdftotext >/dev/null 2>&1; then
  check_body_limit "${build_dir}/main.pdf" "public paper"
  check_body_limit "${build_dir}/anonymous.pdf" "anonymous paper"
else
  printf '%s\n' \
    "warning: pdftotext unavailable; skipped nine-page body-limit check" >&2
fi

paper_hash="$(sha256_file "${paper_dir}/quantcortex_audit_neurips2026.pdf")"
printf '%s  %s\n' "${paper_hash}" "quantcortex_audit_neurips2026.pdf" \
  > "${paper_dir}/quantcortex_audit_neurips2026.sha256"
anonymous_hash="$(sha256_file "${paper_dir}/quantcortex_audit_anonymous.pdf")"
printf '%s  %s\n' "${anonymous_hash}" "quantcortex_audit_anonymous.pdf" \
  > "${paper_dir}/quantcortex_audit_anonymous.sha256"

source_manifest="${paper_dir}/quantcortex_audit_neurips2026.sources.sha256"
source_manifest_tmp="${source_manifest}.tmp"
source_files=(
  "main.tex"
  "anonymous.tex"
  "checklist.tex"
  "references.bib"
  "neurips_2026.sty"
  "preregistration.md"
  "results/generated_values.tex"
  "results/manifest.json"
  "expansion/protocol.json"
  "expansion/results/generated_values.tex"
  "expansion/results/manifest.json"
  "figures/accounting_summary.pdf"
  "figures/audit_protocol.pdf"
  "figures/bootstrap_robustness.pdf"
  "figures/engine_comparison.pdf"
  "figures/return_attribution_and_protocol_switches.pdf"
  "figures/sensitivity_and_ablation.pdf"
  "expansion/figures/baseline_performance.pdf"
  "expansion/figures/contract_effects_return.pdf"
  "expansion/figures/contract_effects_sharpe.pdf"
  "expansion/figures/engine_conformance.pdf"
  "expansion/figures/learned_seed_sensitivity.pdf"
)
: > "${source_manifest_tmp}"
for relative_path in "${source_files[@]}"; do
  printf '%s  %s\n' \
    "$(sha256_file "${paper_dir}/${relative_path}")" \
    "${relative_path}" >> "${source_manifest_tmp}"
done
mv "${source_manifest_tmp}" "${source_manifest}"

source_commit="${PAPER_SOURCE_COMMIT:-$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf unavailable)}"
build_manifest="${paper_dir}/build_manifest.json"
"${python_bin}" - \
  "${build_manifest}" \
  "${source_commit}" \
  "${SOURCE_DATE_EPOCH}" \
  "${tectonic_version}" \
  "${bundle_sha256}" \
  "${paper_hash}" \
  "${anonymous_hash}" \
  "$(sha256_file "${source_manifest}")" <<'PY'
import json
import sys
from pathlib import Path

(
    output_path,
    source_commit,
    source_date_epoch,
    tectonic_version,
    bundle_sha256,
    pdf_sha256,
    anonymous_pdf_sha256,
    source_manifest_sha256,
) = sys.argv[1:]
manifest = {
    "schema_version": 1,
    "source_commit": source_commit,
    "source_date_epoch": int(source_date_epoch),
    "tectonic_version": tectonic_version,
    "tectonic_bundle": {
        "name": "default_bundle_v33.tar",
        "sha256": bundle_sha256,
    },
    "pdf": {
        "path": "quantcortex_audit_neurips2026.pdf",
        "sha256": pdf_sha256,
    },
    "anonymous_pdf": {
        "path": "quantcortex_audit_anonymous.pdf",
        "sha256": anonymous_pdf_sha256,
    },
    "source_manifest": {
        "path": "quantcortex_audit_neurips2026.sources.sha256",
        "sha256": source_manifest_sha256,
    },
}
Path(output_path).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY

printf '%s\n' "built ${output_dir}/quantcortex_audit_neurips2026.pdf"
