#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/tmp/pdfs/build"
output_dir="${repo_root}/output/pdf"
paper_dir="${repo_root}/paper"

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

# Fix PDF timestamps to the experiment vintage so repeated builds with the
# same Tectonic engine and package bundle are byte-identical.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1781568000}"

rm -rf "${build_dir}"
mkdir -p "${build_dir}" "${output_dir}"
(
  cd "${paper_dir}"
  tectonic --keep-logs --keep-intermediates \
    --outdir "${build_dir}" main.tex
)
cp "${build_dir}/main.pdf" \
  "${output_dir}/quantcortex_audit_neurips2026.pdf"
cp "${build_dir}/main.pdf" \
  "${repo_root}/paper/quantcortex_audit_neurips2026.pdf"

if command -v pdftotext >/dev/null 2>&1; then
  first_nine="$(pdftotext -f 1 -l 9 -layout "${build_dir}/main.pdf" -)"
  page_ten="$(pdftotext -f 10 -l 10 -layout "${build_dir}/main.pdf" -)"
  if printf '%s\n' "${first_nine}" | grep -Eq '^[[:space:]]*References[[:space:]]*$'; then
    :
  elif printf '%s\n' "${page_ten}" | grep -Eq '^[[:space:]]*References[[:space:]]*$'; then
    first_page_ten_text="$({
      printf '%s\n' "${page_ten}" | awk '
        {
          gsub(/\f/, "")
          if ($0 ~ /^[[:space:]]*$/) next
          if ($0 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/) next
          sub(/^[[:space:]]+/, "")
          sub(/[[:space:]]+$/, "")
          print
          exit
        }
      '
    })"
    if [[ "${first_page_ten_text}" != "References" ]]; then
      printf '%s\n' \
        "paper body exceeds the nine-page NeurIPS content limit" >&2
      exit 1
    fi
  else
    printf '%s\n' \
      "references begin after page 10; paper body exceeds nine pages" >&2
    exit 1
  fi
else
  printf '%s\n' \
    "warning: pdftotext unavailable; skipped nine-page body-limit check" >&2
fi

paper_hash="$(sha256_file "${paper_dir}/quantcortex_audit_neurips2026.pdf")"
printf '%s  %s\n' "${paper_hash}" "quantcortex_audit_neurips2026.pdf" \
  > "${paper_dir}/quantcortex_audit_neurips2026.sha256"

source_manifest="${paper_dir}/quantcortex_audit_neurips2026.sources.sha256"
source_manifest_tmp="${source_manifest}.tmp"
source_files=(
  "main.tex"
  "checklist.tex"
  "references.bib"
  "neurips_2026.sty"
  "results/generated_values.tex"
  "figures/accounting_summary.pdf"
  "figures/audit_protocol.pdf"
  "figures/bootstrap_robustness.pdf"
  "figures/engine_comparison.pdf"
  "figures/return_attribution_and_protocol_switches.pdf"
  "figures/sensitivity_and_ablation.pdf"
)
: > "${source_manifest_tmp}"
for relative_path in "${source_files[@]}"; do
  printf '%s  %s\n' \
    "$(sha256_file "${paper_dir}/${relative_path}")" \
    "${relative_path}" >> "${source_manifest_tmp}"
done
mv "${source_manifest_tmp}" "${source_manifest}"

printf '%s\n' "built ${output_dir}/quantcortex_audit_neurips2026.pdf"
