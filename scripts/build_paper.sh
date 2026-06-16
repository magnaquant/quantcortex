#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/tmp/pdfs/build"
output_dir="${repo_root}/output/pdf"

command -v tectonic >/dev/null 2>&1 || {
  printf '%s\n' "tectonic is required to build the paper" >&2
  exit 1
}

# Fix PDF timestamps to the experiment vintage so identical sources produce
# byte-identical publication artifacts across machines and rebuilds.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1781568000}"

mkdir -p "${build_dir}" "${output_dir}"
(
  cd "${repo_root}/paper"
  tectonic --keep-logs --keep-intermediates \
    --outdir "${build_dir}" main.tex
)
cp "${build_dir}/main.pdf" \
  "${output_dir}/quantcortex_audit_neurips2026.pdf"
cp "${build_dir}/main.pdf" \
  "${repo_root}/paper/quantcortex_audit_neurips2026.pdf"
if command -v sha256sum >/dev/null 2>&1; then
  paper_hash="$(sha256sum "${repo_root}/paper/quantcortex_audit_neurips2026.pdf" | awk '{print $1}')"
else
  paper_hash="$(shasum -a 256 "${repo_root}/paper/quantcortex_audit_neurips2026.pdf" | awk '{print $1}')"
fi
printf '%s  %s\n' "${paper_hash}" "quantcortex_audit_neurips2026.pdf" \
  > "${repo_root}/paper/quantcortex_audit_neurips2026.sha256"

printf '%s\n' "built ${output_dir}/quantcortex_audit_neurips2026.pdf"
