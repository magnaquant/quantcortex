# Research Paper

`main.tex` is a NeurIPS 2026-format preprint on executable evaluation contracts
for target-weight trading pipelines. It presents exact return attribution,
single-assumption diagnostics, and a fixed negative case study with
uncertainty-aware ablations and block-length sensitivity. It uses the official
`neurips_2026.sty` with the `preprint` option. The work is not represented as
accepted by or submitted to NeurIPS 2026; the full-paper deadline was May 6,
2026.

The style file and checklist were obtained from the official
[NeurIPS 2026 formatting package](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
without modification to the style implementation or checklist questions.

Build from the repository root:

```bash
scripts/build_paper.sh
```

The verified PDF is written to
`output/pdf/quantcortex_audit_neurips2026.pdf` and copied to the tracked
`paper/quantcortex_audit_neurips2026.pdf` publication artifact. Tectonic is
required for the build. When Poppler's `pdftotext` is available, the build also
checks that body text does not spill past the nine-page NeurIPS limit;
`pdftoppm` is used for visual QA. The build updates
`quantcortex_audit_neurips2026.sha256`.
It also writes `quantcortex_audit_neurips2026.sources.sha256`, which binds the
tracked PDF to the current LaTeX, bibliography, generated values, and figures.

Reproduce the fixed experiment with an authorized local adjusted-close matrix:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_paper_experiments.py \
  --prices-csv local_data/published_rotation_prices.csv \
  --cash-proxy-symbol SHV \
  --output-dir paper --bootstrap-replications 5000 \
  --data-provider 'Yahoo Finance via yfinance 1.4.1' \
  --permission-basis \
    'Repository owner authorizes publication of derived aggregate results; provider terms not independently verified' \
  --retrieved-at 2026-06-16 \
  --adjustment-method 'yfinance adjusted close with auto_adjust=False'
```

The input must contain `QQQ`, `VGT`, `GLD`, `TLT`, `SPY`, `VIG`, and `SHV` and
must match the SHA-256 digest in `results/manifest.json` for exact
reproduction. The fixed experiment requires complete rows, performs no forward
fill, and rejects fewer than 274 pre-evaluation sessions. Raw provider data is
not committed. Aggregate tables, generated
LaTeX values, figures, the explicit experiment design, complete package
source-tree fingerprints, source metadata, base Git revision, worktree state,
package versions, and artifact hashes are committed under `paper/results/`,
`paper/figures/`, and the manifest. `return_decomposition.csv` records the exact
allocation, exposure-timing, passive-exposure, cost, and net-cash components.
`ablation_uncertainty.csv` records joint 21-session block-bootstrap intervals
for every named overlay variant.
`protocol_switches.csv` records the audited result beside one-assumption
diagnostics. The primary 21-session joint block bootstrap is accompanied by 5-
and 63-session sensitivity results.

The primary accounting path is the event-driven engine. It holds explicit
adjusted-close pseudo-shares between rebalances, sizes targets against post-cost
NAV, and reports both one-way turnover and gross two-sided traded notional. The
vectorized engine remains an approximation and parity diagnostic.

Citation keys are checked against `references.bib`, and DOI/arXiv identifiers
are kept explicit. Revalidate the unversioned 2026 preprints before any future
submission because their metadata and claims may change.

The absent raw matrix is a material reproducibility limitation. The committed
artifacts permit audit of the computation, but exact independent reproduction
still requires authorized access to a matrix matching the manifest digest.

## Submission Readiness

The tracked PDF is a public, identified preprint, not an anonymous conference
submission. The NeurIPS 2026 [main](https://neurips.cc/Conferences/2026/CallForPapers)
and [Evaluations & Datasets](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)
deadlines passed on May 6, 2026. A future double-blind submission must remove
author and repository identifiers from the review build. For the Evaluations &
Datasets track, the current non-redistributed input is also a release blocker:
reviewers must receive accessible, properly permitted data without requesting
it from the author. Resolve that requirement with a redistributable or
reviewer-accessible licensed snapshot before describing this artifact as
submission-ready.

Do not edit generated tables or figures by hand. Change the experiment driver,
run its focused tests, regenerate all outputs, inspect the diffs, then rebuild
and visually review every PDF page.
