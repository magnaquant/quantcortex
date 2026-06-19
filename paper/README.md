# Research Paper

`main.tex` is a NeurIPS 2026-format public preprint on executable evaluation
contracts for target-weight trading pipelines. It presents exact return
attribution, a retrospective negative case, and a repository-frozen expansion
across four strategy families and two real-data panels. The expansion estimates
five one-switch effects with joint block resampling and reports every frozen
learned-model seed.
`anonymous.tex` builds the same preprint without author or repository identifiers.
The work is not represented as accepted by or submitted to NeurIPS 2026; the
full-paper deadline was May 6, 2026.

The style file and checklist were obtained from the official
[NeurIPS 2026 formatting package](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
without modification to the style implementation or checklist questions.

Build from the repository root:

```bash
scripts/build_paper.sh
```

The verified public PDF is written to
`output/pdf/quantcortex_audit_neurips2026.pdf` and copied to the tracked
`paper/quantcortex_audit_neurips2026.pdf` publication artifact. The anonymous
build is `paper/quantcortex_audit_anonymous.pdf`. Tectonic 0.16.9 is required;
the build fails on another version. When Poppler's `pdftotext` is available, it
checks that body text does not spill past the nine-page NeurIPS limit;
`pdftoppm` is used for visual QA. The build updates
both PDF checksum files and `build_manifest.json`.
It also writes `quantcortex_audit_neurips2026.sources.sha256`, which binds the
tracked PDF to the current LaTeX, bibliography, generated values, and figures.
Reviewed host details and measured wrapper wall times are in
[`COMPUTE.md`](COMPUTE.md).

Release the fixed experiment from a committed source revision with an
authorized local adjusted-close matrix:

```bash
scripts/release_paper_artifacts.sh \
  local_data/published_rotation_prices.csv
```

The input must contain `QQQ`, `VGT`, `GLD`, `TLT`, `SPY`, `VIG`, and `SHV` and
must match the SHA-256 digest in `results/manifest.json`; the release wrapper
rejects another matrix rather than attaching the reviewed provenance to it. It
uses the recorded source commit and timestamp when all release-critical paths
are unchanged, so the final artifact commit can reproduce without timestamp-only
drift. Changed experiment, report, or paper source must set
`QUANTCORTEX_GENERATED_AT` explicitly; that creates a release from current
`HEAD`. The wrapper requires committed tracked source, regenerates the reviewed
`docs/img/` gallery and paper experiment in a detached clean worktree with the
input mounted outside it, verifies the recorded source commit and clean start
state, then copies reviewed artifacts back. The
fixed experiment requires complete rows, performs no forward fill, and rejects
fewer than 274 pre-evaluation sessions. Raw provider data is not committed.
Aggregate tables, generated LaTeX values, figures, the explicit experiment
design, package source-tree fingerprints, dependency lock, source metadata,
configuration hash, clean source revision, package and thread-library versions,
and artifact hashes are committed in the manifest. `return_decomposition.csv` records the exact
allocation, exposure-timing, passive-exposure, cost, and net-cash components.
`ablation_uncertainty.csv` records joint 21-session block-bootstrap intervals
for every named overlay variant.
`protocol_switches.csv` records the audited result beside one-assumption
diagnostics. The primary 21-session joint block bootstrap is accompanied by 5-
and 63-session sensitivity results. `sharpe_uncertainty.csv` directly resamples
the conventional sample Sharpe statistic. `comparator_diagnostics.csv` records
the causal target-exposure comparator after its own costs, while
`evaluation_contract.json` records the machine-readable semantics.
`target_tape_hashes.json` records the canonical payload hash, decision count,
record count, and symbol set for every audited strategy variant without
publishing the underlying provider matrix.

Release the expansion from the same clean source revision and its two local
panel matrices:

```bash
QUANTCORTEX_EXPANSION_GENERATED_AT=2026-06-18T22:38:40Z \
  scripts/release_expansion_artifacts.sh local_data/expansion
```

The directory must contain `us_sector_etfs.csv`,
`country_equity_etfs.csv`, and their `.metadata.json` sidecars. The wrapper
rejects a protocol or input hash mismatch, runs in a detached worktree, and
publishes only aggregate CSV/JSON, generated LaTeX, and figures under
`paper/expansion/`. The paper wrapper requires these artifacts to name the same
clean source commit. A changed release must use explicit, fixed UTC timestamps
for both wrappers; unchanged reviewed source reuses the recorded timestamps.

The primary accounting path is the event-driven engine. It holds explicit
adjusted-close pseudo-shares between rebalances, sizes targets against post-cost
NAV, and reports both one-way turnover and gross two-sided traded notional. The
vectorized engine remains an approximation and model-convention sensitivity
diagnostic; equality with pseudo-share accounting is not expected.

Citation keys are checked against `references.bib`, and DOI/arXiv identifiers
are kept explicit. Revalidate the unversioned 2026 preprints before any future
submission because their metadata and claims may change.

The three absent raw matrices are a material reproducibility limitation. The
open target tape, schemas, and synthetic conformance fixtures reproduce software
semantics, not the historical returns. Exact independent reproduction still
requires access to matrices matching the manifests' digests. Data-source
acceptance rules are in `docs/data-source-due-diligence.md`; the expansion
protocol in `paper/preregistration.md` was repository-frozen before retrieval
but was not externally registered.

## Submission Readiness

The tracked public PDF is an identified preprint, not a conference submission.
The generated anonymized preprint removes direct author and repository
identifiers, but venue-specific rules still require a fresh review and style
change before submission. The
NeurIPS 2026 [main](https://neurips.cc/Conferences/2026/CallForPapers)
and [Evaluations & Datasets](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)
deadlines passed on May 6, 2026. No 2027 track is assumed. For any venue that
requires reviewer-reproducible empirical evidence, the current
non-redistributed input is a release blocker:
reviewers must receive accessible, properly permitted data without requesting
it from the author. Resolve that requirement with a redistributable or
reviewer-accessible licensed snapshot before describing this artifact as
submission-ready.

Do not edit generated tables or figures by hand. Change the experiment driver,
run its focused tests, regenerate all outputs, inspect the diffs, then rebuild
and visually review every PDF page.
