# Reviewed Compute Record

This record describes the artifact release generated from source commit
`e0443b8f77cd23aee8f1fa64a2bc237e47626c47` with fixed timestamp
`2026-06-18T23:53:33Z`. It is a reproducibility note, not a performance
benchmark.

## Host

- Apple M1, 8 physical and 8 logical CPU cores, arm64
- 8 GiB system memory
- macOS 26.4
- CPU-only execution; no GPU or remote worker
- Model fitting constrained to one thread by `threadpoolctl`

Package, Python, BLAS/OpenMP, and platform details are recorded in the two
experiment manifests. Peak resident memory was not instrumented; both releases
completed on the 8 GiB host. Final paper, figure, and aggregate artifacts occupy
less than 6 MiB. Temporary worktrees, package caches, and LaTeX intermediates
require additional local storage.

## Measured Releases

| Release command | Wall time | User CPU | System CPU |
|---|---:|---:|---:|
| `scripts/release_expansion_artifacts.sh local_data/expansion` | 325.19 s | 297.16 s | 10.61 s |
| `scripts/release_paper_artifacts.sh local_data/published_rotation_prices.csv` | 145.83 s | 124.67 s | 7.70 s |
| Total | 471.02 s | 421.83 s | 18.31 s |

The expansion time includes ten seeded walk-forward GBRT runs, 120 five-thousand-draw
bootstrap cells, five PNG/PDF figure pairs, and manifest hashing. The paper
release time includes the retrospective experiment, ten-plot report gallery,
public and anonymous Tectonic builds, and checksum validation. Cold Matplotlib
font-cache creation is included. Provider retrieval, dependency installation,
the test suite, and manual visual review are not included.
