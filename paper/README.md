# Research Paper

`main.tex` is a NeurIPS 2026-format preprint describing the audit protocol and
fixed negative-result case study. It uses the official `neurips_2026.sty` with
the `preprint` option. The work is not represented as accepted by or submitted
to NeurIPS 2026; the full-paper deadline was May 6, 2026.

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
required for the build; Poppler's `pdftoppm` is used for visual QA. The build
also updates `quantcortex_audit_neurips2026.sha256`.

Reproduce the fixed experiment with an authorized local adjusted-close matrix:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_paper_experiments.py \
  --prices-csv local_data/published_rotation_prices.csv \
  --output-dir paper --bootstrap-replications 5000
```

The input must contain `QQQ`, `VGT`, `GLD`, `TLT`, `SPY`, `VIG`, and `SHV` and
must match the SHA-256 digest in `results/manifest.json` for exact
reproduction. Raw provider data is not committed. Aggregate tables, figures,
the fixed experiment design, source metadata, generator revision, and artifact
hashes are committed under `paper/results/`, `paper/figures/`, and the manifest.

Do not edit generated tables or figures by hand. Change the experiment driver,
run its focused tests, regenerate all outputs, inspect the diffs, then rebuild
and visually review every PDF page.
