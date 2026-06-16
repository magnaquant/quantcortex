# Optional History Rewrite Plan

No history rewrite has been executed. The current commit removes the files from
`main`, but older commits remain retrievable. Rewriting is destructive, changes
every later commit ID, can invalidate pull-request diffs, and can be undone by
an old clone pushing stale history. Proceed only after a separate owner approval
and a legal determination that purging history is necessary.

As of June 15, 2026, the public repository has no tags or GitHub releases. The
snapshot and generated charts entered history in `d62de1e`; the snapshot moved
from `data/sample/` to `quantcortex/data/sample/` in `599ce13`. Recheck branches,
tags, releases, forks, and open pull requests immediately before any rewrite.

## Proposed Scope

Remove every historical version of:

- `data/sample/rotation_prices.csv`
- `quantcortex/data/sample/rotation_prices.csv`
- `docs/img/equity_vs_benchmarks.png`
- `docs/img/drawdown.png`
- `docs/img/rolling_sharpe.png`
- `research/01_data_quality.ipynb` through `research/05_live_trading_bridge.ipynb`

Removing and then re-adding the five clean notebooks is more reliable than an
untested JSON callback and guarantees that embedded historical outputs are gone.
The tradeoff is loss of notebook source history.

## Procedure

1. Freeze pushes, close or merge open pull requests, record the remote `main`
   SHA, and make an offline mirror backup.
2. Save the current output-free notebooks outside the clone.
3. In a fresh clone with the latest `git-filter-repo`, run an
   `--invert-paths` rewrite naming every path above, including both historical
   CSV locations. Do not improvise this in the working repository.
4. Re-add the saved clean notebooks in one new commit. Run the full test, lint,
   package, notebook-structure, and forbidden-artifact checks.
5. Verify with `git log --all --name-status -- <each-removed-path>` and
   `git rev-list --objects --all`; neither the CSV nor generated images should
   remain. Confirm the notebooks first appear only in the clean re-add commit.
6. Review `git-filter-repo`'s changed-ref report. Force-push only after checking
   that the recorded remote SHA has not moved and temporarily adjusting branch
   protection if required.
7. Require collaborators to discard or carefully clean old clones. Coordinate
   with fork owners. Contact GitHub Support about cached objects or pull-request
   refs only if GitHub considers the material eligible for removal.

Follow the upstream
[git-filter-repo manual](https://github.com/newren/git-filter-repo/blob/main/Documentation/git-filter-repo.txt)
and GitHub's
[history-removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
