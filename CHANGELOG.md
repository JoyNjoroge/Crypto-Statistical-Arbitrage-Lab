# Changelog

## 1.1.0

- Downloaded Binance archives are verified against the published SHA-256 `.CHECKSUM` files. A mismatch raises and the file is discarded. `--no-verify` opts out.
- `summary.json` and the research report carry a 95% circular block-bootstrap interval for the annualized out-of-sample Sharpe.
- Download progress is logged (`--quiet` silences it). `--seed` controls the synthetic demo. `--version` on both commands.
- `summary.json` records the package version that produced it.
- Test suite grew from 21 to 71 tests: CLI end to end, no-eligible-pair path, walk-forward, downloader against a fake HTTP session, funding and CSV error paths, paper monitor CLI. Coverage is enforced at 90%.
- `ruff` (with import sorting and bugbear rules) and `mypy` run in CI; the package ships `py.typed`.
- CI covers Python 3.10–3.13 on Linux plus Windows and macOS, then builds the wheel and installs it in a clean environment.
- Added `Makefile`, `Dockerfile`, `.pre-commit-config.yaml`, `CONTRIBUTING.md`.
- pandas minimum raised to 2.2 (the lowercase `"h"` frequency alias).
- Silenced the `statsmodels` 0.15 `adfuller` FutureWarning without depending on the new keyword, so 0.14 still works.

## 1.0.0

Initial release: archive downloader, local CSV reader, formation-only screening, z-score signals, next-open backtester with drift-aware rebalancing and modeled costs, funding support, trade ledger, cost sensitivity, walk-forward folds, generated reports, read-only paper monitor, synthetic demo, tests and notebook.
