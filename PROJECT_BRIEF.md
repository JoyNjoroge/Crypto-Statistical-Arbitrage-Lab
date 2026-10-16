# Project 01 | Crypto Statistical Arbitrage

**Intended portfolio position:** Crypto quantitative research, market-neutral trading research, financial econometrics, Python, risk.

**Question:** Across a prespecified candidate universe, are there crypto-asset price relationships that are cointegrated on an initial formation window and economically exploitable on a held-out future window after trading costs?

**Initial market:** Binance spot historical hourly OHLC (reference prices only). **Trading venue extension:** USDT perpetual swaps with funding and executable fills.

**Null/alternatives:** Non-cointegration is the default null for Engle-Granger screening. An apparently stationary residual does not imply net-tradable profits. Empirically measure both statistical evidence and economic significance.

## Progress and next research steps

- **Implemented:** archive downloader with checksum verification, local CSV reader, formation-only screening, z-score signals, stop/hold/cooldown rules, next-open backtester with drift-aware rebalancing and modeled costs, funding-input support, trade ledger, cost sensitivity, block-bootstrap Sharpe interval, walk-forward folds, generated reports, read-only paper signal monitor, synthetic example, tests and research notebook.
- **Session B (next):** fetch real data, run prespecified candidates, inspect pair-screen output, confirm gaps/spikes/exchange changes and trade counts. Do not adjust parameters to maximize holdout performance.
- **Session C:** interpret walk-forward results, check regimes, compare a rolling hedge-ratio experiment to the frozen baseline without tuning on the holdout, and establish uncertainty ranges.
- **Session D:** independently verify futures funding series and instrument-specific costs, practical capital/margin constraints, market-microstructure and liquidity checks, then document conclusions for GitHub.

## Portfolio acceptance tests

- No future data in formation selection or z-score reference statistics.
- No fabricated prices for missing timestamps, no forward filling into an exchange outage.
- Every return is defined under explicit trade timing and exposure conventions.
- All fee, slippage, financing and execution limitations are disclosed.
- A negative finding is acceptable and should be documented.
- Results have an unambiguous **synthetic vs historical** label.
- Figures, code, parameters, and raw data provenance can be reproduced.

**Not yet verified with real market data:** Binance archive run in this environment, venue-specific fee/borrow rates and complete funding series, actual executable fills, margin/liquidation and live forward performance. Walk-forward code and funding inputs exist, but must be evaluated on authentic data. No real-market alpha is claimed.
