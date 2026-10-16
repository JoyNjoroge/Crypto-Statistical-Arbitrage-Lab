# Crypto statistical arbitrage: research report

**SYNTHETIC DATA ONLY. NOT MARKET EVIDENCE.**

## Research question

Can a formation-period cointegration relationship support a pairs strategy out of sample,
after modeled transaction costs?

## Experiment

- Data: deterministic simulated prices, seed 2026
- Market: synthetic
- Formation ends before: 2025-03-23T06:00:00+00:00
- Formation bars: 1950
- Out-of-sample bars: 1050
- Fee: 6.0 bps per traded dollar
- Slippage: 8.0 bps per traded dollar
- Borrow APR assumption: 0.0
- Funding inputs included: False

## Formation sample screening

Engle-Granger tests the null of no cointegration. BH-adjusted q-values account for
screening the predeclared candidate list. I(1) diagnostics are ADF heuristics,
not proof.

- DEMO_Y/DEMO_X: p=3.122e-15, q=6.244e-15, beta=0.974, I(1) plausible=True, selected=True
- DEMO_Y/DEMO_OTHER: p=0.9672, q=0.9672, beta=0.417, I(1) plausible=True, selected=False

## Out-of-sample result

- Selected pair: DEMO_Y/DEMO_X
- Net total return: 29.94%
- Annualized Sharpe (zero risk-free proxy): 10.83, 95% block-bootstrap interval [7.35, 14.47] (24h blocks, 2000 draws)
- Maximum drawdown: -2.29%
- Entries: 20
- Win fraction across completed positions: 1.0
- Time in positions: 25.3%

## Execution cost sensitivity

The strategy and fit are held fixed; these are diagnostics, not OOS tuning.

| Total one-way cost (bps) | Net total return |
|---:|---:|
| 0.0 | 37.77% |
| 5.0 | 34.92% |
| 10.0 | 32.13% |
| 14.0 | 29.94% |
| 20.0 | 26.72% |
| 40.0 | 16.55% |

## Interpretation / limitations

This backtest can reject the idea that a strategy survives the modeled costs,
but it cannot establish a deployable edge. Price bars are not order-book quotes.
Signals use the completed close and execute at the next open, assuming fills
without market impact beyond the stated cost assumptions.
Spot shorts require a borrow mechanism and borrow rates; perpetual futures require
funding, margin, mark-price and liquidation assumptions. Missing costs remain
a risk.
A single holdout is not an independently replicated result, and trying many ideas
on it will overfit the research process.

## Next questions

1. Did the relationship persist in each volatility regime?
2. Would the signal remain after actual venue-specific fees and spread estimates?
3. Can we reproduce it on a genuinely untouched later period?
4. For futures, do funding and liquidation risks change the conclusion?
