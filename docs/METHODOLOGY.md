# Methodology

## What we're testing

Two related crypto assets may wander apart temporarily even when their log prices share a stable long-run relationship. I'm testing whether this relation can predict convergence that is large enough to trade after costs. Correlation by itself is not evidence.

## Pair choice

The candidate pairs are listed before looking at the holdout. I regress log(Y) on log(X) with an intercept on the formation sample, then use Engle-Granger to test the null that the series are not cointegrated. I use the Benjamini-Hochberg procedure for the stated screening family. I also report ADF p-values for levels and first differences. Those ADF checks are an approximate screen for I(1), not a formal guarantee.

The selected pair is the first eligible candidate by adjusted q-value. If nothing passes, I don't run a trading backtest. Reversing Y and X may change the test and is a different preregistered hypothesis. Broad searches over dozens of assets need stronger attention to multiple testing and survivorship bias.

## Signals and fills

I freeze the regression coefficients at the formation boundary. The residual is:

`s[t] = log(Y[t]) - alpha - beta * log(X[t])`

The z-score uses the **previous** 168 residuals for its mean and standard deviation, excluding the latest residual. At the close of an hour I may enter at +/-2, exit near zero (+/-0.5), stop beyond +/-4, or close after 168 bars. The cooldown is 12 bars after exiting. A signal from close `t` is tradable no earlier than open `t+1`. The first open of the test starts flat, and the last open liquidates positions.

With a positive log-price beta, I hold the long-spread portfolio as `wY = 1/(1+beta)` and `wX = -beta/(1+beta)` with gross absolute weight 1. The reverse trade negates both weights. Log-price hedge weights approximate a local return hedge. They don't perfectly dollar-neutralize every market move, and the ratio can drift.

At each open, the simulator marks previously held weights to the new prices. It then calculates the turnover needed to reach target weights and removes fees/slippage from equity. Turnover therefore includes small trades required when holdings drift, not just entries and exits. We can't assume every target fill is available at the displayed open.

## Costs and instrument assumptions

The base case models 6 bps fees and 8 bps slippage **for each traded dollar**, one-way. These are explicit assumptions, not guaranteed exchange quotes. The default borrow APR is zero and represents an omitted spot-short cost. For USD-M perpetual candles, funding can be supplied as a CSV of settlement events and rates. Rates should be *decimal fractions* (0.0001 means 1 basis point). Positive funding charges longs and credits shorts. Funding is applied at event timestamps to positions carried up to that open. Funding completeness needs independent verification. Without a comprehensive funding series, results are incomplete.

The model excludes market depth, variable spreads, actual order sizing, adverse selection, exchange downtime, financing collateral, margin utilization, liquidation, tax, and operational errors. It assumes returns and weights are based on quoted linear notional exposures, which is closer to linear perpetuals than spot borrowing.

## Holdout and walk-forward

The default experiment reserves the last 35% as a chronological holdout. Pair selection and regression happen on formation only. Fixed signal thresholds are specified before looking at holdout results. Cost sensitivity applies different hypothetical costs to *the same chosen model*, not to select a favorite OOS setting. The optional walk-forward mode repeats screening using expanding training windows and evaluates successive nonoverlapping future windows. It is an extra diagnostic, not a license to optimize the strategy repeatedly on the same data.

## Interpreting results

I care about the risk-adjusted OOS return, worst drawdown, number of independent trade entries, fraction of time invested, total traded notional, and sensitivity to cost assumptions. High annualized Sharpe from a short or low-activity sample can be unstable. To make that concrete, the summary carries a 95% interval for the annualized Sharpe from a circular block bootstrap of the hourly net returns (24-hour blocks, 2000 draws, fixed seed). Blocks keep the within-trade dependence that an i.i.d. resample would destroy. If the interval includes zero, the point estimate is not something to lean on. The interval describes sampling noise for *this* strategy on *this* sample; it does not correct for the choices made before the holdout was looked at. A pair that does not pass screening or loses money after costs is a valid result. Success is a reproducible explanation, not a forced green number.
