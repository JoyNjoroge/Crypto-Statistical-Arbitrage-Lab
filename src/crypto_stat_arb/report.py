"""Saved outputs are explicitly separated from any interpretation of profitability."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import metrics, run_backtest
from .model import PairFit


def save_plot(results: pd.DataFrame, path: Path, label: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(results.index, results.zscore, linewidth=.7)
    axes[0].axhline(2, linestyle="--", color="gray", linewidth=.7)
    axes[0].axhline(-2, linestyle="--", color="gray", linewidth=.7)
    axes[0].set_ylabel("spread z-score")
    axes[0].set_title(f"{label} | out-of-sample")
    axes[1].plot(results.index, results.equity, label="net equity", linewidth=1)
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=.7)
    axes[1].set_ylabel("equity (starts at 1)")
    axes[1].legend()
    axes[2].fill_between(results.index, results.drawdown, 0, alpha=.4)
    axes[2].set_ylabel("drawdown")
    axes[2].set_xlabel("UTC")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_json(data: dict | list, path: Path) -> None:
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        raise TypeError(str(type(obj)))
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=convert) + "\n")


def cost_stress(opens: pd.DataFrame, closes: pd.DataFrame, fit: PairFit,
                train_stop: int, kwargs: dict, funding: pd.DataFrame | None = None) -> pd.DataFrame:
    extra = [0, 5, 10, 20, 40]
    standard_cost = kwargs["fee_bps"] + kwargs["slippage_bps"]
    rates = sorted({*extra, standard_cost})
    rows = []
    for total_cost in rates:
        result = run_backtest(opens, closes, fit, train_stop,
                              **{**kwargs, "fee_bps": 0., "slippage_bps": total_cost}, funding=funding)
        m = metrics(result)
        rows.append({"one_way_total_bps": total_cost, "net_total_return": m["net_total_return"],
                     "sharpe": m["hourly_sharpe_zero_rf"], "max_drawdown": m["max_drawdown"],
                     "entries": m["entries"]})
    return pd.DataFrame(rows)


def write_research_report(out: Path, report: dict, ranking: pd.DataFrame,
                          stress: pd.DataFrame | None = None) -> None:
    m = report.get("metrics")
    banner = "SYNTHETIC DATA ONLY. NOT MARKET EVIDENCE." if report["demo"] else (
        "HISTORICAL BACKTEST ONLY. NOT A LIVE TRADING RESULT.")
    lines = ["# Crypto statistical arbitrage: research report", "", f"**{banner}**", "",
             "## Research question", "",
             "Can a formation-period cointegration relationship support a pairs strategy out of sample,",
             "after modeled transaction costs?", "",
             "## Experiment", "",
             f"- Data: {report['source']}",
             f"- Market: {report['market']}",
             f"- Formation ends before: {report['split_at_utc']}",
             f"- Formation bars: {report['train_bars']}",
             f"- Out-of-sample bars: {report['oos_bars']}",
             f"- Fee: {report['parameters']['fee_bps']} bps per traded dollar",
             f"- Slippage: {report['parameters']['slippage_bps']} bps per traded dollar",
             f"- Borrow APR assumption: {report['parameters']['borrow_apr']}",
             f"- Funding inputs included: {report['funding_file_used']}", "",
             "## Formation sample screening", "",
             "Engle-Granger tests the null of no cointegration. BH-adjusted q-values account for",
             "screening the predeclared candidate list. I(1) diagnostics are ADF heuristics,",
             "not proof.", ""]
    for _, row in ranking.iterrows():
        lines.append(f"- {row.y}/{row.x}: p={row.pvalue:.4g}, q={row.qvalue:.4g}, "
                     f"beta={row.beta:.3f}, I(1) plausible={bool(row.i1_plausible)}, "
                     f"selected={bool(row.eligible)}")
    if m is None:
        lines.extend(["", "## Result", "",
                      "No pair passed the preregistered screen. No OOS trades were simulated."])
    else:
        sharpe = m["hourly_sharpe_zero_rf"]
        boot = m.get("sharpe_block_bootstrap")
        sharpe_line = (f"- Annualized Sharpe (zero risk-free proxy): "
                       f"{sharpe:.2f}" if sharpe is not None else
                       "- Annualized Sharpe (zero risk-free proxy): undefined")
        if boot is not None:
            sharpe_line += (f", {boot['level']:.0%} block-bootstrap interval "
                            f"[{boot['sharpe_low']:.2f}, {boot['sharpe_high']:.2f}] "
                            f"({boot['block_hours']}h blocks, {boot['draws']} draws)")
        lines.extend(["", "## Out-of-sample result", "",
                      f"- Selected pair: {report['pair']['y']}/{report['pair']['x']}",
                      f"- Net total return: {m['net_total_return']:.2%}",
                      sharpe_line,
                      f"- Maximum drawdown: {m['max_drawdown']:.2%}",
                      f"- Entries: {m['entries']}",
                      f"- Win fraction across completed positions: {m['winning_trade_fraction']}",
                      f"- Time in positions: {m['exposure_fraction']:.1%}", ""])
        if stress is not None:
            lines.extend(["## Execution cost sensitivity", "",
                          "The strategy and fit are held fixed; these are diagnostics, not OOS tuning.",
                          "",
                          "| Total one-way cost (bps) | Net total return |", "|---:|---:|"])
            for _, row in stress.iterrows():
                lines.append(f"| {row.one_way_total_bps:.1f} | {row.net_total_return:.2%} |")
    lines.extend(["", "## Interpretation / limitations", "",
                  "This backtest can reject the idea that a strategy survives the modeled costs,",
                  "but it cannot establish a deployable edge. Price bars are not order-book quotes.",
                  "Signals use the completed close and execute at the next open, assuming fills",
                  "without market impact beyond the stated cost assumptions.",
                  "Spot shorts require a borrow mechanism and borrow rates; perpetual futures require",
                  "funding, margin, mark-price and liquidation assumptions. Missing costs remain",
                  "a risk.",
                  "A single holdout is not an independently replicated result, and trying many ideas",
                  "on it will overfit the research process.", "",
                  "## Next questions", "",
                  "1. Did the relationship persist in each volatility regime?",
                  "2. Would the signal remain after actual venue-specific fees and spread estimates?",
                  "3. Can we reproduce it on a genuinely untouched later period?",
                  "4. For futures, do funding and liquidation risks change the conclusion?", ""])
    (out / "RESEARCH_REPORT.md").write_text("\n".join(lines))
