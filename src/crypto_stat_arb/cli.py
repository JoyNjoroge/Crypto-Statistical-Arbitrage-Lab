"""Run a reproducible experiment and write everything to an output folder."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .backtest import metrics, run_backtest, trade_ledger
from .data import load_panel, read_funding
from .model import from_row, screen_pairs
from .report import cost_stress, save_json, save_plot, write_research_report

DEFAULT_PAIRS = "ETHUSDT:ETCUSDT,BTCUSDT:ETHUSDT,SOLUSDT:AVAXUSDT"


Panel = tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]


def synthetic_market(seed: int = 2026, n: int = 3000) -> Panel:
    rng = np.random.default_rng(seed)
    log_x = np.log(1600) + np.cumsum(rng.normal(0, .008, n))
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = .92 * residual[i-1] + rng.normal(0, .008)
    log_y = .7 + .97 * log_x + residual
    log_unrelated = np.log(400) + np.cumsum(rng.normal(0, .011, n))
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    closes = pd.DataFrame({"DEMO_Y": np.exp(log_y), "DEMO_X": np.exp(log_x),
                           "DEMO_OTHER": np.exp(log_unrelated)}, index=idx)
    opens = closes.shift(1)
    opens.iloc[0] = closes.iloc[0]
    return opens, closes, [("DEMO_Y", "DEMO_X"), ("DEMO_Y", "DEMO_OTHER")]


def parse_pairs(raw: str) -> list[tuple[str, str]]:
    pairs = []
    for item in raw.split(","):
        parts = [s.strip().upper() for s in item.split(":")]
        if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
            raise ValueError("pairs must be different assets, separated as Y:X,Y:X")
        pairs.append((parts[0], parts[1]))
    if len(set(pairs)) < len(pairs):
        raise ValueError("duplicate pairs")
    return pairs


def walk_forward(opens: pd.DataFrame, closes: pd.DataFrame, pairs: list[tuple[str, str]],
                 first_split: int, fold_bars: int, threshold: float, require_i1: bool,
                 params: dict, funding: pd.DataFrame | None, out: Path) -> None:
    records = []
    returns = []
    for start in range(first_split, len(closes) - 2, fold_bars):
        end = min(len(closes), start + fold_bars)
        if end - start < 24:
            break
        screen = screen_pairs(closes, pairs, start, threshold, require_i1)
        valid = screen[screen.eligible]
        record = {"train_end": str(closes.index[start - 1]),
                  "test_start": str(closes.index[start]),
                  "test_end": str(closes.index[end - 1]),
                  "bars": end - start, "selected_pair": None}
        if valid.empty:
            record["status"] = "no eligible pair"
            records.append(record)
            continue
        fit = from_row(valid.iloc[0])
        result = run_backtest(opens, closes, fit, start, test_end=end, funding=funding, **params)
        record["selected_pair"] = f"{fit.y}/{fit.x}"
        record.update(metrics(result))
        record["status"] = "backtested"
        records.append(record)
        frame = result[["net_return"]].copy()
        frame["pair"] = record["selected_pair"]
        returns.append(frame)
    save_json(records, out / "walk_forward_folds.json")
    if returns:
        pd.concat(returns).to_csv(out / "walk_forward_returns.csv")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Crypto statistical arbitrage research, no order placement")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--demo", action="store_true", help="synthetic data, no internet")
    p.add_argument("--seed", type=int, default=2026, help="seed for --demo prices")
    p.add_argument("--market", choices=["spot", "um"], default="spot",
                   help="Binance spot or USD-M perpetual klines")
    p.add_argument("--start", default="2025-07", help="YYYY-MM")
    p.add_argument("--end", default="2026-09", help="YYYY-MM, must be a published month")
    p.add_argument("--pairs", default=DEFAULT_PAIRS, help="Y:X,Y:X, selected in training only")
    p.add_argument("--local-dir", type=Path, help="folder of SYMBOL.csv files; bypasses downloads")
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--no-verify", action="store_true",
                   help="skip SHA-256 verification of downloaded archives")
    p.add_argument("--output", type=Path, default=Path("results/run"))
    p.add_argument("--train-fraction", type=float, default=.65)
    p.add_argument("--threshold", type=float, default=.05)
    p.add_argument("--skip-i1-check", action="store_true", help="exploratory only")
    p.add_argument("--lookback", type=int, default=168)
    p.add_argument("--entry", type=float, default=2.)
    p.add_argument("--exit", type=float, default=.5)
    p.add_argument("--stop", type=float, default=4.)
    p.add_argument("--max-holding", type=int, default=168)
    p.add_argument("--cooldown", type=int, default=12)
    p.add_argument("--fee-bps", type=float, default=6.)
    p.add_argument("--slippage-bps", type=float, default=8.)
    p.add_argument("--borrow-apr", type=float, default=0.,
                   help="annualized borrow assumption; 0 means unmodeled if shorting spot")
    p.add_argument("--funding-file", type=Path, help="futures funding events CSV")
    p.add_argument("--walk-forward", action="store_true", help="expanding formation, fixed future folds")
    p.add_argument("--fold-bars", type=int, default=720)
    p.add_argument("--no-cost-stress", action="store_true")
    p.add_argument("--quiet", action="store_true", help="hide download progress")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if not .5 <= args.train_fraction <= .85:
        p.error("train-fraction must be between .5 and .85")
    if args.fold_bars < 24:
        p.error("fold-bars must be >= 24")
    if args.demo and (args.local_dir or args.funding_file):
        p.error("demo cannot be combined with local or funding files")
    if args.funding_file and args.market != "um":
        p.error("funding-file only applies to --market um")
    try:
        pairs = parse_pairs(args.pairs)
        if args.demo:
            opens, closes, pairs = synthetic_market(args.seed)
            source = f"deterministic simulated prices, seed {args.seed}"
            market = "synthetic"
        else:
            symbols = [s for pair in pairs for s in pair]
            opens, closes = load_panel(symbols, args.start, args.end, args.cache,
                                       args.market, args.local_dir, not args.no_verify)
            source = (str(args.local_dir) if args.local_dir else
                      "https://data.binance.vision (monthly 1h candles)")
            market = "Binance USD-M perpetual" if args.market == "um" else "Binance spot proxy"
        split = int(len(closes) * args.train_fraction)
        if split < 250 or len(closes) - split < 24:
            raise ValueError("need at least 250 training bars and 24 testing bars")
        funding = None
        if args.funding_file:
            funding = read_funding(args.funding_file, pd.DatetimeIndex(closes.index),
                                   list(closes.columns))
        ranking = screen_pairs(closes, pairs, split, args.threshold, not args.skip_i1_check)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out / "pair_screening_train_only.csv", index=False)
    print("\nFormation-only pair screening")
    print(ranking[["y", "x", "pvalue", "qvalue", "i1_plausible", "eligible"]].to_string(index=False))
    params = {"lookback": args.lookback, "entry": args.entry, "exit": args.exit,
              "stop": args.stop, "max_holding": args.max_holding, "cooldown": args.cooldown,
              "fee_bps": args.fee_bps, "slippage_bps": args.slippage_bps,
              "borrow_apr": args.borrow_apr}
    report = {"demo": args.demo, "source": source, "market": market,
              "market_code": args.market if not args.demo else "synthetic",
              "package_version": __version__,
              "split_at_utc": closes.index[split].isoformat(), "train_bars": split,
              "oos_bars": len(closes) - split, "candidate_pairs": [list(x) for x in pairs],
              "require_i1": not args.skip_i1_check, "threshold": args.threshold,
              "funding_file_used": bool(funding is not None), "parameters": params,
              "pair": None, "metrics": None}
    eligible = ranking[ranking.eligible]
    if not eligible.empty:
        fit = from_row(eligible.iloc[0])
        result = run_backtest(opens, closes, fit, split, funding=funding, **params)
        report["pair"] = {"y": fit.y, "x": fit.x, "alpha": fit.alpha,
                          "beta": fit.beta, "pvalue": fit.pvalue, "qvalue": fit.qvalue}
        report["metrics"] = metrics(result)
        result.to_csv(out / "out_of_sample_backtest.csv")
        trades = result.loc[result.position_at_open.ne(result.position_at_open.shift(1).fillna(0))]
        trades.to_csv(out / "position_changes.csv")
        trade_ledger(result).to_csv(out / "trade_ledger.csv", index=False)
        save_plot(result, out / "backtest.png", f"{fit.y}/{fit.x}")
        stress = None
        if not args.no_cost_stress:
            stress = cost_stress(opens, closes, fit, split, params, funding)
            stress.to_csv(out / "cost_sensitivity.csv", index=False)
        print(f"\nOOS pair: {fit.y}/{fit.x}")
        print(f"Net return: {report['metrics']['net_total_return']:.2%}")
        print(f"Max drawdown: {report['metrics']['max_drawdown']:.2%}")
        print(f"Entries: {report['metrics']['entries']}")
    else:
        stress = None
        print("\nNo pair passed the training screen. No trade simulation was run.")
    save_json(report, out / "summary.json")
    write_research_report(out, report, ranking, stress)
    if args.walk_forward:
        walk_forward(opens, closes, pairs, split, args.fold_bars, args.threshold,
                     not args.skip_i1_check, params, funding, out)
    print(f"\nResearch outputs: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
