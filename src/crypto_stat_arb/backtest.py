"""At each hourly open, mark existing positions and execute last close's decision."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import PairFit, positions_from_z, spread_zscore


def run_backtest(opens: pd.DataFrame, closes: pd.DataFrame, fit: PairFit,
                 test_start: int, lookback: int = 168, entry: float = 2.,
                 exit: float = .5, stop: float = 4., max_holding: int = 168,
                 cooldown: int = 12, fee_bps: float = 6., slippage_bps: float = 8.,
                 borrow_apr: float = 0., funding: pd.DataFrame | None = None,
                 test_end: int | None = None) -> pd.DataFrame:
    if not opens.index.equals(closes.index) or not opens.columns.equals(closes.columns):
        raise ValueError("open and close panels must align")
    if opens.isna().any().any() or closes.isna().any().any():
        raise ValueError("prices must not contain missing values")
    if (opens <= 0).any().any() or (closes <= 0).any().any():
        raise ValueError("prices must be positive")
    if fit.beta <= 0:
        raise ValueError("positive hedge beta required")
    if min(fee_bps, slippage_bps, borrow_apr) < 0:
        raise ValueError("costs/borrow cannot be negative")
    end = len(opens) if test_end is None else test_end
    if test_start <= lookback or test_start < 250 or end > len(opens) or end <= test_start + 2:
        raise ValueError("invalid out-of-sample boundaries")
    if funding is not None:
        if not opens.index.equals(funding.index):
            raise ValueError("funding and candle timestamps must align")
        if not {fit.y, fit.x}.issubset(funding.columns):
            raise ValueError("missing funding leg")
        if funding.isna().any().any():
            raise ValueError("funding must be numerical")

    z = spread_zscore(closes.iloc[:end], fit, lookback)
    signal = positions_from_z(z.iloc[test_start:end], entry, exit, stop, max_holding, cooldown)
    leg_symbols = [fit.y, fit.x]
    prices = opens[leg_symbols].to_numpy(dtype=float)
    index = opens.index
    base = np.array([1.0, -fit.beta]) / (1 + fit.beta)
    rate = (fee_bps + slippage_bps) / 10000
    prev_weights = np.zeros(2)
    equity = 1.0
    peak = 1.0
    rows = []

    for i in range(test_start, end):
        returns = prices[i] / prices[i - 1] - 1 if i > test_start else np.zeros(2)
        gross = float(prev_weights @ returns)
        funding_pnl = 0.0
        if funding is not None and i > test_start:
            funding_pnl = -float(prev_weights @ funding.loc[index[i], leg_symbols].to_numpy(dtype=float))
        borrow = 0.0
        if i > test_start:
            borrow = float(np.maximum(-prev_weights, 0).sum() * borrow_apr / (24 * 365))
        growth = 1 + gross + funding_pnl - borrow
        if growth <= 0:
            raise ArithmeticError("strategy equity depleted; backtest cannot continue")
        pretrade = prev_weights * (1 + returns) / growth
        target_state = int(signal.iloc[i - test_start - 1]) if i > test_start else 0
        if i == end - 1:
            target_state = 0
        target_weights = base * target_state

        # solve for costs against post-cost equity, not against imaginary free trades
        cost_pretrade = 0.0
        for _ in range(12):
            turnover = float(np.abs(target_weights * (1 - cost_pretrade) - pretrade).sum())
            updated = rate * turnover
            if abs(updated - cost_pretrade) < 1e-13:
                cost_pretrade = updated
                break
            cost_pretrade = updated
        turnover = float(np.abs(target_weights * (1 - cost_pretrade) - pretrade).sum())
        cost_pretrade = rate * turnover
        previous_equity = equity
        equity *= growth * (1 - cost_pretrade)
        net_return = equity / previous_equity - 1
        peak = max(peak, equity)
        traded_dollars_per_pretrade_equity = turnover
        rows.append({
            "timestamp": index[i], "zscore": float(z.iloc[i]),
            "signal_at_close": int(signal.iloc[i - test_start]),
            "position_at_open": target_state,
            "held_y": float(target_weights[0]), "held_x": float(target_weights[1]),
            "gross_return": gross, "funding_pnl": funding_pnl,
            "borrow_cost": borrow,
            "execution_cost": growth * cost_pretrade,
            "traded_notional": traded_dollars_per_pretrade_equity,
            "net_return": net_return, "equity": equity,
            "drawdown": equity / peak - 1,
        })
        prev_weights = target_weights

    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, name="timestamp_utc")
    return df



def trade_ledger(results: pd.DataFrame) -> pd.DataFrame:
    positions = results.position_at_open.to_numpy(dtype=int)
    equity = results.equity.to_numpy(dtype=float)
    times = results.index
    trades = []
    entry = None
    for i, direction in enumerate(positions):
        previous = positions[i - 1] if i else 0
        if entry is not None and (direction == 0 or direction != previous):
            start, side, beginning_equity = entry
            trades.append({"entry_utc": times[start], "exit_utc": times[i],
                           "direction": "long Y / short X" if side == 1 else "short Y / long X",
                           "holding_hours": i - start,
                           "net_return_on_trade_equity": float(equity[i] / beginning_equity - 1)})
            entry = None
        if direction and direction != previous:
            entry = (i, direction, equity[i - 1] if i else 1.0)
    if entry is not None:
        raise ValueError("last position wasn't liquidated")
    return pd.DataFrame(trades, columns=["entry_utc", "exit_utc", "direction",
                                          "holding_hours", "net_return_on_trade_equity"])

def bootstrap_sharpe(returns: pd.Series, block: int = 24, draws: int = 2000,
                     seed: int = 0, level: float = .95) -> dict | None:
    """Circular block bootstrap of the hourly net returns. Blocks keep short-range dependence
    (positions last many hours) that an i.i.d. resample would throw away."""
    if block < 1 or draws < 100 or not 0 < level < 1:
        raise ValueError("invalid bootstrap settings")
    r = returns.to_numpy(dtype=float)
    n = len(r)
    if n < 2 * block or not np.isfinite(r).all():
        return None
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(draws, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n
    samples = r[idx.reshape(draws, -1)[:, :n]]
    sd = samples.std(axis=1, ddof=1)
    ok = sd > 0
    if ok.sum() < draws // 2:
        return None
    sharpe = np.sqrt(8760) * samples[ok].mean(axis=1) / sd[ok]
    tail = (1 - level) / 2
    lo, hi = np.quantile(sharpe, [tail, 1 - tail])
    return {"level": level, "block_hours": block, "draws": int(ok.sum()),
            "sharpe_low": float(lo), "sharpe_high": float(hi),
            "fraction_of_draws_with_sharpe_at_or_below_zero": float((sharpe <= 0).mean())}


def metrics(results: pd.DataFrame) -> dict:
    if results.empty:
        raise ValueError("no rows to evaluate")
    ret = results.net_return
    std = float(ret.std(ddof=1))
    total_hours = len(ret) - 1
    final = float(results.equity.iloc[-1])
    annualized = final ** (8760 / total_hours) - 1 if total_hours > 0 and final > 0 else None
    sharpe = float(np.sqrt(8760) * ret.mean() / std) if std > 0 else None
    downside = float(np.sqrt(np.mean(np.minimum(0, ret.to_numpy()) ** 2)))
    sortino = float(np.sqrt(8760) * ret.mean() / downside) if downside > 0 else None
    trades = trade_ledger(results)
    trade_returns = trades.net_return_on_trade_equity if not trades.empty else pd.Series(dtype=float)
    trading = results.loc[results.position_at_open.ne(0)]
    return {
        "net_total_return": final - 1,
        "annualized_return": annualized,
        "hourly_sharpe_zero_rf": sharpe,
        "sharpe_block_bootstrap": bootstrap_sharpe(ret),
        "sortino_zero_target": sortino,
        "max_drawdown": float(results.drawdown.min()),
        "entries": len(trades),
        "winning_trade_fraction": float((trade_returns > 0).mean()) if len(trades) else None,
        "median_trade_return": float(trade_returns.median()) if len(trades) else None,
        "hours_with_position": len(trading),
        "exposure_fraction": float(len(trading) / len(results)),
        "total_execution_cost_as_fraction_of_initial_equity": float(
            (results.execution_cost * results.equity.shift(1).fillna(1)).sum()),
        "total_borrow_cost_in_return_units": float(results.borrow_cost.sum()),
        "total_funding_pnl_in_return_units": float(results.funding_pnl.sum()),
        "sum_trade_notional_over_pretrade_equity": float(results.traded_notional.sum()),
        "positive_return_hours_fraction": float((ret > 0).mean()),
        "oos_bars": len(results),
    }
