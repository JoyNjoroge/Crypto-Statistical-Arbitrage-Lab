"""Formation-period pair screening and close-of-bar spread signals."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class PairFit:
    y: str
    x: str
    alpha: float
    beta: float
    pvalue: float
    qvalue: float


def adf_pvalue(series: pd.Series) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(adfuller(series, autolag="AIC")[1])


def screen_pairs(closes: pd.DataFrame, pairs: list[tuple[str, str]], train_stop: int,
                 threshold: float = .05, require_i1: bool = True) -> pd.DataFrame:
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie between 0 and 1")
    if train_stop < 250 or train_stop > len(closes):
        raise ValueError("need at least 250 formation bars")
    if not pairs:
        raise ValueError("no pairs supplied")
    rows = []
    for y, x in pairs:
        if y == x or y not in closes or x not in closes:
            raise ValueError(f"invalid pair {y}/{x}")
        frame = closes.loc[:, [y, x]].iloc[:train_stop]
        if frame.isna().any().any() or (frame <= 0).any().any():
            raise ValueError(f"missing/nonpositive training values in {y}/{x}")
        logs = np.log(frame)
        yy, xx = logs[y], logs[x]
        params = sm.OLS(yy, sm.add_constant(xx)).fit().params
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                _, p, _ = coint(yy, xx, trend="c", autolag="aic")
        except (ValueError, np.linalg.LinAlgError):
            p = 1.0
        p = float(p) if np.isfinite(p) else 1.0
        level_ps, diff_ps = [], []
        for col in (y, x):
            try:
                level_ps.append(adf_pvalue(logs[col]))
                diff_ps.append(adf_pvalue(logs[col].diff().dropna()))
            except (ValueError, np.linalg.LinAlgError):
                level_ps.append(0.0)
                diff_ps.append(1.0)
        i1 = all(v > threshold for v in level_ps) and all(v < threshold for v in diff_ps)
        rows.append({"y": y, "x": x, "alpha": float(params.iloc[0]),
                     "beta": float(params.iloc[1]), "pvalue": p,
                     "adf_level_y": level_ps[0], "adf_level_x": level_ps[1],
                     "adf_diff_y": diff_ps[0], "adf_diff_x": diff_ps[1],
                     "i1_plausible": bool(i1)})
    table = pd.DataFrame(rows)
    table["qvalue"] = multipletests(table.pvalue.to_numpy(), method="fdr_bh")[1]
    table["eligible"] = ((table.qvalue <= threshold) & (table.beta > 0) &
                         (table.i1_plausible | (not require_i1)))
    return table.sort_values(["qvalue", "y", "x"]).reset_index(drop=True)


def from_row(row: pd.Series) -> PairFit:
    return PairFit(str(row.y), str(row.x), float(row.alpha), float(row.beta),
                   float(row.pvalue), float(row.qvalue))


def spread_zscore(closes: pd.DataFrame, fit: PairFit, lookback: int) -> pd.Series:
    if lookback < 20:
        raise ValueError("lookback must be at least 20")
    spread = np.log(closes[fit.y]) - fit.alpha - fit.beta * np.log(closes[fit.x])
    old_mean = spread.rolling(lookback, min_periods=lookback).mean().shift(1)
    old_sd = spread.rolling(lookback, min_periods=lookback).std(ddof=1).shift(1)
    return ((spread - old_mean) / old_sd.replace(0, np.nan)).rename("zscore")


@dataclass
class SignalState:
    position: int = 0
    age: int = 0
    wait: int = 0


def next_signal(value: float, state: SignalState, entry: float = 2., exit: float = .5,
                stop: float = 4., max_holding: int = 168, cooldown: int = 12) -> SignalState:
    if not (0 <= exit < entry < stop):
        raise ValueError("need 0 <= exit < entry < stop")
    if max_holding < 1 or cooldown < 0:
        raise ValueError("invalid holding/cooldown")
    direction, age, wait = state.position, state.age, state.wait
    if not np.isfinite(value):
        if direction:
            return SignalState(0, 0, cooldown)
        return state
    if direction:
        age += 1
        stopped = (direction == 1 and value <= -stop) or (direction == -1 and value >= stop)
        reverted = (direction == 1 and value >= -exit) or (direction == -1 and value <= exit)
        if stopped or reverted or age >= max_holding:
            direction, age, wait = 0, 0, cooldown
    elif wait:
        wait -= 1
    elif -stop < value <= -entry:
        direction, age = 1, 0
    elif entry <= value < stop:
        direction, age = -1, 0
    return SignalState(direction, age, wait)


def positions_from_z(z: pd.Series, entry: float = 2., exit: float = .5,
                     stop: float = 4.0, max_holding: int = 168, cooldown: int = 12) -> pd.Series:
    state = SignalState()
    positions = []
    for value in z.to_numpy():
        state = next_signal(float(value), state, entry, exit, stop, max_holding, cooldown)
        positions.append(state.position)
    return pd.Series(positions, index=z.index, dtype="int8", name="signal_at_close")
