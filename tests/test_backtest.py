import numpy as np
import pandas as pd

from crypto_stat_arb.backtest import metrics, run_backtest
from crypto_stat_arb.cli import synthetic_market
from crypto_stat_arb.model import PairFit

FIT = PairFit("DEMO_Y", "DEMO_X", .7, .97, 0., 0.)


def sim(**params):
    o, c, _ = synthetic_market(n=1150)
    return run_backtest(o, c, FIT, 750, lookback=80, **params)


def test_initial_flat_final_liquidation_and_accounting():
    r = sim()
    assert r.position_at_open.iloc[0] == 0
    assert r.position_at_open.iloc[-1] == 0
    assert abs(r.held_y.iloc[-1]) == abs(r.held_x.iloc[-1]) == 0
    assert np.allclose(r.equity, (1 + r.net_return).cumprod())
    assert ((r.drawdown <= 0) & (r.drawdown >= -1)).all()
    assert np.allclose(r.net_return, r.gross_return + r.funding_pnl -
                       r.borrow_cost - r.execution_cost)


def test_higher_execution_costs_reduce_equity():
    free = sim(fee_bps=0, slippage_bps=0)
    priced = sim(fee_bps=6, slippage_bps=8)
    assert free.equity.iloc[-1] >= priced.equity.iloc[-1]
    assert priced.execution_cost.sum() > 0
    assert priced.traded_notional.sum() > 0


def test_signal_cannot_see_the_next_close():
    o, c, _ = synthetic_market(n=1150)
    original = run_backtest(o, c, FIT, 750, lookback=80)
    changed = c.copy()
    changed.loc[changed.index[-1], "DEMO_Y"] *= 9
    after = run_backtest(o, changed, FIT, 750, lookback=80)
    pd.testing.assert_series_equal(original.position_at_open, after.position_at_open)
    pd.testing.assert_series_equal(original.net_return, after.net_return)


def test_trade_occurs_open_after_signal():
    r = sim()
    prior = r.signal_at_close.shift(1).fillna(0).astype(int)
    assert (r.position_at_open.iloc[:-1] == prior.iloc[:-1]).all()


def test_funding_positive_rate_charges_long_and_credits_short():
    o, c, _ = synthetic_market(n=1150)
    baseline = run_backtest(o, c, FIT, 750, lookback=80, fee_bps=0, slippage_bps=0)
    funding = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    funding["DEMO_Y"] = .001
    funded = run_backtest(o, c, FIT, 750, lookback=80, fee_bps=0,
                          slippage_bps=0, funding=funding)
    in_long_y = funded.held_y.shift(1).fillna(0) > 0
    assert (funded.loc[in_long_y, "funding_pnl"] < 0).all()
    assert not np.allclose(baseline.net_return, funded.net_return)


def test_metrics_and_finite_results():
    r = sim()
    summary = metrics(r)
    assert summary["oos_bars"] == len(r)
    assert summary["entries"] >= 0
    assert np.isfinite(r.net_return).all()


def test_closed_trade_ledger_is_consistent():
    from crypto_stat_arb.backtest import trade_ledger
    r = sim()
    trades = trade_ledger(r)
    assert len(trades) == metrics(r)["entries"]
    assert (trades.holding_hours > 0).all()
    assert trades.entry_utc.is_monotonic_increasing
    assert trades.exit_utc.max() <= r.index[-1]
