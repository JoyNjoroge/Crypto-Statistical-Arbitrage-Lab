import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from crypto_stat_arb.backtest import bootstrap_sharpe, metrics, run_backtest, trade_ledger
from crypto_stat_arb.cli import synthetic_market
from crypto_stat_arb.data import load_panel, read_funding, read_local_symbol
from crypto_stat_arb.model import PairFit, next_signal, screen_pairs, spread_zscore
from crypto_stat_arb.paper import main as paper_main
from crypto_stat_arb.paper import run_once
from crypto_stat_arb.report import save_json, save_plot, write_research_report

FIT = PairFit("DEMO_Y", "DEMO_X", .7, .97, 0., 0.)
IDX = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")


# ---- funding and local CSV inputs -------------------------------------------------------

def write_funding(path, rows):
    pd.DataFrame(rows, columns=["timestamp", "symbol", "funding_rate"]).to_csv(path, index=False)
    return path


def test_funding_rejects_bad_inputs(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"timestamp": [IDX[0]], "symbol": ["AAA"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="needs timestamp,symbol,funding_rate"):
        read_funding(p, IDX, ["AAA"])
    write_funding(p, [[IDX[1], "AAA", .0001], [IDX[1], "AAA", .0002]])
    with pytest.raises(ValueError, match="duplicate funding event"):
        read_funding(p, IDX, ["AAA"])
    write_funding(p, [[IDX[1], "AAA", .0001]])
    with pytest.raises(ValueError, match="missing a required symbol"):
        read_funding(p, IDX, ["AAA", "BBB"])
    write_funding(p, [[IDX[-1] + pd.Timedelta(hours=1), "AAA", .0001]])
    with pytest.raises(ValueError, match="outside candle timestamps"):
        read_funding(p, IDX, ["AAA"])
    write_funding(p, [[IDX[1], "AAA", float("inf")]])
    with pytest.raises(ValueError, match="nonfinite"):
        read_funding(p, IDX, ["AAA"])


def test_funding_accepts_epoch_millisecond_timestamps(tmp_path):
    p = write_funding(tmp_path / "f.csv", [[int(IDX[2].timestamp() * 1000), "aaa", -.0003]])
    f = read_funding(p, IDX, ["AAA"])
    assert f.loc[IDX[2], "AAA"] == pytest.approx(-.0003)
    assert f.drop(index=IDX[2]).eq(0).all().all()


def test_local_csv_validation(tmp_path):
    p = tmp_path / "A.csv"
    pd.DataFrame({"timestamp": IDX, "open": [1, 1, 1, 1, 1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="must contain"):
        read_local_symbol(p)
    pd.DataFrame({"timestamp": IDX, "open": [1, 1, 0, 1, 1], "close": [1] * 5}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="invalid prices"):
        read_local_symbol(p)
    pd.DataFrame({"timestamp": IDX, "open": [1, 1, "x", 1, 1], "close": [1] * 5}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="invalid prices"):
        read_local_symbol(p)


def test_panel_needs_enough_overlap(tmp_path):
    for s in ("AAA", "BBB"):
        pd.DataFrame({"timestamp": IDX, "open": [1.] * 5, "close": [1.] * 5}).to_csv(
            tmp_path / f"{s}.csv", index=False)
    with pytest.raises(ValueError, match="at least 300"):
        load_panel(["AAA", "BBB"], "2025-01", "2025-01", tmp_path, local_dir=tmp_path)


# ---- screening and signals --------------------------------------------------------------

def test_screen_pairs_validation():
    _, c, pairs = synthetic_market(n=600)
    with pytest.raises(ValueError, match="threshold"):
        screen_pairs(c, pairs, 400, threshold=1.5)
    with pytest.raises(ValueError, match="250 formation bars"):
        screen_pairs(c, pairs, 100)
    with pytest.raises(ValueError, match="no pairs"):
        screen_pairs(c, [], 400)
    with pytest.raises(ValueError, match="invalid pair"):
        screen_pairs(c, [("DEMO_Y", "NOPE")], 400)
    bad = c.copy()
    bad.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="missing/nonpositive"):
        screen_pairs(bad, pairs, 400)


def test_screen_can_skip_i1_requirement():
    _, c, pairs = synthetic_market(n=600)
    strict = screen_pairs(c, pairs, 400, require_i1=True)
    loose = screen_pairs(c, pairs, 400, require_i1=False)
    assert (loose.eligible >= strict.eligible).all()
    assert list(strict.columns) == list(loose.columns)


def test_zscore_lookback_floor_and_zero_variance():
    _, c, _ = synthetic_market(n=300)
    with pytest.raises(ValueError, match="lookback"):
        spread_zscore(c, FIT, 5)
    flat = pd.DataFrame({"DEMO_Y": 2., "DEMO_X": 1.}, index=c.index)
    assert spread_zscore(flat, FIT, 30).iloc[-1] != spread_zscore(flat, FIT, 30).iloc[-1]  # NaN


def test_signal_state_validation_and_nan_handling():
    from crypto_stat_arb.model import SignalState
    with pytest.raises(ValueError, match="exit < entry < stop"):
        next_signal(0., SignalState(), entry=1., exit=2., stop=4.)
    with pytest.raises(ValueError, match="holding/cooldown"):
        next_signal(0., SignalState(), max_holding=0)
    flat_after_nan = next_signal(float("nan"), SignalState(position=1, age=3), cooldown=4)
    assert (flat_after_nan.position, flat_after_nan.wait) == (0, 4)
    assert next_signal(float("nan"), SignalState()) == SignalState()
    # a beyond-stop reading while flat must not open a position
    assert next_signal(5., SignalState()).position == 0
    assert next_signal(-5., SignalState()).position == 0


# ---- backtest guards and metrics --------------------------------------------------------

def test_backtest_input_validation():
    o, c, _ = synthetic_market(n=1150)
    with pytest.raises(ValueError, match="align"):
        run_backtest(o.iloc[1:], c, FIT, 750, lookback=80)
    with pytest.raises(ValueError, match="positive hedge beta"):
        run_backtest(o, c, PairFit("DEMO_Y", "DEMO_X", 0, -1, 0, 0), 750, lookback=80)
    with pytest.raises(ValueError, match="cannot be negative"):
        run_backtest(o, c, FIT, 750, lookback=80, fee_bps=-1)
    with pytest.raises(ValueError, match="out-of-sample boundaries"):
        run_backtest(o, c, FIT, 750, lookback=800)
    with pytest.raises(ValueError, match="out-of-sample boundaries"):
        run_backtest(o, c, FIT, 750, lookback=80, test_end=751)
    bad = c.copy()
    bad.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        run_backtest(o, bad, FIT, 750, lookback=80)
    bad.iloc[3, 0] = -1
    with pytest.raises(ValueError, match="positive"):
        run_backtest(o, bad, FIT, 750, lookback=80)
    funding = pd.DataFrame(0., index=c.index[:-1], columns=c.columns)
    with pytest.raises(ValueError, match="funding and candle"):
        run_backtest(o, c, FIT, 750, lookback=80, funding=funding)
    funding = pd.DataFrame(0., index=c.index, columns=["DEMO_Y"])
    with pytest.raises(ValueError, match="missing funding leg"):
        run_backtest(o, c, FIT, 750, lookback=80, funding=funding)
    funding = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    with pytest.raises(ValueError, match="numerical"):
        run_backtest(o, c, FIT, 750, lookback=80, funding=funding)


def test_borrow_cost_charged_only_on_short_leg():
    o, c, _ = synthetic_market(n=1150)
    r = run_backtest(o, c, FIT, 750, lookback=80, fee_bps=0, slippage_bps=0, borrow_apr=.1)
    short_notional = np.minimum(0, r[["held_y", "held_x"]].shift(1).fillna(0)).abs().sum(axis=1)
    expected = short_notional * .1 / 8760
    assert np.allclose(r.borrow_cost, expected)
    assert r.borrow_cost.sum() > 0


def test_no_trades_means_flat_equity_and_undefined_ratios():
    o, c, _ = synthetic_market(n=1150)
    r = run_backtest(o, c, FIT, 750, lookback=80, entry=50., exit=.5, stop=60.)
    assert (r.position_at_open == 0).all()
    assert np.allclose(r.equity, 1.)
    m = metrics(r)
    assert m["entries"] == 0
    assert m["hourly_sharpe_zero_rf"] is None
    assert m["sharpe_block_bootstrap"] is None
    assert m["winning_trade_fraction"] is None
    assert trade_ledger(r).empty
    with pytest.raises(ValueError, match="no rows"):
        metrics(r.iloc[0:0])


def test_trade_ledger_rejects_open_position():
    o, c, _ = synthetic_market(n=1150)
    r = run_backtest(o, c, FIT, 750, lookback=80)
    r.loc[r.index[-1], "position_at_open"] = 1
    with pytest.raises(ValueError, match="liquidated"):
        trade_ledger(r)


def test_depleted_equity_stops_the_simulation():
    o, c, _ = synthetic_market(n=1150)
    funding = pd.DataFrame(0., index=c.index, columns=c.columns)
    funding["DEMO_Y"] = 5.0  # absurd, forces growth <= 0 while long Y
    with pytest.raises(ArithmeticError, match="depleted"):
        run_backtest(o, c, FIT, 750, lookback=80, funding=funding)


def test_bootstrap_sharpe_properties():
    rng = np.random.default_rng(0)
    pos = pd.Series(rng.normal(.0005, .002, 2000))
    neg = -pos
    bp, bn = bootstrap_sharpe(pos, seed=1), bootstrap_sharpe(neg, seed=1)
    assert bp["sharpe_low"] < bp["sharpe_high"]
    assert bp["sharpe_low"] == pytest.approx(-bn["sharpe_high"])
    assert bp["fraction_of_draws_with_sharpe_at_or_below_zero"] < .05
    assert bootstrap_sharpe(pos, seed=1) == bootstrap_sharpe(pos, seed=1)
    assert bootstrap_sharpe(pos, seed=1) != bootstrap_sharpe(pos, seed=2)
    assert bootstrap_sharpe(pd.Series(np.zeros(500))) is None
    assert bootstrap_sharpe(pd.Series([1., 2.])) is None
    assert bootstrap_sharpe(pd.Series([1., np.nan] * 100)) is None
    with pytest.raises(ValueError):
        bootstrap_sharpe(pos, draws=10)


# ---- report helpers ---------------------------------------------------------------------

def test_save_json_handles_numpy_and_refuses_nan(tmp_path):
    p = tmp_path / "x.json"
    save_json({"a": np.int64(3), "b": np.float64(.5), "t": pd.Timestamp("2025-01-01", tz="UTC")}, p)
    assert json.loads(p.read_text()) == {"a": 3, "b": .5, "t": "2025-01-01T00:00:00+00:00"}
    with pytest.raises(ValueError):
        save_json({"a": float("nan")}, p)
    with pytest.raises(TypeError):
        save_json({"a": object()}, p)


def test_plot_and_report_for_real_data_banner(tmp_path):
    o, c, _ = synthetic_market(n=1150)
    r = run_backtest(o, c, FIT, 750, lookback=80)
    save_plot(r, tmp_path / "p.png", "Y/X")
    assert (tmp_path / "p.png").stat().st_size > 1000
    ranking = pd.DataFrame([{"y": "A", "x": "B", "pvalue": .01, "qvalue": .02, "beta": 1.,
                             "i1_plausible": True, "eligible": True}])
    report = {"demo": False, "source": "s", "market": "m", "split_at_utc": "t",
              "train_bars": 1, "oos_bars": 2, "funding_file_used": False,
              "parameters": {"fee_bps": 1, "slippage_bps": 1, "borrow_apr": 0},
              "pair": {"y": "A", "x": "B"}, "metrics": {**metrics(r), "sharpe_block_bootstrap": None}}
    write_research_report(tmp_path, report, ranking, None)
    text = (tmp_path / "RESEARCH_REPORT.md").read_text()
    assert "HISTORICAL BACKTEST ONLY" in text
    assert "A/B" in text
    assert "block-bootstrap" not in text
    report["metrics"]["hourly_sharpe_zero_rf"] = None
    write_research_report(tmp_path, report, ranking, None)
    assert "undefined" in (tmp_path / "RESEARCH_REPORT.md").read_text()


# ---- paper monitor ----------------------------------------------------------------------

def paper_summary(tmp_path, **overrides):
    summary = {"demo": False, "market_code": "spot",
               "pair": {"y": "AAA", "x": "BBB", "alpha": .1, "beta": 1.0,
                        "pvalue": .001, "qvalue": .002},
               "parameters": {"lookback": 25, "entry": 2., "exit": .5, "stop": 4.,
                              "max_holding": 168, "cooldown": 12}}
    summary.update(overrides)
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(summary))
    return p


def test_paper_refuses_demo_or_mismatched_summaries(tmp_path):
    state = tmp_path / "state.json"
    with pytest.raises(ValueError, match="real-data summary"):
        run_once(paper_summary(tmp_path, demo=True), state, "spot")
    with pytest.raises(ValueError, match="market must match"):
        run_once(paper_summary(tmp_path), state, "um")
    big = {"lookback": 999, "entry": 2., "exit": .5, "stop": 4., "max_holding": 168, "cooldown": 12}
    with pytest.raises(ValueError, match="lookback <= 998"):
        run_once(paper_summary(tmp_path, parameters=big), state, "spot")


def test_paper_detects_feed_gaps_and_short_history(tmp_path):
    summary = paper_summary(tmp_path)
    state = tmp_path / "state.json"
    index = pd.date_range("2026-01-01", periods=70, freq="h", tz="UTC")
    x = pd.Series(100 * np.exp(np.cumsum(np.sin(np.arange(70)) * .002)), index=index)
    a = pd.DataFrame({"AAA": np.exp(.1) * x.values}, index=index)
    b = pd.DataFrame({"BBB": x.values}, index=index)
    with patch("crypto_stat_arb.paper.latest_closed", side_effect=[a.iloc[:10], b.iloc[:10]]), \
            pytest.raises(ValueError, match="not enough completed candles"):
        run_once(summary, state, "spot")
    with patch("crypto_stat_arb.paper.latest_closed", side_effect=[a.drop(index=index[30]), b]), \
            pytest.raises(ValueError, match="missing candle"):
        run_once(summary, state, "spot")


def test_paper_state_advances_one_bar_at_a_time(tmp_path):
    summary = paper_summary(tmp_path)
    state = tmp_path / "state.json"
    index = pd.date_range("2026-01-01", periods=71, freq="h", tz="UTC")
    x = pd.Series(100 * np.exp(np.cumsum(np.sin(np.arange(71)) * .002)), index=index)
    a = pd.DataFrame({"AAA": np.exp(.1) * x.values}, index=index)
    b = pd.DataFrame({"BBB": x.values}, index=index)
    with patch("crypto_stat_arb.paper.latest_closed", side_effect=[a.iloc[:70], b.iloc[:70], a, b]):
        first = run_once(summary, state, "spot")
        second = run_once(summary, state, "spot")
    assert "initialized" in first["status"]
    assert second["status"] in ("new signal", "no change")
    assert second["last_bar_utc"] == index[-1].isoformat()
    assert second["position"] in (-1, 0, 1)


def test_paper_cli_argument_errors(tmp_path):
    with pytest.raises(SystemExit) as exc:
        paper_main(["--summary", str(tmp_path / "missing.json"), "--poll-seconds", "1"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        paper_main(["--summary", str(tmp_path / "missing.json"), "--state", str(tmp_path / "s.json")])
    assert exc.value.code == 2


def test_paper_cli_runs_once_and_prints_json(tmp_path, capsys):
    summary = paper_summary(tmp_path)
    index = pd.date_range("2026-01-01", periods=70, freq="h", tz="UTC")
    x = pd.Series(100 * np.exp(np.cumsum(np.sin(np.arange(70)) * .002)), index=index)
    a = pd.DataFrame({"AAA": np.exp(.1) * x.values}, index=index)
    b = pd.DataFrame({"BBB": x.values}, index=index)
    with patch("crypto_stat_arb.paper.latest_closed", side_effect=[a, b]):
        assert paper_main(["--summary", str(summary), "--state", str(tmp_path / "s.json")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["position"] == 0
