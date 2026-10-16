import json

import numpy as np
import pandas as pd
import pytest

from crypto_stat_arb import __version__
from crypto_stat_arb.cli import main, parse_pairs, synthetic_market, walk_forward

EXPECTED_FILES = [
    "pair_screening_train_only.csv", "summary.json", "out_of_sample_backtest.csv",
    "position_changes.csv", "trade_ledger.csv", "cost_sensitivity.csv", "backtest.png",
    "RESEARCH_REPORT.md",
]


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    assert main(["--demo", "--output", str(out), "--quiet"]) == 0
    return out


def test_demo_writes_every_documented_artifact(demo_run):
    for name in EXPECTED_FILES:
        assert (demo_run / name).exists(), name
    assert not (demo_run / "walk_forward_folds.json").exists()


def test_summary_schema_and_labels(demo_run):
    summary = json.loads((demo_run / "summary.json").read_text())
    assert summary["demo"] is True
    assert summary["market_code"] == "synthetic"
    assert summary["package_version"] == __version__
    assert summary["pair"]["y"] == "DEMO_Y"
    assert summary["pair"]["x"] == "DEMO_X"
    assert summary["train_bars"] + summary["oos_bars"] == 3000
    m = summary["metrics"]
    for key in ("net_total_return", "hourly_sharpe_zero_rf", "max_drawdown", "entries",
                "exposure_fraction", "sharpe_block_bootstrap"):
        assert key in m
    boot = m["sharpe_block_bootstrap"]
    assert boot["sharpe_low"] <= m["hourly_sharpe_zero_rf"] <= boot["sharpe_high"]
    assert summary["parameters"]["lookback"] == 168
    report = (demo_run / "RESEARCH_REPORT.md").read_text()
    assert "SYNTHETIC DATA ONLY" in report
    assert "block-bootstrap interval" in report


def test_saved_series_are_internally_consistent(demo_run):
    result = pd.read_csv(demo_run / "out_of_sample_backtest.csv", index_col=0, parse_dates=True)
    summary = json.loads((demo_run / "summary.json").read_text())
    assert len(result) == summary["oos_bars"]
    assert result.equity.iloc[-1] == pytest.approx(1 + summary["metrics"]["net_total_return"])
    ledger = pd.read_csv(demo_run / "trade_ledger.csv")
    assert len(ledger) == summary["metrics"]["entries"]
    stress = pd.read_csv(demo_run / "cost_sensitivity.csv")
    assert stress.one_way_total_bps.is_monotonic_increasing
    assert stress.net_total_return.is_monotonic_decreasing
    base = stress.loc[stress.one_way_total_bps == 14.0, "net_total_return"].iloc[0]
    assert base == pytest.approx(summary["metrics"]["net_total_return"])


def test_demo_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    main(["--demo", "--output", str(a), "--quiet", "--no-cost-stress"])
    main(["--demo", "--output", str(b), "--quiet", "--no-cost-stress"])
    assert json.loads((a / "summary.json").read_text()) == json.loads((b / "summary.json").read_text())


def test_demo_seed_changes_prices(tmp_path):
    main(["--demo", "--seed", "7", "--output", str(tmp_path), "--quiet", "--no-cost-stress"])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "seed 7" in summary["source"]


def test_walk_forward_writes_folds(tmp_path):
    main(["--demo", "--output", str(tmp_path), "--quiet", "--no-cost-stress",
          "--walk-forward", "--fold-bars", "480"])
    folds = json.loads((tmp_path / "walk_forward_folds.json").read_text())
    assert len(folds) >= 2
    assert all(f["status"] in ("backtested", "no eligible pair") for f in folds)
    assert all(f["bars"] <= 480 for f in folds)
    starts = [pd.Timestamp(f["test_start"]) for f in folds]
    assert starts == sorted(starts)
    if any(f["status"] == "backtested" for f in folds):
        returns = pd.read_csv(tmp_path / "walk_forward_returns.csv", index_col=0, parse_dates=True)
        assert returns.index.is_monotonic_increasing
        assert not returns.index.has_duplicates


def test_walk_forward_with_no_eligible_pair_records_fold(tmp_path):
    opens, closes, _ = synthetic_market(n=1400)
    params = {"lookback": 80, "entry": 2., "exit": .5, "stop": 4., "max_holding": 168,
              "cooldown": 12, "fee_bps": 6., "slippage_bps": 8., "borrow_apr": 0.}
    walk_forward(opens, closes, [("DEMO_Y", "DEMO_OTHER")], 900, 250, .05, True, params, None, tmp_path)
    folds = json.loads((tmp_path / "walk_forward_folds.json").read_text())
    assert folds
    assert all(f["status"] == "no eligible pair" for f in folds)
    assert not (tmp_path / "walk_forward_returns.csv").exists()


def test_no_eligible_pair_means_no_trades(tmp_path, capsys):
    index = pd.date_range("2025-01-01", periods=1200, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"AAAUSDT": np.exp(np.cumsum(rng.normal(0, .01, 1200))) * 100,
                          "BBBUSDT": np.exp(np.cumsum(rng.normal(0, .01, 1200))) * 50}, index=index)
    local = tmp_path / "local"
    local.mkdir()
    for symbol in frame:
        pd.DataFrame({"timestamp": index, "open": frame[symbol].shift(1).bfill(),
                      "close": frame[symbol]}).to_csv(local / f"{symbol}.csv", index=False)
    out = tmp_path / "out"
    assert main(["--local-dir", str(local), "--pairs", "AAAUSDT:BBBUSDT",
                 "--output", str(out), "--quiet"]) == 0
    assert "No pair passed" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text())
    assert summary["pair"] is None
    assert summary["metrics"] is None
    assert summary["demo"] is False
    assert (out / "pair_screening_train_only.csv").exists()
    assert not (out / "out_of_sample_backtest.csv").exists()
    assert "No pair passed the preregistered screen" in (out / "RESEARCH_REPORT.md").read_text()


@pytest.mark.parametrize("argv", [
    ["--demo", "--train-fraction", "0.95"],
    ["--demo", "--fold-bars", "5"],
    ["--demo", "--local-dir", "."],
    ["--demo", "--funding-file", "x.csv"],
    ["--funding-file", "x.csv", "--market", "spot"],
    ["--demo", "--pairs", "AAA:AAA"],
    ["--local-dir", "/nonexistent/dir", "--pairs", "AAA:BBB"],
])
def test_argument_validation_exits_with_usage_error(argv, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main([*argv, "--output", str(tmp_path), "--quiet"])
    assert exc.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_parse_pairs_normalizes_and_rejects_duplicates():
    assert parse_pairs(" ethusdt:btcusdt , SOLUSDT:AVAXUSDT") == [("ETHUSDT", "BTCUSDT"),
                                                                ("SOLUSDT", "AVAXUSDT")]
    with pytest.raises(ValueError, match="duplicate"):
        parse_pairs("A:B,A:B")
    with pytest.raises(ValueError):
        parse_pairs("A:B:C")
