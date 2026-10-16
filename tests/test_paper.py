import json
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from crypto_stat_arb.model import SignalState, next_signal, positions_from_z
from crypto_stat_arb.paper import check_state, latest_closed


def test_live_state_machine_matches_research_signal():
    z = pd.Series([-2.3, -2.1, -1, -.3, 2.3, 4.5, 3, 2, 0.1])
    state = SignalState()
    states = []
    for value in z:
        state = next_signal(value, state, cooldown=1)
        states.append(state.position)
    assert states == positions_from_z(z, cooldown=1).tolist()


def test_state_rejects_skipped_bars():
    previous = pd.Timestamp("2026-01-01T00:00Z")
    state = {"signature": "test", "last_bar_utc": previous.isoformat()}
    with pytest.raises(ValueError, match="missed"):
        check_state(state, "test", previous + pd.Timedelta(hours=2))
    with pytest.raises(ValueError, match="different model"):
        check_state(state, "other", previous)


def test_latest_closed_ignores_unfinished_candle():
    base = 1735689600000
    records = [[base, "1", "1", "1", "1.2", "0", base + 3600000 - 1],
               [base + 3600000, "1", "1", "1", "1.9", "0", base + 7200000 - 1]]
    session = Mock()
    response = Mock()
    response.json.return_value = records
    session.get.return_value = response
    f = latest_closed(session, "BTCUSDT", "spot", base + 3600000 + 2)
    assert len(f) == 1
    assert f.BTCUSDT.iloc[0] == 1.2


def test_paper_session_persists_and_does_not_repeat(tmp_path):
    import numpy as np

    from crypto_stat_arb.paper import run_once

    summary = tmp_path / "summary.json"
    state = tmp_path / "state.json"
    summary.write_text(json.dumps({
        "demo": False, "market_code": "spot", "pair": {
            "y": "AAA", "x": "BBB", "alpha": .1, "beta": 1.0,
            "pvalue": .001, "qvalue": .002},
        "parameters": {"lookback": 25, "entry": 2., "exit": .5, "stop": 4.,
                       "max_holding": 168, "cooldown": 12}}))
    index = pd.date_range("2026-01-01", periods=70, freq="h", tz="UTC")
    x = pd.Series(np.exp(np.log(100) + np.cumsum(np.sin(np.arange(70))*.002)), index=index)
    data_a = pd.DataFrame({"AAA": np.exp(.1) * x.values}, index=index)
    data_b = pd.DataFrame({"BBB": x.values}, index=index)
    with patch("crypto_stat_arb.paper.latest_closed", side_effect=[data_a, data_b, data_a, data_b]):
        first = run_once(summary, state, "spot")
        again = run_once(summary, state, "spot")
    assert first["position"] == 0
    assert "initialized" in first["status"]
    assert again["status"] == "already processed"
    assert json.loads(state.read_text())["last_bar_utc"] == index[-1].isoformat()
