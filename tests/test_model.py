import pandas as pd
import pytest

from crypto_stat_arb.cli import parse_pairs, synthetic_market
from crypto_stat_arb.model import PairFit, positions_from_z, screen_pairs, spread_zscore


def test_screen_formation_cannot_see_future():
    _, c, pairs = synthetic_market(n=1700)
    split = 1100
    original = screen_pairs(c, pairs, split)
    changed = c.copy()
    changed.iloc[split:, :] *= 100
    pd.testing.assert_frame_equal(original, screen_pairs(changed, pairs, split))


def test_zscore_reference_does_not_contain_current_close():
    _, c, _ = synthetic_market(n=420)
    fit = PairFit("DEMO_Y", "DEMO_X", .7, .97, 0, 0)
    z = spread_zscore(c, fit, 30)
    changed = c.copy()
    changed.iloc[200:, :] *= 5
    newer = spread_zscore(changed, fit, 30)
    pd.testing.assert_series_equal(z.iloc[:199], newer.iloc[:199])


def test_entry_exit_stop_and_cooldown():
    z = pd.Series([-2.1, -1., -.4, 2.2, 5., 2.1, 2.2, 0., 2.1])
    out = positions_from_z(z, cooldown=2)
    assert out.tolist() == [1, 1, 0, 0, 0, -1, -1, 0, 0]


def test_holding_limit():
    z = pd.Series([-2.1] * 6)
    assert positions_from_z(z, max_holding=3, cooldown=2).tolist() == [1, 1, 1, 0, 0, 0]


def test_validation():
    with pytest.raises(ValueError):
        parse_pairs("AAA:AAA")
    with pytest.raises(ValueError):
        positions_from_z(pd.Series([2]), exit=3)
