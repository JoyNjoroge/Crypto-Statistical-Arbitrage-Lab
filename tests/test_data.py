import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from crypto_stat_arb.data import load_panel, parse_kline_archive, read_funding, read_local_symbol


def kline_zip(ts):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("k.csv", f"{ts},1.2,1.3,1.1,1.25,5,{ts},6,1,2,3,0\n")
    return b.getvalue()


@pytest.mark.parametrize("stamp", [1735689600000, 1735689600000000])
def test_binance_time_units(stamp):
    f = parse_kline_archive(kline_zip(stamp))
    assert f.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert f.open.iloc[0] == pytest.approx(1.2)


def test_local_files_and_gap_detection(tmp_path):
    index = pd.date_range("2025-01-01", periods=320, freq="h", tz="UTC")
    for symbol in ("AAAUSDT", "BBBUSDT"):
        pd.DataFrame({"timestamp": index, "open": np.ones(320), "close": np.ones(320)}).to_csv(
            tmp_path / f"{symbol}.csv", index=False)
    o, c = load_panel(["AAAUSDT", "BBBUSDT"], "2025-01", "2025-01", tmp_path, local_dir=tmp_path)
    assert len(o) == len(c) == 320
    bad = pd.read_csv(tmp_path / "BBBUSDT.csv").drop(index=10)
    bad.to_csv(tmp_path / "BBBUSDT.csv", index=False)
    with pytest.raises(ValueError, match="missing or non-hourly"):
        load_panel(["AAAUSDT", "BBBUSDT"], "2025-01", "2025-01", tmp_path, local_dir=tmp_path)


def test_duplicate_timestamp_rejected(tmp_path):
    path = tmp_path / "A.csv"
    pd.DataFrame({"timestamp": ["2025-01-01T00:00Z"] * 2,
                  "open": [10, 10], "close": [10, 10]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_local_symbol(path)


def test_funding_decimal_signs_and_alignment(tmp_path):
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    p = tmp_path / "rates.csv"
    pd.DataFrame({"timestamp": [idx[1], idx[1]],
                  "symbol": ["AAA", "BBB"], "funding_rate": [.0001, -.0002]}).to_csv(p, index=False)
    f = read_funding(p, idx, ["AAA", "BBB"])
    assert f.loc[idx[1], "AAA"] == pytest.approx(.0001)
    assert f.loc[idx[2], "AAA"] == 0
