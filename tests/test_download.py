"""The archive downloader, exercised against a fake HTTP session. Nothing here touches the network."""
import hashlib
import io
import zipfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from crypto_stat_arb.data import (
    BASE_URL,
    fetch_archive,
    load_symbol,
    months,
    parse_kline_archive,
    verify_checksum,
)

START_MS = 1735689600000  # 2025-01-01T00:00Z


def month_zip(n_hours: int, first_ms: int = START_MS) -> bytes:
    rows = []
    for k in range(n_hours):
        t = first_ms + k * 3_600_000
        rows.append(f"{t},100,101,99,100.5,5,{t + 3_599_999},6,1,2,3,0")
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("k.csv", "\n".join(rows) + "\n")
    return b.getvalue()


class FakeSession:
    """Serves a dict of url -> bytes | status code, and remembers what was asked for."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        hit = self.routes.get(url, 404)
        response = Mock(spec=requests.Response)
        if isinstance(hit, int):
            response.status_code = hit
            error = requests.HTTPError(f"{hit} for {url}", response=response)
            response.raise_for_status.side_effect = error
            return response
        if isinstance(hit, Exception):
            raise hit
        response.status_code = 200
        response.content = hit
        response.text = hit.decode(errors="replace")
        response.raise_for_status.return_value = None
        return response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def routes_for(symbol: str, month: str, blob: bytes, market: str = "spot",
               checksum: bool = True) -> dict:
    url = f"{BASE_URL}/{market}/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
    out = {url: blob}
    if checksum:
        digest = hashlib.sha256(blob).hexdigest()
        out[url + ".CHECKSUM"] = f"{digest}  {symbol}-1h-{month}.zip\n".encode()
    return out


def test_months_validation():
    assert len(months("2025-01", "2025-03")) == 3
    with pytest.raises(ValueError, match="YYYY-MM"):
        months("not-a-month", "2025-03")
    with pytest.raises(ValueError, match="before"):
        months("2025-03", "2025-01")


def test_verify_checksum_accepts_sha256sum_format_and_rejects_mismatch():
    blob = b"abc"
    good = hashlib.sha256(blob).hexdigest()
    verify_checksum(blob, f"{good}  file.zip\n", "file.zip")
    verify_checksum(blob, good.upper(), "file.zip")
    with pytest.raises(ValueError, match="mismatch"):
        verify_checksum(b"abd", f"{good}  file.zip", "file.zip")
    with pytest.raises(ValueError, match="unreadable"):
        verify_checksum(blob, "not-a-hash", "file.zip")


def test_download_verifies_and_caches(tmp_path):
    blob = month_zip(744)
    session = FakeSession(routes_for("BTCUSDT", "2025-01", blob))
    with patch("crypto_stat_arb.data.requests.Session", return_value=session):
        frame = load_symbol("btcusdt", "2025-01", "2025-01", tmp_path)
        assert len(frame) == 744
        assert frame.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
        assert (tmp_path / "spot" / "BTCUSDT" / "BTCUSDT-1h-2025-01.zip").read_bytes() == blob
        assert len(session.calls) == 2  # archive + checksum
        load_symbol("BTCUSDT", "2025-01", "2025-01", tmp_path)
        assert len(session.calls) == 2  # second call served from cache


def test_corrupted_download_is_rejected_and_not_cached(tmp_path):
    blob = month_zip(10)
    routes = routes_for("BTCUSDT", "2025-01", blob)
    routes[f"{BASE_URL}/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2025-01.zip"] = blob[:-5]
    session = FakeSession(routes)
    with patch("crypto_stat_arb.data.requests.Session", return_value=session), \
            pytest.raises(ValueError, match="checksum mismatch"):
        load_symbol("BTCUSDT", "2025-01", "2025-01", tmp_path)
    assert not (tmp_path / "spot" / "BTCUSDT" / "BTCUSDT-1h-2025-01.zip").exists()


def test_missing_checksum_file_is_tolerated_with_warning(tmp_path, caplog):
    blob = month_zip(10)
    session = FakeSession(routes_for("ETHUSDT", "2025-02", blob, checksum=False))
    with patch("crypto_stat_arb.data.requests.Session", return_value=session), \
            caplog.at_level("WARNING", logger="crypto_stat_arb.data"):
        load_symbol("ETHUSDT", "2025-02", "2025-02", tmp_path)
    assert "unverified" in caplog.text


def test_verification_can_be_disabled(tmp_path):
    blob = month_zip(10)
    session = FakeSession(routes_for("ETHUSDT", "2025-02", blob, checksum=False))
    with patch("crypto_stat_arb.data.requests.Session", return_value=session):
        load_symbol("ETHUSDT", "2025-02", "2025-02", tmp_path, verify=False)
    assert len(session.calls) == 1


def test_unpublished_month_is_a_clear_error(tmp_path):
    session = FakeSession({})
    with patch("crypto_stat_arb.data.requests.Session", return_value=session), \
            pytest.raises(FileNotFoundError, match="No archive for BTCUSDT 2031-01"):
        load_symbol("BTCUSDT", "2031-01", "2031-01", tmp_path)


def test_transient_errors_are_retried_then_raised(tmp_path):
    url = f"{BASE_URL}/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2025-01.zip"
    session = FakeSession({url: requests.ConnectionError("reset")})
    with patch("crypto_stat_arb.data.requests.Session", return_value=session), \
            patch("crypto_stat_arb.data.time.sleep") as sleep, \
            pytest.raises(requests.ConnectionError):
        load_symbol("BTCUSDT", "2025-01", "2025-01", tmp_path)
    assert session.calls.count(url) == 3
    assert sleep.call_count == 2


def test_fetch_archive_writes_atomically(tmp_path):
    blob = month_zip(3)
    url = f"{BASE_URL}/spot/monthly/klines/X/1h/X-1h-2025-01.zip"
    routes = {url: blob, url + ".CHECKSUM": hashlib.sha256(blob).hexdigest().encode()}
    target = tmp_path / "X-1h-2025-01.zip"
    fetch_archive(FakeSession(routes), url, target)
    assert target.read_bytes() == blob
    assert not target.with_suffix(".tmp").exists()


def test_unparseable_cached_archive_is_evicted(tmp_path):
    path = tmp_path / "spot" / "BTCUSDT" / "BTCUSDT-1h-2025-01.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a zip")
    with patch("crypto_stat_arb.data.requests.Session", return_value=FakeSession({})), \
            pytest.raises(zipfile.BadZipFile):
        load_symbol("BTCUSDT", "2025-01", "2025-01", tmp_path)
    assert not path.exists()


def test_invalid_symbol_and_market():
    with pytest.raises(ValueError, match="invalid symbol"):
        load_symbol("BT", "2025-01", "2025-01", "/tmp")
    with pytest.raises(ValueError, match="unknown market"):
        load_symbol("BTCUSDT", "2025-01", "2025-01", "/tmp", market="coin-m")


def test_archive_parser_drops_bad_rows_and_duplicates():
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("k.csv", f"open_time,open,high,low,close\n{START_MS},1,1,1,1,0,{START_MS},0,0,0,0,0\n"
                            f"{START_MS},2,2,2,2,0,{START_MS},0,0,0,0,0\n"
                            f"{START_MS + 3600000},0,2,2,2,0,{START_MS},0,0,0,0,0\n")
    frame = parse_kline_archive(b.getvalue())
    assert len(frame) == 1
    assert frame.open.iloc[0] == 2  # keep="last" on duplicate timestamps
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("a.csv", "")
        z.writestr("b.csv", "")
    with pytest.raises(ValueError, match="one CSV"):
        parse_kline_archive(empty.getvalue())
