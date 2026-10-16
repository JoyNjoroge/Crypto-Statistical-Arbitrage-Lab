"""Historical hourly candles from Binance's public archive or local CSV files."""
from __future__ import annotations

import hashlib
import io
import logging
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
MARKETS = {"spot": "spot", "um": "futures/um"}
BASE_URL = "https://data.binance.vision/data"
log = logging.getLogger(__name__)


def months(start: str, end: str) -> pd.PeriodIndex:
    try:
        left, right = pd.Period(start, freq="M"), pd.Period(end, freq="M")
    except ValueError as exc:
        raise ValueError("start/end must be YYYY-MM") from exc
    if right < left:
        raise ValueError("end cannot be before start")
    return pd.period_range(left, right, freq="M")


def parse_kline_archive(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("expected one CSV per archive")
        with zf.open(names[0]) as f:
            raw = pd.read_csv(f, names=COLUMNS, header=None, low_memory=False)
    raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce")
    raw = raw.dropna(subset=["open_time"])
    if raw.empty:
        raise ValueError("archive has no usable timestamps")
    timestamps = raw["open_time"].to_numpy(dtype="int64")
    units = np.where(timestamps > 10**14, timestamps // 1000, timestamps)
    index = pd.DatetimeIndex(pd.to_datetime(units, unit="ms", utc=True), name="timestamp")
    df = pd.DataFrame(index=index)
    for col in ("open", "close", "volume"):
        df[col] = pd.to_numeric(raw[col], errors="coerce").to_numpy()
    df = df.loc[(df.open > 0) & (df.close > 0)].copy()
    return df.loc[~df.index.duplicated(keep="last")].sort_index()


def _get(session: requests.Session, url: str, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if getattr(exc.response, "status_code", None) == 404:
                raise FileNotFoundError(f"No archive at {url}") from exc
            if attempt == retries - 1:
                raise
            log.warning("retrying %s after %s", url, exc.__class__.__name__)
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def verify_checksum(blob: bytes, checksum_text: str, name: str) -> None:
    """Binance publishes `<archive>.CHECKSUM` as `sha256  filename`, the sha256sum format."""
    parts = checksum_text.split()
    if len(parts) < 1 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
        raise ValueError(f"unreadable checksum file for {name}")
    actual = hashlib.sha256(blob).hexdigest()
    if actual.lower() != parts[0].lower():
        raise ValueError(f"checksum mismatch for {name}: expected {parts[0]}, got {actual}")


def fetch_archive(session: requests.Session, url: str, path: Path,
                  verify: bool = True) -> None:
    response = _get(session, url)
    blob = response.content
    if verify:
        try:
            verify_checksum(blob, _get(session, url + ".CHECKSUM").text, path.name)
        except FileNotFoundError:
            log.warning("no CHECKSUM published for %s; archive accepted unverified", path.name)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)


def load_symbol(symbol: str, start: str, end: str, cache_dir: Path, market: str = "spot",
                verify: bool = True) -> pd.DataFrame:
    if market not in MARKETS:
        raise ValueError(f"unknown market {market}")
    symbol = symbol.upper()
    if not re.fullmatch(r"[A-Z0-9]{4,25}", symbol):
        raise ValueError(f"invalid symbol {symbol}")
    cache_dir = Path(cache_dir) / market
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    period = months(start, end)
    with requests.Session() as session:
        for k, month in enumerate(period, start=1):
            name = f"{symbol}-1h-{month}.zip"
            path = cache_dir / symbol / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                url = f"{BASE_URL}/{MARKETS[market]}/monthly/klines/{symbol}/1h/{name}"
                log.info("downloading %s (%d/%d)", name, k, len(period))
                try:
                    fetch_archive(session, url, path, verify)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(f"No archive for {symbol} {month}: {url}") from exc
            try:
                frames.append(parse_kline_archive(path.read_bytes()))
            except Exception:
                path.unlink(missing_ok=True)
                raise
    out = pd.concat(frames).sort_index()
    return out.loc[~out.index.duplicated(keep="last")]


def _timestamps(series: pd.Series) -> pd.DatetimeIndex:
    numbers = pd.to_numeric(series, errors="coerce")
    if numbers.notna().all():
        stamp = numbers.astype("int64").to_numpy()
        millis = np.where(stamp > 10**14, stamp // 1000, stamp)
        return pd.DatetimeIndex(pd.to_datetime(millis, unit="ms", utc=True), name="timestamp")
    dt = pd.to_datetime(series, utc=True, errors="raise")
    return pd.DatetimeIndex(dt, name="timestamp")


def read_local_symbol(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"timestamp", "open", "close"}
    if not required.issubset(raw.columns):
        raise ValueError(f"{path} must contain timestamp,open,close")
    index = _timestamps(raw.timestamp)
    if index.has_duplicates:
        raise ValueError(f"duplicate timestamp in {path}")
    df = pd.DataFrame(index=index)
    for col in ("open", "close"):
        df[col] = pd.to_numeric(raw[col], errors="coerce").to_numpy()
    if df.isna().any().any() or (df <= 0).any().any():
        raise ValueError(f"invalid prices in {path}")
    return df.sort_index()


def load_panel(symbols: list[str], start: str, end: str, cache_dir: Path, market: str = "spot",
               local_dir: Path | None = None, verify: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = sorted(set(symbols))
    if local_dir is not None:
        data = {s: read_local_symbol(Path(local_dir) / f"{s}.csv") for s in symbols}
    else:
        data = {s: load_symbol(s, start, end, cache_dir, market, verify) for s in symbols}
    o = pd.concat({s: frame.open for s, frame in data.items()}, axis=1).dropna()
    c = pd.concat({s: frame.close for s, frame in data.items()}, axis=1).dropna()
    idx = o.index.intersection(c.index)
    if len(idx) < 300:
        raise ValueError("need at least 300 overlapping hourly candles")
    expected = pd.date_range(idx[0], idx[-1], freq="h", tz="UTC")
    if not idx.equals(expected):
        missing = expected.difference(idx)
        raise ValueError(f"missing or non-hourly candles: {len(missing)} missing; "
                         "fix data before running")
    return o.loc[idx], c.loc[idx]


def read_funding(path: Path, index: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if not {"timestamp", "symbol", "funding_rate"}.issubset(raw.columns):
        raise ValueError("funding CSV needs timestamp,symbol,funding_rate")
    raw["timestamp"] = _timestamps(raw["timestamp"])
    raw["symbol"] = raw["symbol"].str.upper()
    raw["funding_rate"] = pd.to_numeric(raw["funding_rate"], errors="raise")
    if not np.isfinite(raw.funding_rate).all():
        raise ValueError("nonfinite funding rate")
    if raw.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("duplicate funding event")
    if any(s not in raw.symbol.unique() for s in symbols):
        raise ValueError("funding file is missing a required symbol")
    funding = raw.pivot(index="timestamp", columns="symbol", values="funding_rate")
    if not funding.index.isin(index).all():
        raise ValueError("funding events outside candle timestamps")
    return funding.reindex(index=index, columns=symbols, fill_value=0).fillna(0.0)
