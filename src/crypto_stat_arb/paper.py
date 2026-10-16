"""A read-only public-data signal monitor. It never places trades."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import __version__
from .model import PairFit, SignalState, next_signal, spread_zscore

URLS = {
    "spot": "https://api.binance.com/api/v3/klines",
    "um": "https://fapi.binance.com/fapi/v1/klines",
}


def latest_closed(session: requests.Session, symbol: str, market: str, now_ms: int) -> pd.DataFrame:
    params: dict[str, str | int] = {"symbol": symbol, "interval": "1h", "limit": 1000}
    response = session.get(URLS[market], params=params, timeout=30)
    response.raise_for_status()
    records = response.json()
    if not isinstance(records, list) or not records:
        raise ValueError(f"empty or invalid market response for {symbol}")
    result = []
    for r in records:
        if len(r) < 7:
            raise ValueError(f"bad candle row for {symbol}")
        if int(r[6]) > now_ms:
            continue
        result.append((int(r[0]), float(r[4])))
    if not result:
        raise ValueError(f"no completed candles for {symbol}")
    index = pd.to_datetime([t for t, _ in result], unit="ms", utc=True)
    data = pd.DataFrame({symbol: [c for _, c in result]}, index=index)
    if data.index.has_duplicates or data[symbol].isna().any() or (data[symbol] <= 0).any():
        raise ValueError(f"invalid candles for {symbol}")
    return data.sort_index()


def check_state(state: dict, signature: str, last: pd.Timestamp) -> None:
    if state["signature"] != signature:
        raise ValueError("saved state belongs to a different model; use a fresh state file")
    previous = pd.Timestamp(state["last_bar_utc"])
    if last < previous:
        raise ValueError("public feed returned earlier data than saved state")
    if last > previous + pd.Timedelta(hours=1):
        raise ValueError("feed missed one or more closed hourly candles; "
                         "restart paper monitor with a new state")


def run_once(summary_path: Path, state_path: Path, market: str,
             now_ms: int | None = None) -> dict:
    report = json.loads(Path(summary_path).read_text())
    if report.get("demo") or not report.get("pair"):
        raise ValueError("paper monitor needs a real-data summary with a selected pair")
    if market != report["market_code"]:
        raise ValueError("paper feed market must match the fitted market")
    pair = report["pair"]
    config = report["parameters"]
    if config["lookback"] + 2 > 1000:
        raise ValueError("REST feed limit requires lookback <= 998")
    signature = sha256(json.dumps({"pair": pair, "parameters": config,
                                   "market": market}, sort_keys=True).encode()).hexdigest()
    now_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    with requests.Session() as session:
        y = latest_closed(session, pair["y"], market, now_ms)
        x = latest_closed(session, pair["x"], market, now_ms)
    prices = y.join(x, how="inner").dropna()
    if len(prices) < config["lookback"] + 2:
        raise ValueError("not enough completed candles to calculate the z-score")
    expected = pd.date_range(prices.index[0], prices.index[-1], freq="h", tz="UTC")
    if not prices.index.equals(expected):
        raise ValueError("market feed has a missing candle")
    fit = PairFit(pair["y"], pair["x"], pair["alpha"], pair["beta"],
                  pair["pvalue"], pair["qvalue"])
    z = float(spread_zscore(prices, fit, config["lookback"]).iloc[-1])
    if not np.isfinite(z):
        raise ValueError("latest residual z-score is undefined")
    latest = prices.index[-1]
    if state_path.exists():
        state = json.loads(state_path.read_text())
        check_state(state, signature, latest)
        if latest == pd.Timestamp(state["last_bar_utc"]):
            return {"status": "already processed", "last_bar_utc": latest.isoformat(),
                    "position": state["position"]}
        before = SignalState(state["position"], state["age"], state["wait"])
        updated = next_signal(z, before, **{k: config[k] for k in
                              ("entry", "exit", "stop", "max_holding", "cooldown")})
        status = "new signal" if updated.position != before.position else "no change"
    else:
        updated = SignalState()
        status = "initialized flat; first new signal after next completed candle"
    state = {"signature": signature, "last_bar_utc": latest.isoformat(), **asdict(updated)}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(state_path)
    return {"status": status, "last_bar_utc": latest.isoformat(),
            "zscore": z, "position": updated.position,
            "meaning": {0: "flat", 1: "long Y / short X", -1: "short Y / long X"}[updated.position],
            "disclaimer": "research signal only; not an executed order or tracked P&L"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Read-only hourly paper signals from public Binance candles")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--summary", type=Path, required=True, help="a real-data run's summary.json")
    p.add_argument("--state", type=Path, default=Path("results/paper_state.json"))
    p.add_argument("--market", choices=["spot", "um"], default="spot")
    p.add_argument("--watch", action="store_true", help="keep checking on a fixed cadence")
    p.add_argument("--poll-seconds", type=int, default=60)
    args = p.parse_args(argv)
    if args.poll_seconds < 15:
        p.error("poll-seconds must be >= 15")
    try:
        while True:
            result = run_once(args.summary, args.state, args.market)
            print(json.dumps(result, indent=2), flush=True)
            if not args.watch:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("stopped")
    except (ValueError, FileNotFoundError, requests.RequestException) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
