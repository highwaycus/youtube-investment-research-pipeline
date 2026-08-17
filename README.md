from __future__ import annotations

import logging
import math
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
MARKET_DB_PATH = ROOT / "market_features.db"

# Stable classification map. Unknown assets fall back to SPY and carry a flag.
SECTOR_PROXY: Dict[str, str] = {
    "NVDA": "SOXX", "AMD": "SOXX", "AVGO": "SOXX", "INTC": "SOXX",
    "MU": "SOXX", "MRVL": "SOXX", "QCOM": "SOXX", "ADI": "SOXX",
    "NXPI": "SOXX", "ON": "SOXX", "ARM": "SOXX", "ASML": "SOXX",
    "TSM": "SOXX", "2330.TW": "SOXX", "2454.TW": "SOXX",
    "2408.TW": "SOXX", "2344.TW": "SOXX", "2303.TW": "SOXX",
    "000660.KS": "SOXX", "005930.KS": "SOXX", "AMKR": "SOXX",
    "WDC": "SOXX", "STX": "SOXX", "SNDK": "SOXX", "LITE": "SOXX",
    "COHR": "SOXX", "CRDO": "SOXX", "AAOI": "SOXX", "AXTI": "SOXX",
    "MSFT": "IGV", "ORCL": "IGV", "CRM": "IGV", "ADBE": "IGV",
    "PLTR": "IGV", "CRWD": "IGV", "PANW": "IGV", "NET": "IGV",
    "DDOG": "IGV", "NOW": "IGV", "SNOW": "IGV", "MDB": "IGV",
    "OKTA": "IGV", "TEAM": "IGV", "WDAY": "IGV", "INTU": "IGV",
    "META": "XLC", "GOOG": "XLC", "GOOGL": "XLC", "NFLX": "XLC",
    "TTD": "XLC", "0700.HK": "XLC",
    "AAPL": "XLK", "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY",
    "LOW": "XLY", "COST": "XLP", "WMT": "XLP", "KO": "XLP",
    "UNH": "XLV", "LLY": "XLV", "NVO": "XLV", "CVS": "XLV",
    "MS": "XLF", "GS": "XLF", "BA": "XLI", "RTX": "XLI",
    "LMT": "XLI", "NOC": "XLI", "AVAV": "XLI",
    "XOM": "XLE", "CVX": "XLE", "VST": "XLU",
}

NON_EQUITIES = {
    "SPY", "QQQ", "IWM", "SOXX", "SOXL", "SOXS", "IGV", "TLT", "GLD",
    "SLV", "UUP", "XLE", "XLK", "XLF", "XLV", "XLI", "XLY", "XLP",
    "XLU", "XLC", "EEM", "EWZ", "REMX", "CIBR", "BITO", "USO", "UNG",
    "0050.TW",
}

TICKER_ALIASES = {
    "BRK.A": "BRK-A", "BRK.B": "BRK-B",
    "BF.A": "BF-A", "BF.B": "BF-B",
}
NON_US_SUFFIXES = (
    ".TW", ".TWO", ".HK", ".KS", ".KQ", ".T", ".L", ".TO", ".V",
    ".AX", ".SI", ".SS", ".SZ", ".DE", ".PA", ".MI", ".AS",
)
KNOWN_INVALID_TICKERS = {"VRTV", "CWAV"}


def normalize_market_ticker(value: Any) -> Optional[str]:
    """Return a Yahoo-compatible US symbol, or None before any download."""
    ticker = str(value or "").upper().strip().replace("$", "")
    ticker = TICKER_ALIASES.get(ticker, ticker)
    if (
        not ticker
        or ticker in KNOWN_INVALID_TICKERS
        or ticker.endswith(NON_US_SUFFIXES)
        or ticker.startswith("^")
        or ticker.endswith("-USD")
        or ticker.endswith("=F")
    ):
        return None
    if not all(character.isalnum() or character == "-" for character in ticker):
        return None
    return ticker

PRICE_FEATURES = [
    "return_5d", "return_20d", "return_60d", "return_120d",
    "distance_to_ma20", "distance_to_ma60", "volatility_20d",
    "volatility_60d", "drawdown_252d", "relative_spy_20d",
    "relative_spy_60d", "relative_spy_120d", "sector_return_20d",
    "sector_return_60d", "sector_return_120d", "relative_sector_20d",
    "relative_sector_60d", "relative_sector_120d", "spy_return_20d",
    "spy_return_60d", "spy_volatility_20d", "sector_proxy_is_market",
]
FUNDAMENTAL_FEATURES = [
    "trailing_pe_point_in_time", "earnings_yield_point_in_time",
    "last_eps_surprise", "eps_yoy_growth", "days_since_earnings",
    "positive_trailing_eps",
]
MARKET_NUMERIC_FEATURES = PRICE_FEATURES + FUNDAMENTAL_FEATURES


def connect_market_db(path: Path = MARKET_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS daily_prices (
            ticker TEXT NOT NULL, price_date TEXT NOT NULL, close REAL NOT NULL,
            PRIMARY KEY (ticker, price_date)
        );
        CREATE TABLE IF NOT EXISTS earnings_events (
            ticker TEXT NOT NULL, report_date TEXT NOT NULL,
            reported_eps REAL, estimated_eps REAL, surprise_pct REAL,
            PRIMARY KEY (ticker, report_date)
        );
        CREATE TABLE IF NOT EXISTS fetch_status (
            data_type TEXT NOT NULL, ticker TEXT NOT NULL, fetched_at TEXT NOT NULL,
            first_date TEXT, last_date TEXT, status TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (data_type, ticker)
        );
        """
    )
    return conn


def _status(
    conn: sqlite3.Connection, kind: str, ticker: str, status: str,
    first: Optional[str] = None, last: Optional[str] = None, detail: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO fetch_status VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(data_type,ticker) DO UPDATE SET
          fetched_at=excluded.fetched_at, first_date=excluded.first_date,
          last_date=excluded.last_date, status=excluded.status,
          detail=excluded.detail
        """,
        (kind, ticker, datetime.now(timezone.utc).isoformat(),
         first, last, status, detail[:500]),
    )


def _needs_prices(
    conn: sqlite3.Connection, ticker: str, start: date, end: date
) -> bool:
    row = conn.execute(
        "SELECT MIN(price_date),MAX(price_date) FROM daily_prices WHERE ticker=?",
        (ticker,),
    ).fetchone()
    if not row or not row[0]:
        return True
    return (
        date.fromisoformat(row[0]) > start + timedelta(days=7)
        or date.fromisoformat(row[1]) < end - timedelta(days=5)
    )


def _close_series(data: pd.DataFrame, ticker: str, single: bool) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        top = set(map(str, data.columns.get_level_values(0)))
        if ticker in top and "Close" in data[ticker]:
            return data[ticker]["Close"].dropna()
        if "Close" in top and ticker in data["Close"]:
            return data["Close"][ticker].dropna()
    if single and "Close" in data:
        return data["Close"].dropna()
    return pd.Series(dtype=float)


def refresh_prices(
    conn: sqlite3.Connection, tickers: Sequence[str], start: date, end: date
) -> Dict[str, Any]:
    requested = sorted({str(x).upper() for x in tickers if x})
    pending = [x for x in requested if _needs_prices(conn, x, start, end)]
    downloaded: List[str] = []
    failed: Dict[str, str] = {}
    for offset in range(0, len(pending), 20):
        batch = pending[offset:offset + 20]
        try:
            data = yf.download(
                batch if len(batch) > 1 else batch[0],
                start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=True, progress=False, group_by="ticker",
                threads=False, timeout=25,
            )
        except Exception as exc:
            for ticker in batch:
                failed[ticker] = f"{type(exc).__name__}: {exc}"
                _status(conn, "price", ticker, "failed", detail=failed[ticker])
            conn.commit()
            continue
        for ticker in batch:
            series = _close_series(data, ticker, len(batch) == 1)
            rows = []
            for stamp, value in series.items():
                try:
                    number = float(value)
                    if math.isfinite(number) and number > 0:
                        rows.append(
                            (ticker, pd.Timestamp(stamp).date().isoformat(), number)
                        )
                except (TypeError, ValueError):
                    pass
            if rows:
                conn.executemany(
                    """
                    INSERT INTO daily_prices VALUES (?,?,?)
                    ON CONFLICT(ticker,price_date) DO UPDATE SET close=excluded.close
                    """,
                    rows,
                )
                _status(conn, "price", ticker, "ok", rows[0][1], rows[-1][1])
                downloaded.append(ticker)
            else:
                failed[ticker] = "No adjusted close data returned."
                _status(conn, "price", ticker, "failed", detail=failed[ticker])
        conn.commit()
    return {
        "requested": len(requested),
        "already_cached": len(requested) - len(pending),
        "downloaded": len(downloaded),
        "failed_count": len(failed),
        "failed_tickers": sorted(failed),
    }


def _equity(ticker: str) -> bool:
    return (
        ticker not in NON_EQUITIES and not ticker.startswith("^")
        and not ticker.endswith("-USD") and not ticker.endswith("=F")
    )


def _fresh_earnings(conn: sqlite3.Connection, ticker: str) -> bool:
    row = conn.execute(
        """
        SELECT fetched_at,status FROM fetch_status
        WHERE data_type='earnings' AND ticker=?
        """,
        (ticker,),
    ).fetchone()
    if not row or row[1] == "failed":
        return False
    try:
        stamp = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - stamp < timedelta(days=30)
    except ValueError:
        return False


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def refresh_earnings(
    conn: sqlite3.Connection, ticker_counts: Dict[str, int]
) -> Dict[str, Any]:
    requested = sorted(
        ticker for ticker, count in ticker_counts.items()
        if count >= 2 and _equity(ticker)
    )
    pending = [x for x in requested if not _fresh_earnings(conn, x)]
    downloaded: List[str] = []
    failed: List[str] = []
    for ticker in pending:
        try:
            events = yf.Ticker(ticker).get_earnings_dates(limit=20)
            rows = []
            if events is not None and not events.empty:
                for stamp, item in events.iterrows():
                    surprise = _number(item.get("Surprise(%)"))
                    if surprise is not None and abs(surprise) > 2:
                        surprise /= 100.0
                    rows.append((
                        ticker, pd.Timestamp(stamp).date().isoformat(),
                        _number(item.get("Reported EPS")),
                        _number(item.get("EPS Estimate")), surprise,
                    ))
            if rows:
                conn.executemany(
                    """
                    INSERT INTO earnings_events VALUES (?,?,?,?,?)
                    ON CONFLICT(ticker,report_date) DO UPDATE SET
                      reported_eps=excluded.reported_eps,
                      estimated_eps=excluded.estimated_eps,
                      surprise_pct=excluded.surprise_pct
                    """,
                    rows,
                )
                _status(conn, "earnings", ticker, "ok",
                        min(x[1] for x in rows), max(x[1] for x in rows))
                downloaded.append(ticker)
            else:
                failed.append(ticker)
                _status(conn, "earnings", ticker, "unavailable",
                        detail="No earnings history returned.")
        except Exception as exc:
            failed.append(ticker)
            _status(conn, "earnings", ticker, "failed",
                    detail=f"{type(exc).__name__}: {exc}")
        conn.commit()
    return {
        "requested": len(requested),
        "already_cached": len(requested) - len(pending),
        "downloaded": len(downloaded),
        "failed_count": len(failed),
        "failed_tickers": failed,
    }


def sector_proxy(ticker: str) -> str:
    ticker = normalize_market_ticker(ticker) or "SPY"
    sector_etfs = {
        "SOXX", "IGV", "XLE", "XLK", "XLF", "XLV", "XLI", "XLY",
        "XLP", "XLU", "XLC",
    }
    return ticker if ticker in sector_etfs else SECTOR_PROXY.get(ticker, "SPY")


def refresh_market_cache(
    observations: pd.DataFrame, market_db_path: Path = MARKET_DB_PATH
) -> Dict[str, Any]:
    if observations.empty:
        return {"price": {}, "earnings": {}}
    observations = observations.copy()
    observations["ticker"] = observations["ticker"].map(normalize_market_ticker)
    observations = observations[observations["ticker"].notna()].copy()
    if observations.empty:
        return {"price": {}, "earnings": {}}
    stamps = pd.to_datetime(observations["published_date"])
    start = stamps.min().date() - timedelta(days=400)
    end = max(stamps.max().date(), datetime.now(timezone.utc).date())
    counts = {
        str(key).upper(): int(value)
        for key, value in observations["ticker"].value_counts().items()
    }
    tickers = set(counts) | {sector_proxy(x) for x in counts} | {"SPY"}
    conn = connect_market_db(market_db_path)
    try:
        prices = refresh_prices(conn, sorted(tickers), start, end)
        earnings = refresh_earnings(conn, counts)
        return {"price": prices, "earnings": earnings}
    finally:
        conn.close()


def _history(conn: sqlite3.Connection, ticker: str) -> pd.Series:
    rows = conn.execute(
        "SELECT price_date,close FROM daily_prices WHERE ticker=? ORDER BY price_date",
        (ticker,),
    ).fetchall()
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return pd.Series(
        [float(x[1]) for x in rows],
        index=pd.to_datetime([x[0] for x in rows]),
        dtype=float,
    )


def _ret(series: pd.Series, days: int) -> float:
    if len(series) <= days or float(series.iloc[-days - 1]) <= 0:
        return np.nan
    return float(series.iloc[-1] / series.iloc[-days - 1] - 1)


def _vol(series: pd.Series, days: int) -> float:
    values = series.pct_change().dropna().tail(days)
    return (
        float(values.std(ddof=1) * math.sqrt(252))
        if len(values) >= max(10, days // 2) else np.nan
    )


def _price_metrics(series: pd.Series) -> Dict[str, float]:
    keys = [
        "return_5d", "return_20d", "return_60d", "return_120d",
        "distance_to_ma20", "distance_to_ma60", "volatility_20d",
        "volatility_60d", "drawdown_252d",
    ]
    if series.empty:
        return {key: np.nan for key in keys}
    last = float(series.iloc[-1])
    result = {
        "return_5d": _ret(series, 5), "return_20d": _ret(series, 20),
        "return_60d": _ret(series, 60), "return_120d": _ret(series, 120),
        "distance_to_ma20": np.nan, "distance_to_ma60": np.nan,
        "volatility_20d": _vol(series, 20),
        "volatility_60d": _vol(series, 60), "drawdown_252d": np.nan,
    }
    for days in (20, 60):
        if len(series) >= days:
            average = float(series.tail(days).mean())
            result[f"distance_to_ma{days}"] = last / average - 1 if average else np.nan
    window = series.tail(252)
    if len(window) >= 60:
        result["drawdown_252d"] = last / float(window.max()) - 1
    return result


def _fundamentals(
    conn: sqlite3.Connection, ticker: str, stamp: pd.Timestamp, price: float
) -> Dict[str, float]:
    result = {key: np.nan for key in FUNDAMENTAL_FEATURES}
    rows = conn.execute(
        """
        SELECT report_date,reported_eps,estimated_eps,surprise_pct
        FROM earnings_events WHERE ticker=? AND report_date<?
        ORDER BY report_date DESC LIMIT 8
        """,
        (ticker, stamp.date().isoformat()),
    ).fetchall()
    if not rows:
        return result
    result["days_since_earnings"] = float(
        (stamp.date() - date.fromisoformat(rows[0][0])).days
    )
    result["last_eps_surprise"] = (
        float(rows[0][3]) if rows[0][3] is not None else np.nan
    )
    eps = [float(x[1]) for x in rows if x[1] is not None]
    if len(eps) >= 4:
        trailing = sum(eps[:4])
        result["positive_trailing_eps"] = float(trailing > 0)
        if price > 0 and trailing > 0:
            result["trailing_pe_point_in_time"] = price / trailing
            result["earnings_yield_point_in_time"] = trailing / price
    if len(eps) >= 5 and abs(eps[4]) > 1e-9:
        result["eps_yoy_growth"] = (eps[0] - eps[4]) / abs(eps[4])
    return result


def build_market_feature_frame(
    observations: pd.DataFrame, market_db_path: Path = MARKET_DB_PATH
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    result = observations.copy()
    for column in MARKET_NUMERIC_FEATURES:
        result[column] = np.nan
    if result.empty:
        return result, {"rows": 0, "price_feature_coverage": 0.0}
    conn = connect_market_db(market_db_path)
    cache: Dict[str, pd.Series] = {}

    def prices(ticker: str, stamp: pd.Timestamp) -> pd.Series:
        if ticker not in cache:
            cache[ticker] = _history(conn, ticker)
        # Exact intraday publication time is unavailable. Strictly prior close
        # avoids leaking the same day's close into a signal.
        return cache[ticker][cache[ticker].index < stamp.normalize()]

    try:
        for index, row in result.iterrows():
            ticker = normalize_market_ticker(row["ticker"])
            stamp = pd.Timestamp(row["published_date"])
            asset = prices(ticker, stamp) if ticker else pd.Series(
                dtype=float, index=pd.DatetimeIndex([])
            )
            spy = prices("SPY", stamp)
            proxy = sector_proxy(ticker or "SPY")
            sector = prices(proxy, stamp)
            a, s, i = _price_metrics(asset), _price_metrics(spy), _price_metrics(sector)
            values: Dict[str, float] = dict(a)
            for days in (20, 60, 120):
                values[f"relative_spy_{days}d"] = (
                    a[f"return_{days}d"] - s[f"return_{days}d"]
                )
                values[f"sector_return_{days}d"] = i[f"return_{days}d"]
                values[f"relative_sector_{days}d"] = (
                    a[f"return_{days}d"] - i[f"return_{days}d"]
                )
            values["spy_return_20d"] = s["return_20d"]
            values["spy_return_60d"] = s["return_60d"]
            values["spy_volatility_20d"] = s["volatility_20d"]
            values["sector_proxy_is_market"] = float(proxy == "SPY")
            if not asset.empty:
                values.update(_fundamentals(
                    conn, ticker, stamp, float(asset.iloc[-1])
                ))
            for column, value in values.items():
                result.at[index, column] = value
    finally:
        conn.close()
    return result, {
        "rows": int(len(result)),
        "price_feature_coverage": float(result["return_20d"].notna().mean()),
        "valuation_coverage": float(
            result["trailing_pe_point_in_time"].notna().mean()
        ),
        "earnings_feature_coverage": float(
            result["days_since_earnings"].notna().mean()
        ),
        "sector_proxy_coverage": float(
            (result["sector_proxy_is_market"] == 0).mean()
        ),
        "point_in_time_policy": (
            "Prices use completed sessions strictly before the signal date. "
            "Earnings use only earlier report dates. Missing historical "
            "fundamentals remain missing; current values are never backfilled."
        ),
    }
