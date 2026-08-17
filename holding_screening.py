from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from market_features import NON_EQUITIES, sector_proxy
from research import FUNDAMENTALS_CACHE_PATH, _load_fundamentals_cache, market_evidence


LEVERAGED_OR_INVERSE = {
    "SOXL", "SOXS", "TQQQ", "SQQQ", "UPRO", "SPXU", "FAS", "FAZ",
    "LABU", "LABD", "NUGT", "DUST", "BITX", "BITI",
}

SECTOR_NAME_PROXY = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
}

# Livestream transcripts often mix local shares, ADRs, share classes, and stale
# symbols.  Keep the candidate universe limited to securities the user can
# actually trade in a US-only account, and collapse economically identical
# exposure before deciding whether something is new.
TICKER_ALIASES = {
    "BRK.A": "BRK-A", "BRK.B": "BRK-B",
    "BF.A": "BF-A", "BF.B": "BF-B",
    "GOOG": "GOOGL",
    "2330.TW": "TSM",
}
EXPOSURE_ALIASES = {
    "GOOG": "GOOGLE", "GOOGL": "GOOGLE",
    "BRK-A": "BERKSHIRE", "BRK-B": "BERKSHIRE",
    "TSM": "TSMC", "2330.TW": "TSMC",
}
NON_US_SUFFIXES = (
    ".TW", ".TWO", ".HK", ".KS", ".KQ", ".T", ".L", ".TO", ".V",
    ".AX", ".SI", ".SS", ".SZ", ".DE", ".PA", ".MI", ".AS",
)
KNOWN_INVALID_TICKERS = {"VRTV", "CWAV"}
US_EXCHANGES = {
    "NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BTS", "BATS",
    "NASDAQ", "NYSE", "NYSEARCA", "NYSEAMERICAN", "CBOE",
}


def normalize_ticker(value: Any) -> str:
    ticker = str(value or "").upper().strip().replace("$", "")
    return TICKER_ALIASES.get(ticker, ticker)


def exposure_key(value: Any) -> str:
    ticker = normalize_ticker(value)
    return EXPOSURE_ALIASES.get(ticker, ticker)


def _syntactically_us_ticker(ticker: str) -> bool:
    if not ticker or ticker in KNOWN_INVALID_TICKERS:
        return False
    if ticker.endswith(NON_US_SUFFIXES) or ticker.startswith("^"):
        return False
    if ticker.endswith("-USD") or ticker.endswith("=F"):
        return False
    return all(character.isalnum() or character == "-" for character in ticker)


def _confirmed_us_listing(ticker: str, asset: Dict[str, Any]) -> bool:
    if not _syntactically_us_ticker(ticker) or asset.get("price") is None:
        return False
    fields = asset.get("fundamentals", {})
    quote_type = str(fields.get("quoteType") or "").upper()
    exchange = str(fields.get("exchange") or "").upper().replace(" ", "")
    # Fail closed: if exchange metadata is unavailable, the symbol does not
    # enter either the actionable list or the watchlist.
    return quote_type in {"EQUITY", "ETF"} and exchange in US_EXCHANGES


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _difference(left: Any, right: Any) -> Optional[float]:
    a, b = _number(left), _number(right)
    return a - b if a is not None and b is not None else None


def _days_until(value: Any) -> Optional[int]:
    if not value:
        return None
    try:
        stamp = date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
    return (stamp - datetime.now(timezone.utc).date()).days


def _signal_summary(
    research: Iterable[Dict[str, Any]], held_exposures: set[str]
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for video in research:
        for signal in video.get("signals", []):
            raw_ticker = str(signal.get("ticker", "")).upper().strip()
            ticker = normalize_ticker(raw_ticker)
            if raw_ticker.endswith(NON_US_SUFFIXES) or not _syntactically_us_ticker(ticker):
                continue
            if not ticker or exposure_key(ticker) in held_exposures:
                continue
            item = grouped.setdefault(ticker, {
                "ticker": ticker,
                "bullish_signals": 0,
                "bearish_signals": 0,
                "neutral_signals": 0,
                "sources": [],
            })
            stance = int(signal.get("stance", 0) or 0)
            if stance > 0:
                item["bullish_signals"] += 1
            elif stance < 0:
                item["bearish_signals"] += 1
            else:
                item["neutral_signals"] += 1
            item["sources"].append({
                "channel": video.get("channel"),
                "published_date": video.get("published_date"),
                "title": video.get("title"),
                "url": video.get("url"),
                "stance": stance,
                "horizon_days": signal.get("horizon_days"),
                "confidence": signal.get("confidence"),
                "rationale": signal.get("rationale"),
                "original_ticker": raw_ticker,
            })
    for item in grouped.values():
        item["sources"] = sorted(
            item["sources"], key=lambda x: str(x.get("published_date", "")), reverse=True
        )[:8]
    return grouped


def _fundamental_summary(fields: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda",
        "revenueGrowth", "earningsGrowth", "profitMargins", "debtToEquity",
        "returnOnEquity", "nextEarningsDate", "sector", "industry",
        "exchange", "fullExchangeName", "quoteType", "longName",
    )
    return {key: fields.get(key) for key in keys}


def _base_checks(candidate: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    passed: List[str] = []
    failed: List[str] = []
    if candidate["bullish_signals"] > candidate["bearish_signals"]:
        passed.append("最近直播的看多訊號多於看空訊號")
    else:
        failed.append("最近直播沒有形成淨看多訊號")
    if candidate.get("price") is not None and candidate.get("return_60d") is not None:
        passed.append("價格與60日趨勢資料完整")
    else:
        failed.append("價格或60日趨勢資料不足")
    volatility = _number(candidate.get("annualized_volatility"))
    drawdown = _number(candidate.get("drawdown_from_52w_high"))
    if volatility is not None and volatility > 0.90:
        failed.append("年化波動率高於90%")
    if drawdown is not None and drawdown < -0.50:
        failed.append("距52週高點回撤超過50%")
    if candidate["ticker"] in LEVERAGED_OR_INVERSE:
        failed.append("槓桿或反向產品不列為新增持股")
    return passed, failed


def _momentum_checks(candidate: Dict[str, Any]) -> Tuple[int, int, List[str]]:
    tests = [
        ("20日報酬為正", candidate.get("return_20d")),
        ("60日報酬為正", candidate.get("return_60d")),
        ("20日跑贏SPY", candidate.get("relative_spy_20d")),
    ]
    if candidate.get("sector_proxy") != "SPY":
        tests.append(("60日跑贏產業ETF", candidate.get("relative_sector_60d")))
    available = [(label, _number(value)) for label, value in tests if _number(value) is not None]
    passed = [label for label, value in available if value is not None and value > 0]
    return len(passed), len(available), passed


def _fundamental_checks(candidate: Dict[str, Any]) -> Tuple[int, int, List[str], List[str]]:
    f = candidate.get("fundamentals", {})
    tests = [
        ("營收仍成長", _number(f.get("revenueGrowth")), lambda x: x > 0),
        ("獲利仍成長", _number(f.get("earningsGrowth")), lambda x: x > 0),
        ("利潤率為正", _number(f.get("profitMargins")), lambda x: x > 0),
        ("股東權益報酬率為正", _number(f.get("returnOnEquity")), lambda x: x > 0),
        ("負債權益比低於150", _number(f.get("debtToEquity")), lambda x: x < 150),
    ]
    available = [(label, value, check) for label, value, check in tests if value is not None]
    passed = [label for label, value, check in available if check(value)]
    red_flags: List[str] = []
    revenue, earnings = _number(f.get("revenueGrowth")), _number(f.get("earningsGrowth"))
    margin, debt = _number(f.get("profitMargins")), _number(f.get("debtToEquity"))
    if revenue is not None and revenue < -0.15:
        red_flags.append("營收年增率低於-15%")
    if earnings is not None and earnings < -0.20:
        red_flags.append("獲利年增率低於-20%")
    if margin is not None and margin <= 0:
        red_flags.append("目前利潤率不為正")
    if debt is not None and debt > 250:
        red_flags.append("負債權益比高於250")
    return len(passed), len(available), passed, red_flags


def _valuation_checks(candidate: Dict[str, Any]) -> Tuple[int, List[str]]:
    f = candidate.get("fundamentals", {})
    metrics = {
        "trailingPE": _number(f.get("trailingPE")),
        "forwardPE": _number(f.get("forwardPE")),
        "priceToBook": _number(f.get("priceToBook")),
        "enterpriseToEbitda": _number(f.get("enterpriseToEbitda")),
    }
    available = {key: value for key, value in metrics.items() if value is not None and value > 0}
    flags: List[str] = []
    growth = max(
        _number(f.get("revenueGrowth")) or -99,
        _number(f.get("earningsGrowth")) or -99,
    )
    pe = available.get("forwardPE") or available.get("trailingPE")
    if pe is not None and pe > 50 and growth < 0.20:
        flags.append("P/E高於50且營收／獲利成長未達20%")
    ev = available.get("enterpriseToEbitda")
    if ev is not None and ev > 35 and growth < 0.20:
        flags.append("EV/EBITDA高於35且成長未達20%")
    return len(available), flags


def _screen_account(candidate: Dict[str, Any], account_name: str) -> Dict[str, Any]:
    passed, failed = _base_checks(candidate)
    momentum, momentum_available, momentum_labels = _momentum_checks(candidate)
    fundamentals, fundamentals_available, fundamental_labels, fundamental_flags = (
        _fundamental_checks(candidate)
    )
    valuation_available, valuation_flags = _valuation_checks(candidate)
    days_to_earnings = _days_until(candidate.get("fundamentals", {}).get("nextEarningsDate"))
    is_fund = candidate["ticker"] in NON_EQUITIES

    passed.extend(momentum_labels)
    if not is_fund:
        passed.extend(fundamental_labels)
    failed.extend(fundamental_flags)
    failed.extend(valuation_flags)
    if days_to_earnings is not None and 0 <= days_to_earnings <= 7:
        failed.append("七日內接近財報，暫不以直播訊號建立新部位")

    if account_name == "IBKR":
        if momentum_available < 3 or momentum < 2:
            failed.append("長期候選需至少通過2項趨勢／相對強弱檢查")
        if not is_fund:
            if fundamentals_available < 4 or fundamentals < 3:
                failed.append("長期候選的基本面資料或正向項目不足")
            if valuation_available < 1:
                failed.append("長期候選缺少可用估值資料")
        max_weight = 0.05
    else:
        required_momentum = 3 if momentum_available >= 4 else 2
        if momentum_available < 3 or momentum < required_momentum:
            failed.append("戰術候選未通過足夠的20／60日趨勢與相對強弱檢查")
        if not is_fund:
            if candidate.get("sector_proxy") == "SPY":
                failed.append("缺少可靠產業ETF映射，無法檢查產業相對強弱")
            if fundamentals_available < 2:
                failed.append("公司基本面資料不足以排除明顯風險")
            if valuation_available < 1:
                failed.append("缺少估值資料，無法檢查是否已過度反映利多")
        max_weight = 0.03

    status = "eligible_for_portfolio_review" if not failed else "watchlist"
    return {
        "status": status,
        "max_initial_weight": max_weight if status == "eligible_for_portfolio_review" else 0.0,
        "passed_checks": list(dict.fromkeys(passed)),
        "failed_or_missing_checks": list(dict.fromkeys(failed)),
        "scorecard": {
            "momentum_and_relative_strength": f"{momentum}/{momentum_available}",
            "fundamentals": "ETF不適用" if is_fund else f"{fundamentals}/{fundamentals_available}",
            "valuation_metrics_available": "ETF不適用" if is_fund else valuation_available,
            "days_to_next_earnings": days_to_earnings,
        },
    }


def build_new_candidate_universe(
    portfolio: Dict[str, Any],
    research: List[Dict[str, Any]],
    evidence_provider: Callable[[str, Dict[str, Any]], Dict[str, Any]] = market_evidence,
    cache_path: Path = FUNDAMENTALS_CACHE_PATH,
) -> List[Dict[str, Any]]:
    held_exposures = {
        exposure_key(ticker)
        for account in portfolio.get("accounts", [])
        for ticker in account.get("positions", {})
    }
    grouped = _signal_summary(research, held_exposures)
    if not grouped:
        return []

    cache = _load_fundamentals_cache()
    evidence_by_ticker: Dict[str, Dict[str, Any]] = {}

    def evidence(ticker: str) -> Dict[str, Any]:
        if ticker not in evidence_by_ticker:
            evidence_by_ticker[ticker] = evidence_provider(ticker, cache)
        return evidence_by_ticker[ticker]

    spy = evidence("SPY")
    output: List[Dict[str, Any]] = []
    for ticker, signal_data in sorted(grouped.items()):
        asset = evidence(ticker)
        if not _confirmed_us_listing(ticker, asset):
            logging.info("Skipping non-US, invalid, or unavailable candidate %s", ticker)
            continue
        proxy = sector_proxy(ticker)
        if proxy == "SPY" and ticker not in NON_EQUITIES:
            proxy = SECTOR_NAME_PROXY.get(
                str(asset.get("fundamentals", {}).get("sector", "")), "SPY"
            )
        industry = evidence(proxy)
        fundamentals = _fundamental_summary(asset.get("fundamentals", {}))
        candidate: Dict[str, Any] = {
            **signal_data,
            "price": asset.get("price"),
            "return_20d": asset.get("return_20d"),
            "return_60d": asset.get("return_60d"),
            "return_120d": asset.get("return_120d"),
            "relative_spy_20d": _difference(asset.get("return_20d"), spy.get("return_20d")),
            "relative_spy_60d": _difference(asset.get("return_60d"), spy.get("return_60d")),
            "relative_spy_120d": _difference(asset.get("return_120d"), spy.get("return_120d")),
            "sector_proxy": proxy,
            "relative_sector_20d": _difference(asset.get("return_20d"), industry.get("return_20d")),
            "relative_sector_60d": _difference(asset.get("return_60d"), industry.get("return_60d")),
            "relative_sector_120d": _difference(asset.get("return_120d"), industry.get("return_120d")),
            "drawdown_from_52w_high": asset.get("drawdown_from_52w_high"),
            "annualized_volatility": asset.get("annualized_volatility"),
            "fundamentals": fundamentals,
            "fundamentals_as_of": asset.get("fundamentals_as_of"),
            "exposure_key": exposure_key(ticker),
        }
        candidate["account_screening"] = {
            account.get("name"): _screen_account(candidate, str(account.get("name")))
            for account in portfolio.get("accounts", [])
        }
        output.append(candidate)

    output.sort(key=lambda item: (
        not any(
            screen.get("status") == "eligible_for_portfolio_review"
            for screen in item.get("account_screening", {}).values()
        ),
        -(item.get("bullish_signals", 0) - item.get("bearish_signals", 0)),
        item.get("ticker", ""),
    ))

    try:
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logging.warning("Could not save candidate fundamentals cache: %s", exc)
    return output


def eligible_new_tickers(
    candidates: List[Dict[str, Any]], account_name: str
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for candidate in candidates:
        screen = candidate.get("account_screening", {}).get(account_name, {})
        if screen.get("status") == "eligible_for_portfolio_review":
            result[candidate["ticker"]] = float(screen.get("max_initial_weight", 0.0))
    return result
