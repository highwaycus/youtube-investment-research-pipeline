from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional

from candidate_screening import SECTOR_NAME_PROXY, exposure_key, normalize_ticker
from market_features import NON_EQUITIES, sector_proxy
from research import _load_fundamentals_cache, market_evidence


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _difference(left: Any, right: Any) -> Optional[float]:
    a, b = _number(left), _number(right)
    return a - b if a is not None and b is not None else None


def _stream_signals(research: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for video in research:
        for signal in video.get("signals", []):
            ticker = normalize_ticker(signal.get("ticker"))
            key = exposure_key(ticker)
            if not ticker:
                continue
            item = grouped.setdefault(key, {
                "bullish": 0, "bearish": 0, "neutral": 0, "sources": [],
            })
            stance = int(signal.get("stance", 0) or 0)
            if stance > 0:
                item["bullish"] += 1
            elif stance < 0:
                item["bearish"] += 1
            else:
                item["neutral"] += 1
            item["sources"].append({
                "channel": video.get("channel"),
                "date": video.get("published_date"),
                "title": video.get("title"),
                "stance": stance,
                "rationale": signal.get("rationale"),
            })
    return grouped


def _fundamental_red_flags(fields: Dict[str, Any]) -> List[str]:
    checks = (
        ("營收年增率低於 -15%", fields.get("revenueGrowth"), lambda x: x < -0.15),
        ("獲利年增率低於 -20%", fields.get("earningsGrowth"), lambda x: x < -0.20),
        ("目前利潤率不為正", fields.get("profitMargins"), lambda x: x <= 0),
        ("負債權益比高於 250", fields.get("debtToEquity"), lambda x: x > 250),
    )
    return [label for label, value, test in checks if _number(value) is not None and test(float(value))]


def _market_red_flags(
    evidence: Dict[str, Any], spy: Dict[str, Any], sector: Dict[str, Any],
    account_name: str, sector_is_market: bool = False,
) -> tuple[List[str], Dict[str, Optional[float]]]:
    metrics = {
        "return_20d": _number(evidence.get("return_20d")),
        "return_60d": _number(evidence.get("return_60d")),
        "return_120d": _number(evidence.get("return_120d")),
        "relative_spy_20d": _difference(evidence.get("return_20d"), spy.get("return_20d")),
        "relative_spy_120d": _difference(evidence.get("return_120d"), spy.get("return_120d")),
        "relative_sector_60d": _difference(evidence.get("return_60d"), sector.get("return_60d")),
        "relative_sector_120d": _difference(evidence.get("return_120d"), sector.get("return_120d")),
        "drawdown": _number(evidence.get("drawdown_from_52w_high")),
    }
    flags: List[str] = []
    if account_name == "IBKR":
        tests = [
            ("120 日報酬低於 -10%", metrics["return_120d"], lambda x: x < -0.10),
            ("120 日落後 SPY 超過 10%", metrics["relative_spy_120d"], lambda x: x < -0.10),
            ("距 52 週高點回撤超過 30%", metrics["drawdown"], lambda x: x < -0.30),
        ]
        if not sector_is_market:
            tests.append(("120 日落後產業 ETF 超過 10%", metrics["relative_sector_120d"], lambda x: x < -0.10))
    else:
        tests = [
            ("20 日趨勢為負", metrics["return_20d"], lambda x: x < 0),
            ("60 日趨勢為負", metrics["return_60d"], lambda x: x < 0),
            ("20 日落後 SPY 超過 5%", metrics["relative_spy_20d"], lambda x: x < -0.05),
            ("距 52 週高點回撤超過 25%", metrics["drawdown"], lambda x: x < -0.25),
        ]
        if not sector_is_market:
            tests.append(("60 日落後產業 ETF 超過 8%", metrics["relative_sector_60d"], lambda x: x < -0.08))
    for label, value, test in tests:
        if value is not None and test(value):
            flags.append(label)
    return flags, metrics


def build_holding_reviews(
    portfolio: Dict[str, Any], research: List[Dict[str, Any]],
    evidence_provider: Callable[[str, Dict[str, Any]], Dict[str, Any]] = market_evidence,
) -> Dict[str, List[Dict[str, Any]]]:
    signals = _stream_signals(research)
    cache = _load_fundamentals_cache()
    fetched: Dict[str, Dict[str, Any]] = {}

    def evidence(ticker: str) -> Dict[str, Any]:
        if ticker not in fetched:
            fetched[ticker] = evidence_provider(ticker, cache)
        return fetched[ticker]

    spy = evidence("SPY")
    reviews: Dict[str, List[Dict[str, Any]]] = {}
    for account in portfolio.get("accounts", []):
        name = str(account.get("name"))
        max_weight = float(account.get("max_single_weight", 0.15))
        account_reviews: List[Dict[str, Any]] = []
        for position in account.get("positions_snapshot", []):
            ticker = normalize_ticker(position.get("ticker"))
            current_weight = float(position.get("current_weight") or 0.0)
            asset = position.get("market_evidence") or {}
            proxy = sector_proxy(ticker)
            if proxy == "SPY" and ticker not in NON_EQUITIES:
                proxy = SECTOR_NAME_PROXY.get(
                    str(asset.get("fundamentals", {}).get("sector", "")), "SPY"
                )
            industry = spy if proxy == "SPY" else evidence(proxy)
            market_flags, metrics = _market_red_flags(
                asset, spy, industry, name, sector_is_market=(proxy == "SPY")
            )
            fundamental_flags = (
                [] if ticker in NON_EQUITIES
                else _fundamental_red_flags(asset.get("fundamentals", {}))
            )
            stream = signals.get(exposure_key(ticker), {
                "bullish": 0, "bearish": 0, "neutral": 0, "sources": [],
            })
            net_bearish = stream["bearish"] > stream["bullish"]
            over_limit = ticker not in NON_EQUITIES and current_weight > max_weight + 0.001

            status = "hold"
            trigger = "今日沒有足夠的獨立證據支持調整"
            if name == "IBKR":
                if net_bearish and len(fundamental_flags) >= 2 and len(market_flags) >= 2:
                    status = "consider_exit"
                    trigger = "直播看空，且長期趨勢與多項基本面同時惡化"
                elif net_bearish and len(market_flags) + len(fundamental_flags) >= 2:
                    status = "consider_reduce"
                    trigger = "直播看空，且得到長期市場或基本面證據確認"
            else:
                if net_bearish and len(market_flags) >= 4:
                    status = "consider_exit"
                    trigger = "直播看空，且短期價格／相對強弱廣泛轉弱"
                elif net_bearish and len(market_flags) >= 2:
                    status = "consider_reduce"
                    trigger = "直播看空，且得到至少兩項市場證據確認"

            # A mechanical 25% trim is not useful for a position already below
            # 1% of the account. Keep it unless the stronger exit rule above
            # was met; this also avoids misleading 0.0% -> 0.0% report rows.
            if status == "consider_reduce" and current_weight < 0.01:
                status = "hold"
                trigger = "部位低於 1%，不建議進行微量減碼"

            if status == "hold" and over_limit:
                status = "reduce_to_limit"
                trigger = "單檔權重超過帳戶 15% 上限，屬集中度管理"

            # When a moderate bearish review and the concentration cap both
            # apply, the only defensible numeric target is the explicit cap.
            # Do not manufacture a second target from an arbitrary 25% trim.
            if status == "consider_reduce" and over_limit:
                trigger = "直播淨看空且相對強弱轉弱；先降至帳戶 15% 上限"

            if status == "consider_exit":
                target_weight = 0.0
            elif status == "consider_reduce":
                target_weight = max_weight if over_limit else current_weight * 0.75
            elif status == "reduce_to_limit":
                target_weight = max_weight
            else:
                target_weight = current_weight

            account_reviews.append({
                "ticker": ticker,
                "status": status,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "trigger": trigger,
                "stream_signals": stream,
                "market_red_flags": market_flags,
                "fundamental_red_flags": fundamental_flags,
                "metrics": metrics,
                "sector_proxy": proxy,
                "fundamentals": asset.get("fundamentals", {}),
                "fundamentals_as_of": asset.get("fundamentals_as_of"),
            })
        reviews[name] = account_reviews
    return reviews


def actionable_holding_reviews(
    reviews: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for account, items in reviews.items():
        for item in items:
            if item.get("status") != "hold":
                output.append({"account": account, **item})
    priority = {"consider_exit": 0, "consider_reduce": 1, "reduce_to_limit": 2}
    return sorted(output, key=lambda item: (
        priority.get(str(item.get("status")), 9),
        -float(item.get("current_weight") or 0.0),
    ))
