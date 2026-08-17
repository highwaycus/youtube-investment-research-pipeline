from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yfinance as yf
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "research.db"
PLAYBOOK_PATH = ROOT / "channel_playbook.json"
FUNDAMENTALS_CACHE_PATH = ROOT / "fundamentals_cache.json"
HORIZONS = (5, 20, 60)
RETRYABLE_TRANSCRIPT_STATUSES = ("unavailable", "blocked", "temporary_error")


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS videos (
        video_id TEXT PRIMARY KEY,
        channel_id TEXT NOT NULL,
        channel_name TEXT NOT NULL,
        content_type TEXT NOT NULL DEFAULT 'stream',
        title TEXT NOT NULL,
        url TEXT NOT NULL,
        published_date TEXT NOT NULL,
        transcript_status TEXT NOT NULL,
        summary TEXT NOT NULL,
        analyzed_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL REFERENCES videos(video_id),
        ticker TEXT NOT NULL,
        stance INTEGER NOT NULL,
        confidence REAL NOT NULL,
        horizon_days INTEGER NOT NULL,
        rationale TEXT NOT NULL,
        UNIQUE(video_id, ticker, horizon_days)
    );
    CREATE TABLE IF NOT EXISTS performance (
        signal_id INTEGER PRIMARY KEY REFERENCES signals(id),
        entry_date TEXT NOT NULL,
        exit_date TEXT NOT NULL,
        asset_return REAL NOT NULL,
        benchmark_return REAL NOT NULL,
        excess_return REAL NOT NULL,
        correct INTEGER NOT NULL,
        evaluated_at TEXT NOT NULL
    );
    """)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(videos)")}
    if "content_type" not in columns:
        conn.execute(
            "ALTER TABLE videos ADD COLUMN content_type TEXT NOT NULL DEFAULT 'legacy'"
        )
    if "transcript_error" not in columns:
        conn.execute(
            "ALTER TABLE videos ADD COLUMN transcript_error TEXT NOT NULL DEFAULT ''"
        )
    if "transcript_attempts" not in columns:
        conn.execute(
            "ALTER TABLE videos ADD COLUMN transcript_attempts INTEGER NOT NULL DEFAULT 0"
        )
    # The selected research universe is streams only. Keep legacy video rows
    # for auditability, but remove their signals so they cannot influence the
    # stream-only reliability statistics or allocation report.
    conn.execute("""
        DELETE FROM performance
        WHERE signal_id IN (
            SELECT s.id FROM signals s
            JOIN videos v ON v.video_id = s.video_id
            WHERE v.content_type != 'stream'
        )
    """)
    conn.execute("""
        DELETE FROM signals
        WHERE video_id IN (
            SELECT video_id FROM videos WHERE content_type != 'stream'
        )
    """)
    # Never let title/description-only inference affect research results.
    # This also removes legacy signals created before transcripts became
    # mandatory.
    conn.execute("""
        DELETE FROM performance
        WHERE signal_id IN (
            SELECT s.id FROM signals s
            JOIN videos v ON v.video_id = s.video_id
            WHERE v.transcript_status != 'available'
        )
    """)
    conn.execute("""
        DELETE FROM signals
        WHERE video_id IN (
            SELECT video_id FROM videos WHERE transcript_status != 'available'
        )
    """)
    # SPY is the benchmark, so new SPY-vs-SPY performance rows are not created
    # below. Legacy rows remain for auditability but are excluded from every
    # reliability calculation.
    conn.execute("""
        UPDATE videos
        SET summary = CASE transcript_status
            WHEN 'blocked' THEN '未分析：字幕請求受到 YouTube 限制，稍後重試。'
            WHEN 'temporary_error' THEN '未分析：字幕暫時取得失敗，稍後重試。'
            WHEN 'no_transcript' THEN '未分析：影片未提供可讀取字幕。'
            WHEN 'restricted' THEN '未分析：影片受限，無法取得字幕。'
            ELSE '未分析：字幕尚未成功取得，稍後重試。'
        END
        WHERE transcript_status != 'available'
    """)
    conn.commit()
    return conn


def clean_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _transcript_failure_status(exc: Exception) -> str:
    """Separate permanent transcript absence from retryable request failures."""
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    blocked_markers = (
        "requestblocked", "ipblocked", "too many requests", "429",
        "blocking requests from your ip", "temporarily blocked",
    )
    permanent_markers = (
        "transcriptsdisabled", "notranscriptfound",
        "no transcript", "subtitles are disabled",
    )
    restricted_markers = (
        "age restricted", "members-only", "private video",
        "video unavailable",
    )
    combined = f"{name} {message}"
    if any(marker in combined for marker in blocked_markers):
        return "blocked"
    if any(marker in combined for marker in restricted_markers):
        return "restricted"
    if any(marker in combined for marker in permanent_markers):
        return "no_transcript"
    return "temporary_error"


def fetch_transcript(video_id: str) -> Tuple[str, str, str]:
    try:
        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["zh-TW", "zh-Hant", "zh-Hans", "zh-CN", "zh", "en"],
        )
        text = " ".join(item.text for item in transcript)
        return text[:120_000], "available", ""
    except Exception as exc:
        status = _transcript_failure_status(exc)
        error = f"{type(exc).__name__}: {exc}"[:2000]
        logging.warning(
            "Transcript fetch failed for %s [%s]: %s", video_id, status, error
        )
        return "", status, error


def extract_video_research(
    channel_id: str,
    channel_name: str,
    video_id: str,
    title: str,
    url: str,
    published_date: str,
    description: str = "",
    content_type: str = "stream",
) -> Dict[str, Any]:
    transcript, transcript_status, transcript_error = fetch_transcript(video_id)
    if not transcript:
        logging.info(
            "Skipping OpenAI analysis for %s because no transcript is available",
            video_id,
        )
        summaries = {
            "blocked": "未分析：字幕請求受到 YouTube 限制，稍後重試。",
            "temporary_error": "未分析：字幕暫時取得失敗，稍後重試。",
            "restricted": "未分析：影片受限，無法取得字幕。",
            "no_transcript": "未分析：影片未提供可讀取字幕。",
        }
        return {
            "video_id": video_id,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "content_type": content_type,
            "title": title,
            "url": url,
            "published_date": published_date,
            "transcript_status": transcript_status,
            "transcript_error": transcript_error,
            "summary": summaries.get(
                transcript_status, "未分析：字幕尚未成功取得，稍後重試。"
            ),
            "signals": [],
        }
    material = transcript
    prompt = f"""Analyze this investment video using only the supplied material.
Return valid JSON, without markdown, in this shape:
{{
  "summary": "concise Traditional Chinese summary",
  "signals": [
    {{
      "ticker": "Yahoo Finance-compatible ticker such as NVDA, SPY, TSM, 2330.TW",
      "stance": -1,
      "confidence": 0.0,
      "horizon_days": 20,
      "rationale": "Traditional Chinese explanation"
    }}
  ]
}}
stance must be -1 bearish, 0 neutral, or 1 bullish. horizon_days must be one of 5, 20, 60.
Include a signal only when the speaker makes a meaningful directional claim. Convert macro
claims to a liquid proxy only when the mapping is clear (SPY, QQQ, SOXX, IWM, TLT, UUP,
GLD, BTC-USD). Do not invent tickers or claims. Output at most 12 signals.

Channel: {channel_name}
Published: {published_date}
Title: {title}
Transcript status: {transcript_status}
Material:
{material}"""
    response = OpenAI().responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"), input=prompt
    )
    parsed = clean_json(response.output_text)
    summary = str(parsed.get("summary", "")).strip()
    signals = []
    for item in parsed.get("signals", [])[:12]:
        try:
            ticker = str(item["ticker"]).upper().strip()
            stance = int(item["stance"])
            confidence = max(0.0, min(1.0, float(item["confidence"])))
            horizon = int(item["horizon_days"])
            if not ticker or stance not in (-1, 0, 1) or horizon not in HORIZONS:
                continue
            signals.append({
                "ticker": ticker,
                "stance": stance,
                "confidence": confidence,
                "horizon_days": horizon,
                "rationale": str(item.get("rationale", ""))[:1000],
            })
        except (KeyError, TypeError, ValueError):
            continue
    return {
        "video_id": video_id,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "content_type": content_type,
        "title": title,
        "url": url,
        "published_date": published_date,
        "transcript_status": transcript_status,
        "transcript_error": "",
        "summary": summary,
        "signals": signals,
    }


def save_video_research(conn: sqlite3.Connection, data: Dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO videos
        (video_id, channel_id, channel_name, content_type, title, url,
         published_date, transcript_status, summary, analyzed_at,
         transcript_error, transcript_attempts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(video_id) DO UPDATE SET
          channel_id = excluded.channel_id,
          channel_name = excluded.channel_name,
          content_type = excluded.content_type,
          title = excluded.title,
          url = excluded.url,
          published_date = excluded.published_date,
          transcript_status = excluded.transcript_status,
          summary = excluded.summary,
          analyzed_at = excluded.analyzed_at,
          transcript_error = excluded.transcript_error,
          transcript_attempts = videos.transcript_attempts + 1""",
        (
            data["video_id"], data["channel_id"], data["channel_name"],
            data.get("content_type", "stream"), data["title"], data["url"], data["published_date"],
            data["transcript_status"], data["summary"],
            datetime.now(timezone.utc).isoformat(),
            data.get("transcript_error", ""),
        ),
    )
    conn.execute("DELETE FROM signals WHERE video_id = ?", (data["video_id"],))
    for signal in data["signals"]:
        conn.execute(
            """INSERT OR IGNORE INTO signals
            (video_id, ticker, stance, confidence, horizon_days, rationale)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                data["video_id"], signal["ticker"], signal["stance"],
                signal["confidence"], signal["horizon_days"], signal["rationale"],
            ),
        )
    conn.commit()


def _forward_return(ticker: str, published: date, horizon: int) -> Optional[Tuple[str, str, float]]:
    start = published + timedelta(days=1)
    end = published + timedelta(days=max(30, horizon * 3))
    try:
        hist = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True)
        if hist.empty or len(hist) <= horizon:
            return None
        entry = hist.iloc[0]
        exit_row = hist.iloc[horizon]
        entry_price = float(entry["Close"])
        exit_price = float(exit_row["Close"])
        if entry_price <= 0:
            return None
        return (
            hist.index[0].date().isoformat(),
            hist.index[horizon].date().isoformat(),
            exit_price / entry_price - 1.0,
        )
    except Exception as exc:
        logging.warning("Price history failed for %s: %s", ticker, exc)
        return None


def _performance_ticker(value: str) -> Optional[str]:
    """Return a Yahoo-compatible US ticker, or None for unsupported history rows."""
    ticker = str(value or "").upper().strip().replace("$", "")
    aliases = {"BRK.A": "BRK-A", "BRK.B": "BRK-B", "BF.A": "BF-A", "BF.B": "BF-B"}
    ticker = aliases.get(ticker, ticker)
    non_us_suffixes = (
        ".TW", ".TWO", ".HK", ".KS", ".KQ", ".T", ".L", ".TO", ".V",
        ".AX", ".SI", ".SS", ".SZ", ".DE", ".PA", ".MI", ".AS",
    )
    if ticker in {"VRTV", "CWAV"} or ticker.endswith(non_us_suffixes):
        return None
    return ticker


def update_performance(conn: sqlite3.Connection) -> int:
    rows = conn.execute("""
        SELECT s.id, s.ticker, s.stance, s.horizon_days, v.published_date
        FROM signals s JOIN videos v ON v.video_id = s.video_id
        LEFT JOIN performance p ON p.signal_id = s.id
        WHERE p.signal_id IS NULL
          AND s.stance != 0
          AND UPPER(s.ticker) != 'SPY'
          AND v.content_type = 'stream'
    """).fetchall()
    count = 0
    today = datetime.now(timezone.utc).date()
    for row in rows:
        published = date.fromisoformat(row["published_date"][:10])
        # Require enough calendar time to avoid evaluating incomplete horizons.
        if (today - published).days < int(row["horizon_days"] * 1.6 + 7):
            continue
        ticker = _performance_ticker(row["ticker"])
        if ticker is None:
            continue
        asset = _forward_return(ticker, published, row["horizon_days"])
        benchmark = _forward_return("SPY", published, row["horizon_days"])
        if not asset or not benchmark:
            continue
        excess = asset[2] - benchmark[2]
        correct = int(int(row["stance"]) * excess > 0)
        conn.execute(
            """INSERT OR REPLACE INTO performance
            (signal_id, entry_date, exit_date, asset_return, benchmark_return,
             excess_return, correct, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["id"], asset[0], asset[1], asset[2], benchmark[2], excess,
                correct, datetime.now(timezone.utc).isoformat(),
            ),
        )
        count += 1
    conn.commit()
    return count


def reliability_summary(
    conn: sqlite3.Connection, days: int = 183
) -> List[Dict[str, Any]]:
    cutoff = _research_cutoff(conn, days)
    rows = conn.execute("""
        SELECT v.channel_name, s.horizon_days, COUNT(*) AS samples,
               SUM(p.correct) AS wins, AVG(s.stance * p.excess_return) AS signed_excess
        FROM performance p
        JOIN signals s ON s.id = p.signal_id
        JOIN videos v ON v.video_id = s.video_id
        WHERE v.content_type = 'stream'
          AND v.published_date >= ?
          AND UPPER(s.ticker) != 'SPY'
        GROUP BY v.channel_name, s.horizon_days
        ORDER BY v.channel_name, s.horizon_days
    """, (cutoff,)).fetchall()
    result = []
    for row in rows:
        samples = int(row["samples"])
        wins = int(row["wins"])
        # Beta(2,2) shrinkage prevents a tiny sample from looking certain.
        result.append({
            "channel": row["channel_name"],
            "horizon_days": int(row["horizon_days"]),
            "samples": samples,
            "raw_hit_rate": wins / samples if samples else 0.0,
            "shrunk_hit_rate": (wins + 2) / (samples + 4),
            "average_signed_excess_return": float(row["signed_excess"] or 0.0),
        })
    return result


def _backtest_group(
    rows: Sequence[sqlite3.Row],
    fields: Sequence[str],
    minimum_samples: int = 1,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[sqlite3.Row]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        grouped.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        samples = len(items)
        if samples < minimum_samples:
            continue
        wins = sum(int(item["correct"]) for item in items)
        signed = [float(item["signed_excess"]) for item in items]
        record: Dict[str, Any] = {
            field: value for field, value in zip(fields, key)
        }
        record.update({
            "samples": samples,
            "wins": wins,
            "raw_hit_rate": wins / samples,
            # Beta(2,2) shrinkage limits the influence of small groups.
            "shrunk_hit_rate": (wins + 2) / (samples + 4),
            "average_signed_excess_return": sum(signed) / samples,
            "median_signed_excess_return": statistics.median(signed),
        })
        output.append(record)
    return sorted(output, key=lambda item: tuple(item.get(field) for field in fields))


def _research_cutoff(conn: sqlite3.Connection, days: int) -> str:
    """Use a rolling window without dropping a nearly-six-month backfill edge."""
    bounds = conn.execute("""
        SELECT MIN(published_date), MAX(published_date)
        FROM videos
        WHERE content_type = 'stream' AND transcript_status = 'available'
    """).fetchone()
    if not bounds or not bounds[1]:
        return (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    latest = date.fromisoformat(str(bounds[1])[:10])
    cutoff = latest - timedelta(days=days)
    if bounds[0]:
        earliest = date.fromisoformat(str(bounds[0])[:10])
        # The original backfill can span a few extra calendar days depending
        # on when it finished. Preserve that edge until the rolling window has
        # moved materially beyond it.
        if timedelta(0) <= cutoff - earliest <= timedelta(days=7):
            cutoff = earliest
    return cutoff.isoformat()


def backtest_metrics(conn: sqlite3.Connection, days: int = 183) -> Dict[str, Any]:
    """Build deterministic metrics; never let the language model invent them."""
    cutoff = _research_cutoff(conn, days)
    rows = conn.execute("""
        SELECT v.channel_name AS channel,
               UPPER(s.ticker) AS ticker,
               CASE s.stance WHEN 1 THEN 'bullish' ELSE 'bearish' END AS direction,
               s.horizon_days,
               p.correct,
               s.stance * p.excess_return AS signed_excess
        FROM performance p
        JOIN signals s ON s.id = p.signal_id
        JOIN videos v ON v.video_id = s.video_id
        WHERE v.content_type = 'stream'
          AND v.transcript_status = 'available'
          AND v.published_date >= ?
          AND s.stance != 0
          AND UPPER(s.ticker) != 'SPY'
    """, (cutoff,)).fetchall()
    coverage = conn.execute("""
        SELECT
          COUNT(DISTINCT CASE WHEN v.transcript_status = 'available'
                              THEN v.video_id END) AS available_streams,
          COUNT(DISTINCT CASE WHEN v.transcript_status = 'available'
                               AND s.id IS NOT NULL THEN v.video_id END) AS streams_with_signals,
          COUNT(s.id) AS total_signals,
          COUNT(CASE WHEN s.stance != 0 THEN 1 END) AS directional_signals,
          MIN(CASE WHEN v.transcript_status = 'available'
                   THEN v.published_date END) AS first_date,
          MAX(CASE WHEN v.transcript_status = 'available'
                   THEN v.published_date END) AS last_date
        FROM videos v
        LEFT JOIN signals s ON s.video_id = v.video_id
        WHERE v.content_type = 'stream' AND v.published_date >= ?
    """, (cutoff,)).fetchone()
    spy_signal_count = conn.execute("""
        SELECT COUNT(*)
        FROM signals s
        JOIN videos v ON v.video_id = s.video_id
        WHERE v.content_type = 'stream'
          AND v.transcript_status = 'available'
          AND v.published_date >= ?
          AND s.stance != 0
          AND UPPER(s.ticker) = 'SPY'
    """, (cutoff,)).fetchone()[0]
    matured_spy = conn.execute("""
        SELECT COUNT(*)
        FROM performance p
        JOIN signals s ON s.id = p.signal_id
        JOIN videos v ON v.video_id = s.video_id
        WHERE v.content_type = 'stream'
          AND v.transcript_status = 'available'
          AND v.published_date >= ?
          AND s.stance != 0
          AND UPPER(s.ticker) = 'SPY'
    """, (cutoff,)).fetchone()[0]
    valid_ids = conn.execute("""
        SELECT COUNT(*)
        FROM performance p
        JOIN signals s ON s.id = p.signal_id
        JOIN videos v ON v.video_id = s.video_id
        WHERE v.content_type = 'stream'
          AND v.transcript_status = 'available'
          AND v.published_date >= ?
          AND s.stance != 0
          AND UPPER(s.ticker) != 'SPY'
    """, (cutoff,)).fetchone()[0]
    directional = int(coverage["directional_signals"] or 0)
    spy_signal_count = int(spy_signal_count or 0)
    matured_spy = int(matured_spy or 0)
    metrics = {
        "methodology": {
            "benchmark": "SPY",
            "correct_definition": (
                "bullish when asset outperforms SPY; bearish when asset underperforms SPY"
            ),
            "self_comparison_rule": "SPY signals are excluded from SPY-relative scoring",
            "shrinkage": "Beta(2,2)",
            "ticker_minimum_samples": 5,
            "warning": (
                "Signals from the same stream or repeated market theme are correlated; "
                "hit rates are historical tendencies, not calibrated probabilities."
            ),
        },
        "coverage": {
            "days": days,
            "first_date": coverage["first_date"],
            "last_date": coverage["last_date"],
            "available_streams": int(coverage["available_streams"] or 0),
            "streams_with_signals": int(coverage["streams_with_signals"] or 0),
            "total_extracted_signals": int(coverage["total_signals"] or 0),
            "directional_signals": directional,
            "mature_signals_before_benchmark_exclusion": (
                int(valid_ids or 0) + matured_spy
            ),
            "mature_valid_signals": int(valid_ids or 0),
            "excluded_mature_spy_self_comparisons": matured_spy,
            "all_spy_signals": spy_signal_count,
            "unevaluated_non_spy_signals": max(
                0, directional - spy_signal_count - int(valid_ids or 0)
            ),
        },
        "overall": _backtest_group(rows, [])[0] if rows else {
            "samples": 0, "wins": 0, "raw_hit_rate": 0.0,
            "shrunk_hit_rate": 0.5, "average_signed_excess_return": 0.0,
            "median_signed_excess_return": 0.0,
        },
        "by_channel": _backtest_group(rows, ["channel"]),
        "by_horizon": _backtest_group(rows, ["horizon_days"]),
        "by_channel_horizon": _backtest_group(
            rows, ["channel", "horizon_days"]
        ),
        "by_direction": _backtest_group(rows, ["direction"]),
        "by_channel_direction": _backtest_group(
            rows, ["channel", "direction"]
        ),
        "by_ticker": _backtest_group(rows, ["ticker"], minimum_samples=5),
        "by_channel_ticker": _backtest_group(
            rows, ["channel", "ticker"], minimum_samples=5
        ),
    }
    return metrics


def _metric_lookup(
    metrics: Dict[str, Any], group: str, **match: Any
) -> Optional[Dict[str, Any]]:
    values = metrics.get(group, [])
    if isinstance(values, dict):
        return values
    for item in values:
        if all(item.get(key) == value for key, value in match.items()):
            return item
    return None


def annotate_research_with_backtest(
    research: List[Dict[str, Any]], playbook: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Attach a deterministic historical support score to each current signal."""
    metrics = playbook.get("backtest_metrics") or {}
    if not metrics:
        return research
    output = json.loads(json.dumps(research, ensure_ascii=False))
    component_specs = [
        ("overall", 0.05, {}),
        ("by_channel", 0.10, {"channel": None}),
        ("by_horizon", 0.15, {"horizon_days": None}),
        ("by_channel_horizon", 0.30, {"channel": None, "horizon_days": None}),
        ("by_direction", 0.10, {"direction": None}),
        ("by_channel_direction", 0.10, {"channel": None, "direction": None}),
        ("by_ticker", 0.10, {"ticker": None}),
        ("by_channel_ticker", 0.10, {"channel": None, "ticker": None}),
    ]
    for video in output:
        channel = video.get("channel")
        for signal in video.get("signals", []):
            ticker = str(signal.get("ticker", "")).upper()
            if ticker == "SPY" or int(signal.get("stance", 0)) == 0:
                signal["historical_backtest"] = {
                    "status": "not_scored",
                    "reason": "SPY benchmark self-comparison or neutral stance",
                }
                continue
            values = {
                "channel": channel,
                "horizon_days": int(signal.get("horizon_days", 20)),
                "direction": "bullish" if int(signal.get("stance", 0)) > 0 else "bearish",
                "ticker": ticker,
            }
            weighted = 0.0
            total_weight = 0.0
            used = []
            for group, weight, template in component_specs:
                match = {
                    key: values[key] for key in template
                }
                item = _metric_lookup(metrics, group, **match)
                if not item:
                    continue
                samples = int(item.get("samples", 0))
                rate = float(item.get("shrunk_hit_rate", 0.5))
                weighted += weight * rate
                total_weight += weight
                used.append({
                    "group": group,
                    "samples": samples,
                    "shrunk_hit_rate": rate,
                })
            score = weighted / total_weight if total_weight else 0.5
            if score >= 0.60:
                label, evidence_weight = "strong_support", 1.0
            elif score >= 0.54:
                label, evidence_weight = "modest_support", 0.8
            elif score <= 0.40:
                label, evidence_weight = "strongly_weak", 0.25
            elif score <= 0.46:
                label, evidence_weight = "weak", 0.4
            else:
                label, evidence_weight = "neutral", 0.6
            signal["historical_backtest"] = {
                "status": label,
                "historical_support_score": score,
                "stream_evidence_weight": evidence_weight,
                "interpretation": (
                    "This is a historical support tendency, not a probability. "
                    "A weak score reduces reliance and does not reverse the signal."
                ),
                "components": used,
            }
    return output


def build_channel_playbook(
    conn: sqlite3.Connection, path: Path = PLAYBOOK_PATH, days: int = 183
) -> Dict[str, Any]:
    cutoff = _research_cutoff(conn, days)
    rows = conn.execute("""
        SELECT v.video_id, v.channel_name, v.title, v.published_date, v.summary
        FROM videos v
        WHERE v.content_type = 'stream'
          AND v.transcript_status = 'available'
          AND v.published_date >= ?
        ORDER BY v.published_date
    """, (cutoff,)).fetchall()
    evidence = []
    for row in rows:
        signals = conn.execute("""
            SELECT ticker, stance, confidence, horizon_days, rationale
            FROM signals WHERE video_id = ?
        """, (row["video_id"],)).fetchall()
        evidence.append({
            "channel": row["channel_name"],
            "date": row["published_date"],
            "title": row["title"][:300],
            "summary": row["summary"][:800],
            "signals": [
                {
                    "ticker": item["ticker"], "stance": item["stance"],
                    "confidence": item["confidence"],
                    "horizon_days": item["horizon_days"],
                    "rationale": item["rationale"][:250],
                }
                for item in signals
            ],
        })
    if not evidence:
        metrics = backtest_metrics(conn, days=days)
        playbook = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "coverage": metrics["coverage"],
            "backtest_metrics": metrics,
            "recurring_themes": [], "decision_frameworks": [],
            "known_biases_and_limits": ["沒有足夠的有字幕直播資料。"],
        }
        path.write_text(json.dumps(playbook, ensure_ascii=False, indent=2), encoding="utf-8")
        return playbook

    metrics = backtest_metrics(conn, days=days)
    prompt = f"""Distill the supplied six-month livestream research into a reusable evidence
playbook. Return valid JSON only, in Traditional Chinese. This is retrieval and synthesis,
not model training and not a prediction. Do not invent facts outside the supplied records.

Return this shape:
{{
  "coverage": {{"streams": 0, "days": {days}, "channels": ["..."]}},
  "recurring_themes": [
    {{
      "theme": "...", "affected_sectors": ["..."],
      "representative_tickers": ["..."], "historical_view": "...",
      "bullish_conditions": ["..."], "bearish_conditions": ["..."],
      "invalidation": ["..."], "typical_horizon_days": 20,
      "evidence_count": 0, "confidence_note": "..."
    }}
  ],
  "decision_frameworks": ["..."],
  "known_biases_and_limits": ["..."]
}}

Merge genuinely recurring ideas; do not turn one-off comments into a framework. A theme may
be applied to an unmentioned holding later only when sector exposure and current independent
market/fundamental evidence provide an explicit link. Include disagreement and invalidation.

HISTORICAL_STREAM_RECORDS:
{json.dumps(evidence, ensure_ascii=False)}

CHANNEL_RELIABILITY:
{json.dumps(metrics, ensure_ascii=False)}
"""
    response = OpenAI().responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"), input=prompt
    )
    playbook = clean_json(response.output_text)
    # Coverage and performance are computed from SQLite, never generated by AI.
    playbook["coverage"] = metrics["coverage"]
    playbook["backtest_metrics"] = metrics
    playbook["generated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(playbook, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Updated historical channel playbook from %d streams", len(evidence))
    return playbook


def load_channel_playbook(path: Path = PLAYBOOK_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {
            "coverage": {"streams": 0}, "recurring_themes": [],
            "decision_frameworks": [],
            "known_biases_and_limits": ["半年直播框架尚未建立。"],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def channel_playbook_is_stale(
    path: Path = PLAYBOOK_PATH, max_age_days: int = 7
) -> bool:
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("backtest_metrics"):
            return True
        generated = datetime.fromisoformat(str(data["generated_at"]).replace("Z", "+00:00"))
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - generated >= timedelta(days=max_age_days)
    except Exception:
        return True


def current_price(ticker: str) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
        return None if hist.empty else float(hist["Close"].iloc[-1])
    except Exception as exc:
        logging.warning("Current price failed for %s: %s", ticker, exc)
        return None


def _load_fundamentals_cache() -> Dict[str, Any]:
    if not FUNDAMENTALS_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(FUNDAMENTALS_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("Could not read fundamentals cache; rebuilding it")
        return {}


def _return_over_days(closes: List[float], days: int) -> Optional[float]:
    if len(closes) <= days or closes[-days - 1] <= 0:
        return None
    return closes[-1] / closes[-days - 1] - 1.0


def market_evidence(ticker: str, cache: Dict[str, Any]) -> Dict[str, Any]:
    stock = yf.Ticker(ticker)
    evidence: Dict[str, Any] = {
        "price": None, "return_20d": None, "return_60d": None,
        "return_120d": None, "drawdown_from_52w_high": None,
        "annualized_volatility": None,
    }
    try:
        hist = stock.history(period="1y", auto_adjust=True)
        closes = [float(value) for value in hist["Close"].dropna().tolist()]
        if closes:
            evidence["price"] = closes[-1]
            evidence["return_20d"] = _return_over_days(closes, 20)
            evidence["return_60d"] = _return_over_days(closes, 60)
            evidence["return_120d"] = _return_over_days(closes, 120)
            peak = max(closes)
            evidence["drawdown_from_52w_high"] = closes[-1] / peak - 1.0 if peak else None
            returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0]
            if len(returns) > 1:
                mean = sum(returns) / len(returns)
                variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
                evidence["annualized_volatility"] = math.sqrt(variance) * math.sqrt(252)
    except Exception as exc:
        logging.warning("Price evidence failed for %s: %s", ticker, exc)

    today = datetime.now(timezone.utc).date()
    cached = cache.get(ticker, {})
    try:
        fetched = date.fromisoformat(str(cached.get("fetched_date", "")))
    except ValueError:
        fetched = date.min
    cached_fields = cached.get("fields", {})
    metadata_missing = not any(
        key in cached_fields for key in ("exchange", "quoteType", "fullExchangeName")
    )
    if (today - fetched).days >= 7 or metadata_missing:
        fields: Dict[str, Any] = {}
        try:
            info = stock.get_info()
            for key in (
                "sector", "industry", "marketCap", "trailingPE", "forwardPE",
                "priceToBook", "enterpriseToEbitda", "revenueGrowth",
                "earningsGrowth", "profitMargins", "debtToEquity", "returnOnEquity",
                "exchange", "fullExchangeName", "quoteType", "longName",
            ):
                value = info.get(key)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    fields[key] = value
            stamp = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
            if isinstance(stamp, (int, float)):
                fields["nextEarningsDate"] = datetime.fromtimestamp(
                    stamp, timezone.utc
                ).date().isoformat()
        except Exception as exc:
            logging.warning("Fundamental evidence failed for %s: %s", ticker, exc)
        cached = {"fetched_date": today.isoformat(), "fields": fields}
        cache[ticker] = cached
    evidence["fundamentals"] = cached.get("fields", {})
    evidence["fundamentals_as_of"] = cached.get("fetched_date")
    return evidence


def portfolio_snapshot(portfolio_path: Path) -> Dict[str, Any]:
    config = json.loads(portfolio_path.read_text(encoding="utf-8"))
    evidence_cache: Dict[str, Dict[str, Any]] = {}
    fundamentals_cache = _load_fundamentals_cache()
    for account in config["accounts"]:
        positions = []
        stock_value = 0.0
        for ticker, shares in account["positions"].items():
            if ticker not in evidence_cache:
                evidence_cache[ticker] = market_evidence(ticker, fundamentals_cache)
            evidence = evidence_cache[ticker]
            price = evidence.get("price")
            value = float(shares) * price if price is not None else None
            if value is not None:
                stock_value += value
            positions.append({
                "ticker": ticker, "shares": shares, "price": price, "value": value,
                "market_evidence": evidence,
            })
        total = stock_value + float(account.get("cash_usd", 0.0))
        for position in positions:
            position["current_weight"] = (
                position["value"] / total if position["value"] is not None and total else None
            )
        account["positions_snapshot"] = positions
        account["estimated_total_value"] = total
    FUNDAMENTALS_CACHE_PATH.write_text(
        json.dumps(fundamentals_cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return config


def latest_research(conn: sqlite3.Connection, days: int = 7) -> List[Dict[str, Any]]:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    videos = conn.execute(
        """SELECT * FROM videos
        WHERE published_date >= ? AND content_type = 'stream'
        ORDER BY published_date DESC""", (cutoff,)
    ).fetchall()
    result = []
    for video in videos:
        signals = conn.execute(
            "SELECT ticker, stance, confidence, horizon_days, rationale FROM signals WHERE video_id = ?",
            (video["video_id"],),
        ).fetchall()
        result.append({
            "channel": video["channel_name"], "title": video["title"],
            "url": video["url"], "published_date": video["published_date"],
            "summary": video["summary"], "signals": [dict(row) for row in signals],
        })
    return result
