import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


# The production Windows environment installs these from requirements.txt.
# Lightweight stubs keep deterministic unit tests independent of network-only
# adapters in this workspace.
if "yfinance" not in sys.modules:
    module = types.ModuleType("yfinance")
    module.Ticker = object
    module.download = lambda *args, **kwargs: None
    sys.modules["yfinance"] = module
if "openai" not in sys.modules:
    module = types.ModuleType("openai")
    module.OpenAI = object
    sys.modules["openai"] = module
if "dotenv" not in sys.modules:
    module = types.ModuleType("dotenv")
    module.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = module
if "youtube_transcript_api" not in sys.modules:
    module = types.ModuleType("youtube_transcript_api")
    module.YouTubeTranscriptApi = object
    sys.modules["youtube_transcript_api"] = module

from candidate_screening import build_new_candidate_universe
from holding_screening import build_holding_reviews
from main import REPORT_VERSION, generate_allocation_report
from market_features import normalize_market_ticker, refresh_market_cache
from research import _performance_ticker


def good_asset(ticker):
    if ticker == "SPY":
        return {"price": 600, "return_20d": 0.02, "return_60d": 0.05,
                "return_120d": 0.10, "drawdown_from_52w_high": -0.02,
                "annualized_volatility": 0.18,
                "fundamentals": {"exchange": "PCX", "quoteType": "ETF"}}
    return {
        "price": 100, "return_20d": 0.10, "return_60d": 0.20,
        "return_120d": 0.30, "drawdown_from_52w_high": -0.05,
        "annualized_volatility": 0.35, "fundamentals_as_of": "2026-08-12",
        "fundamentals": {
            "exchange": "NMS", "quoteType": "EQUITY", "sector": "Technology",
            "forwardPE": 30, "revenueGrowth": 0.25, "earningsGrowth": 0.30,
            "profitMargins": 0.20, "returnOnEquity": 0.25, "debtToEquity": 30,
            "nextEarningsDate": "2026-10-30",
        },
    }


class CandidateUniverseTests(unittest.TestCase):
    def test_us_only_and_exposure_deduplication(self):
        portfolio = {"accounts": [{"name": "IBKR", "positions": {
            "GOOGL": 1, "BRK-A": 1, "TSM": 1,
        }}]}
        research = [{"channel": "x", "published_date": "2026-08-12", "signals": [
            {"ticker": "2330.TW", "stance": 1},
            {"ticker": "GOOG", "stance": 1},
            {"ticker": "BRK.B", "stance": 1},
            {"ticker": "VRTV", "stance": 1},
            {"ticker": "CWAV", "stance": 1},
            {"ticker": "ANET", "stance": 1},
        ]}]
        with tempfile.TemporaryDirectory() as directory:
            candidates = build_new_candidate_universe(
                portfolio, research,
                evidence_provider=lambda ticker, cache: good_asset(ticker),
                cache_path=Path(directory) / "fundamentals.json",
            )
        self.assertEqual([item["ticker"] for item in candidates], ["ANET"])

    def test_missing_exchange_metadata_fails_closed(self):
        portfolio = {"accounts": [{"name": "IBKR", "positions": {}}]}
        research = [{"signals": [{"ticker": "ANET", "stance": 1}]}]
        asset = good_asset("ANET")
        asset["fundamentals"].pop("exchange")
        with tempfile.TemporaryDirectory() as directory:
            candidates = build_new_candidate_universe(
                portfolio, research, evidence_provider=lambda ticker, cache: (
                    good_asset("SPY") if ticker == "SPY" else asset
                ), cache_path=Path(directory) / "fundamentals.json",
            )
        self.assertEqual(candidates, [])


class HoldingReviewTests(unittest.TestCase):
    def test_bearish_stream_needs_independent_confirmation(self):
        bad = good_asset("BAD")
        bad.update({"return_20d": -0.12, "return_60d": -0.20,
                    "return_120d": -0.30, "drawdown_from_52w_high": -0.40})
        bad["fundamentals"].update({"revenueGrowth": -0.30,
                                    "earningsGrowth": -0.40})
        good = good_asset("GOOD")
        portfolio = {"accounts": [{
            "name": "IBKR", "max_single_weight": 0.15,
            "positions_snapshot": [
                {"ticker": "BAD", "current_weight": 0.10, "market_evidence": bad},
                {"ticker": "GOOD", "current_weight": 0.10, "market_evidence": good},
                {"ticker": "TSM", "current_weight": 0.17, "market_evidence": good},
            ],
        }]}
        research = [{"signals": [
            {"ticker": "BAD", "stance": -1},
            {"ticker": "GOOD", "stance": -1},
        ]}]
        reviews = build_holding_reviews(
            portfolio, research, evidence_provider=lambda ticker, cache: good_asset(ticker)
        )["IBKR"]
        by_ticker = {item["ticker"]: item for item in reviews}
        self.assertEqual(by_ticker["BAD"]["status"], "consider_exit")
        self.assertEqual(by_ticker["GOOD"]["status"], "hold")
        self.assertEqual(by_ticker["TSM"]["status"], "reduce_to_limit")
        self.assertEqual(by_ticker["TSM"]["target_weight"], 0.15)

    def test_price_weakness_without_stream_signal_does_not_trigger_bulk_trim(self):
        weak = good_asset("WEAK")
        weak.update({"return_20d": -0.12, "return_60d": -0.20,
                     "return_120d": -0.30, "drawdown_from_52w_high": -0.40})
        portfolio = {"accounts": [{
            "name": "Robinhood", "max_single_weight": 0.15,
            "positions_snapshot": [
                {"ticker": "WEAK", "current_weight": 0.08, "market_evidence": weak},
            ],
        }]}
        review = build_holding_reviews(
            portfolio, [], evidence_provider=lambda ticker, cache: good_asset(ticker)
        )["Robinhood"][0]
        self.assertEqual(review["status"], "hold")

    def test_confirmed_bearish_stream_can_reduce_but_not_micro_trim(self):
        weak = good_asset("WEAK")
        weak.update({"return_20d": -0.12, "return_60d": 0.10,
                     "drawdown_from_52w_high": -0.20})
        portfolio = {"accounts": [{
            "name": "Robinhood", "max_single_weight": 0.15,
            "positions_snapshot": [
                {"ticker": "WEAK", "current_weight": 0.08, "market_evidence": weak},
                {"ticker": "TINY", "current_weight": 0.005, "market_evidence": weak},
            ],
        }]}
        research = [{"signals": [
            {"ticker": "WEAK", "stance": -1}, {"ticker": "TINY", "stance": -1},
        ]}]
        reviews = build_holding_reviews(
            portfolio, research, evidence_provider=lambda ticker, cache: good_asset(ticker)
        )["Robinhood"]
        by_ticker = {item["ticker"]: item for item in reviews}
        self.assertEqual(by_ticker["WEAK"]["status"], "consider_reduce")
        self.assertEqual(by_ticker["TINY"]["status"], "hold")


class PerformanceTickerTests(unittest.TestCase):
    def test_us_normalization_happens_before_historical_download(self):
        self.assertEqual(_performance_ticker("BRK.B"), "BRK-B")
        self.assertIsNone(_performance_ticker("8299.TW"))
        self.assertIsNone(_performance_ticker("VRTV"))

    def test_market_cache_normalizes_before_real_download_boundary(self):
        observations = pd.DataFrame([
            {"ticker": "BRK.B", "published_date": "2026-08-12"},
            {"ticker": "CWAV", "published_date": "2026-08-12"},
            {"ticker": "8299.TW", "published_date": "2026-08-12"},
        ])
        captured = {}

        def fake_prices(conn, tickers, start, end):
            captured["prices"] = list(tickers)
            return {"downloaded": [], "failed_tickers": []}

        def fake_earnings(conn, counts):
            captured["earnings"] = dict(counts)
            return {"downloaded": [], "failed_tickers": []}

        with tempfile.TemporaryDirectory() as directory, patch(
            "market_features.refresh_prices", side_effect=fake_prices
        ), patch("market_features.refresh_earnings", side_effect=fake_earnings):
            refresh_market_cache(
                observations, Path(directory) / "market_features.db"
            )
        self.assertEqual(normalize_market_ticker("BRK.B"), "BRK-B")
        self.assertNotIn("BRK.B", captured["prices"])
        self.assertNotIn("CWAV", captured["prices"])
        self.assertNotIn("8299.TW", captured["prices"])
        self.assertIn("BRK-B", captured["prices"])
        self.assertEqual(captured["earnings"], {"BRK-B": 1})


class HoldingTargetTests(unittest.TestCase):
    def test_overweight_bearish_position_uses_explicit_cap_not_25pct_trim(self):
        weak = good_asset("NVDA")
        weak.update({"return_20d": 0.02, "return_60d": -0.08})
        portfolio = {"accounts": [{
            "name": "Robinhood", "max_single_weight": 0.15,
            "positions_snapshot": [{
                "ticker": "NVDA", "current_weight": 0.171,
                "market_evidence": weak,
            }],
        }]}
        research = [{"signals": [
            {"ticker": "NVDA", "stance": -1},
            {"ticker": "NVDA", "stance": -1},
            {"ticker": "NVDA", "stance": 1},
        ]}]
        review = build_holding_reviews(
            portfolio, research,
            evidence_provider=lambda ticker, cache: good_asset(ticker),
        )["Robinhood"][0]
        self.assertEqual(review["status"], "consider_reduce")
        self.assertEqual(review["target_weight"], 0.15)


class ReportTests(unittest.TestCase):
    def test_compact_report_has_version_and_no_legacy_sections(self):
        report = generate_allocation_report(
            {}, [], [], {}, [], holding_reviews={}, stream_model_active=False
        )
        self.assertIn(f"版本：v{REPORT_VERSION}", report)
        self.assertIn("今日無建議交易", report)
        self.assertIn("## 建議動作", report)
        self.assertNotIn("頻道歷史證據", report)
        self.assertNotIn("2330.TW", report)

    def test_screen_to_report_flow_is_auditable(self):
        portfolio = {"accounts": [{
            "name": "IBKR", "max_single_weight": 0.15, "positions": {},
            "positions_snapshot": [],
        }, {
            "name": "Robinhood", "max_single_weight": 0.15, "positions": {},
            "positions_snapshot": [],
        }]}
        research = [{"channel": "x", "published_date": "2026-08-12",
                     "signals": [{"ticker": "ANET", "stance": 1}]}]
        with tempfile.TemporaryDirectory() as directory:
            candidates = build_new_candidate_universe(
                portfolio, research,
                evidence_provider=lambda ticker, cache: good_asset(ticker),
                cache_path=Path(directory) / "fundamentals.json",
            )
        report = generate_allocation_report(
            portfolio, [], research, {}, candidates,
            holding_reviews={"IBKR": [], "Robinhood": []},
            stream_model_active=False,
        )
        self.assertIn("ANET", report)
        self.assertIn("20日報酬+10.0%", report)
        self.assertIn("Forward P/E 30.0", report)
        self.assertIn("直播預測模型尚未通過啟用門檻", report)
        self.assertIn("進一步研究", report)
        self.assertIn("並非買入指示", report)

    def test_concentration_action_does_not_relabel_unused_market_flag(self):
        reviews = {"IBKR": [{
            "ticker": "TSM", "status": "reduce_to_limit",
            "current_weight": 0.166, "target_weight": 0.15,
            "trigger": "單檔權重超過帳戶 15% 上限，屬集中度管理",
            "stream_signals": {"bullish": 4, "bearish": 0},
            "market_red_flags": ["120 日落後產業 ETF 超過 10%"],
            "fundamental_red_flags": [],
            "metrics": {},
        }]}
        report = generate_allocation_report(
            {}, [], [], {}, [], holding_reviews=reviews,
            stream_model_active=False,
        )
        self.assertIn("集中度超過 15%", report)
        self.assertNotIn("120 日落後產業 ETF 超過 10%", report)

    def test_etf_candidate_explains_applicable_checks_and_account_limits(self):
        candidate = {
            "ticker": "GLD",
            "return_20d": 0.087,
            "return_60d": -0.030,
            "relative_spy_20d": 0.064,
            "fundamentals": {"quoteType": "ETF"},
            "account_screening": {
                "IBKR": {
                    "status": "eligible_for_portfolio_review",
                    "max_initial_weight": 0.05,
                },
                "Robinhood": {
                    "status": "eligible_for_portfolio_review",
                    "max_initial_weight": 0.03,
                },
            },
        }
        report = generate_allocation_report(
            {}, [], [], {}, [candidate],
            holding_reviews={"IBKR": [], "Robinhood": []},
            stream_model_active=False,
        )
        self.assertIn("IBKR ≤5%；Robinhood ≤3%", report)
        self.assertIn("通過：20日報酬+8.7%、20日相對SPY+6.4%", report)
        self.assertIn("未通過：60日報酬-3.0%", report)
        self.assertIn("ETF不適用公司P/E、營收、獲利與財報日期", report)
        self.assertNotIn("通過價格、相對強弱、估值與基本面初篩", report)


if __name__ == "__main__":
    unittest.main()
