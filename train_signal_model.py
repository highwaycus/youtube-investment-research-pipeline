from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv

from research import _research_cutoff, connect_db
from signal_model import train_signal_model


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(root, ".env"))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    conn = connect_db()
    days = int(os.getenv("BACKFILL_DAYS", "183"))
    report = train_signal_model(conn, cutoff=_research_cutoff(conn, days))
    validation = report.get("validation") or {}
    print()
    print("Signal model training complete.")
    print(f"Training samples: {report.get('training_samples', 0)}")
    print(f"Walk-forward test samples: {validation.get('test_samples', 0)}")
    baseline = validation.get("baseline") or {}
    selected = validation.get("selected_market_model")
    selected_metrics = (validation.get("candidates") or {}).get(selected, {})
    print()
    print("Time-ordered comparison (lower log loss is better):")
    print(f"  historical_mean_baseline: {baseline.get('log_loss', 'n/a')}")
    for name, metrics in (validation.get("candidates") or {}).items():
        print(
            f"  {name}: log_loss={metrics.get('log_loss', 'n/a')} "
            f"AUC={metrics.get('auc', 'n/a')}"
        )
    coverage = report.get("feature_coverage") or {}
    fetched = report.get("market_data_fetch") or {}
    price_fetch = fetched.get("price") or {}
    earnings_fetch = fetched.get("earnings") or {}
    print()
    print(f"Selected market model: {selected or 'none'}")
    print(f"Selected model log loss: {selected_metrics.get('log_loss', 'n/a')}")
    print(f"Selected model AUC: {selected_metrics.get('auc', 'n/a')}")
    print(
        "Point-in-time feature coverage: "
        f"price={coverage.get('price_feature_coverage', 0):.1%}, "
        f"valuation={coverage.get('valuation_coverage', 0):.1%}, "
        f"earnings={coverage.get('earnings_feature_coverage', 0):.1%}"
    )
    print(
        "Market data refresh: "
        f"prices downloaded={price_fetch.get('downloaded', 0)}, "
        f"price failures={price_fetch.get('failed_count', 0)}, "
        f"earnings downloaded={earnings_fetch.get('downloaded', 0)}, "
        f"earnings failures={earnings_fetch.get('failed_count', 0)}"
    )
    print(f"Active in daily report: {report.get('active', False)}")
    if not report.get("active"):
        print(
            "The model remains experimental because it did not beat the "
            "time-ordered baseline."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Signal model training failed")
        raise SystemExit(1)
