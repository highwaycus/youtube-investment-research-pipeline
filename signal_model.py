from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from market_features import (
    MARKET_DB_PATH,
    MARKET_NUMERIC_FEATURES,
    build_market_feature_frame,
    normalize_market_ticker,
    refresh_market_cache,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "signal_model.joblib"
MODEL_REPORT_PATH = ROOT / "signal_model_report.json"
MODEL_VERSION = 3
BASE_FEATURES = ["channel", "direction", "horizon_days", "ticker", "confidence"]
CATEGORICAL = ["channel", "direction", "horizon_days", "ticker"]
MARKET_FEATURES = BASE_FEATURES + MARKET_NUMERIC_FEATURES


def _load_training_frame(conn: sqlite3.Connection, cutoff: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        """
        SELECT v.video_id, date(v.published_date) AS published_date,
               date(p.exit_date) AS exit_date, v.channel_name AS channel,
               UPPER(s.ticker) AS ticker,
               CASE s.stance WHEN 1 THEN 'bullish' ELSE 'bearish' END AS direction,
               CAST(s.horizon_days AS TEXT) AS horizon_days,
               s.confidence, p.correct
        FROM performance p
        JOIN signals s ON s.id=p.signal_id
        JOIN videos v ON v.video_id=s.video_id
        WHERE v.content_type='stream' AND v.transcript_status='available'
          AND v.published_date>=? AND s.stance!=0 AND UPPER(s.ticker)!='SPY'
        ORDER BY v.published_date,v.video_id,s.id
        """,
        conn,
        params=(cutoff,),
    )
    if frame.empty:
        return frame
    frame["ticker"] = frame["ticker"].map(normalize_market_ticker)
    frame = frame[frame["ticker"].notna()].copy()
    frame["published_date"] = pd.to_datetime(frame["published_date"])
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    frame["sample_weight"] = (
        1.0 / frame.groupby("video_id")["video_id"].transform("count")
    )
    return frame


def _preprocessor(numeric: List[str], dense: bool) -> ColumnTransformer:
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", min_frequency=8, sparse_output=not dense
        )),
    ])
    numbers = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("categorical", categorical, CATEGORICAL),
        ("numeric", numbers, numeric),
    ])


def _candidates() -> Dict[str, Tuple[Pipeline, List[str]]]:
    base_numeric = ["confidence"]
    market_numeric = base_numeric + MARKET_NUMERIC_FEATURES
    return {
        "legacy_l2_logistic": (
            Pipeline([
                ("features", _preprocessor(base_numeric, False)),
                ("classifier", LogisticRegression(
                    C=0.01, max_iter=2000, solver="lbfgs"
                )),
            ]),
            BASE_FEATURES,
        ),
        "market_l2_logistic": (
            Pipeline([
                ("features", _preprocessor(market_numeric, False)),
                ("classifier", LogisticRegression(
                    C=0.03, max_iter=2500, solver="lbfgs"
                )),
            ]),
            MARKET_FEATURES,
        ),
        "market_hist_gradient_boosting": (
            Pipeline([
                ("features", _preprocessor(market_numeric, True)),
                ("classifier", HistGradientBoostingClassifier(
                    learning_rate=0.04, max_iter=100, max_depth=2,
                    min_samples_leaf=20, l2_regularization=3.0,
                    random_state=42,
                )),
            ]),
            MARKET_FEATURES,
        ),
        "market_random_forest": (
            Pipeline([
                ("features", _preprocessor(market_numeric, True)),
                ("classifier", RandomForestClassifier(
                    n_estimators=300, max_depth=4, min_samples_leaf=15,
                    max_features=0.6, n_jobs=-1, random_state=42,
                )),
            ]),
            MARKET_FEATURES,
        ),
    }


def _weights(frame: pd.DataFrame) -> np.ndarray:
    values = frame["sample_weight"].to_numpy(float)
    return values * len(values) / values.sum() if values.sum() > 0 else values


def _folds(frame: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    first = pd.Timestamp(
        year=frame["published_date"].min().year,
        month=frame["published_date"].min().month, day=1,
    )
    last = pd.Timestamp(
        year=frame["published_date"].max().year,
        month=frame["published_date"].max().month, day=1,
    )
    current = first + pd.DateOffset(months=1)
    output = []
    while current <= last:
        following = current + pd.DateOffset(months=1)
        # Purging prevents labels unknown at the test date from entering train.
        train = frame[
            (frame["published_date"] < current)
            & (frame["exit_date"] < current)
        ].copy()
        test = frame[
            (frame["published_date"] >= current)
            & (frame["published_date"] < following)
        ].copy()
        if len(train) >= 120 and len(test) >= 30:
            output.append((train, test))
        current = following
    return output


def _fit(
    model: Pipeline, frame: pd.DataFrame, features: List[str], weights: np.ndarray
) -> None:
    model.fit(
        frame[features], frame["correct"].astype(int),
        classifier__sample_weight=weights,
    )


def _metrics(
    labels: List[int], probabilities: List[float], weights: List[float]
) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    auc = None
    if len(set(y)) >= 2:
        auc = float(roc_auc_score(y, probability, sample_weight=sample_weight))
    return {
        "test_samples": len(y),
        "log_loss": float(log_loss(y, probability, sample_weight=sample_weight)),
        "brier_score": float(
            brier_score_loss(y, probability, sample_weight=sample_weight)
        ),
        "auc": auc,
    }


def _validate(frame: pd.DataFrame) -> Dict[str, Any]:
    candidates = _candidates()
    stores = {
        name: {"labels": [], "probabilities": [], "weights": []}
        for name in candidates
    }
    baseline = {"labels": [], "probabilities": [], "weights": []}
    fold_reports = []
    for train, test in _folds(frame):
        train_weight, test_weight = _weights(train), _weights(test)
        mean = float(np.average(
            train["correct"].astype(int), weights=train_weight
        ))
        mean_probability = np.full(len(test), mean)
        baseline["labels"].extend(int(x) for x in test["correct"])
        baseline["probabilities"].extend(float(x) for x in mean_probability)
        baseline["weights"].extend(float(x) for x in test_weight)
        fold_report = {
            "test_month": test["published_date"].min().strftime("%Y-%m"),
            "training_samples": len(train), "test_samples": len(test),
            "models": {},
        }
        for name, (model, features) in candidates.items():
            try:
                _fit(model, train, features, train_weight)
                probability = model.predict_proba(test[features])[:, 1]
                fold_report["models"][name] = {
                    "log_loss": float(log_loss(
                        test["correct"].astype(int), probability,
                        sample_weight=test_weight,
                    ))
                }
                stores[name]["labels"].extend(int(x) for x in test["correct"])
                stores[name]["probabilities"].extend(float(x) for x in probability)
                stores[name]["weights"].extend(float(x) for x in test_weight)
            except Exception as exc:
                fold_report["models"][name] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        fold_reports.append(fold_report)
    if not baseline["labels"]:
        return {
            "folds": fold_reports, "test_samples": 0,
            "passed_activation_gate": False,
            "failure_reason": "Not enough purged walk-forward test data.",
        }
    baseline_metrics = _metrics(**baseline)
    candidate_metrics = {}
    for name, values in stores.items():
        candidate_metrics[name] = (
            _metrics(**values)
            if len(values["labels"]) == len(baseline["labels"])
            else {"test_samples": len(values["labels"]),
                  "error": "Candidate failed in one or more folds."}
        )
    eligible = {
        name: value for name, value in candidate_metrics.items()
        if name.startswith("market_") and "log_loss" in value
    }
    selected = min(
        eligible, key=lambda name: eligible[name]["log_loss"]
    ) if eligible else None
    chosen = eligible.get(selected or "", {})
    passed = bool(
        selected and len(fold_reports) >= 3
        and baseline_metrics["test_samples"] >= 200
        and chosen["log_loss"] <= baseline_metrics["log_loss"] * 0.99
        and chosen["brier_score"] < baseline_metrics["brier_score"]
        and chosen.get("auc") is not None and chosen["auc"] >= 0.52
    )
    return {
        "folds": fold_reports, "test_samples": baseline_metrics["test_samples"],
        "baseline": baseline_metrics, "candidates": candidate_metrics,
        "selected_market_model": selected,
        "passed_activation_gate": passed,
        "activation_gate": {
            "minimum_folds": 3, "minimum_test_samples": 200,
            "log_loss_improvement": "at least 1% versus expanding historical mean",
            "brier_score": "lower than baseline", "minimum_auc": 0.52,
        },
        "failure_reason": (
            None if passed else
            "No market-feature model consistently beat the time-ordered baseline."
        ),
    }


def train_signal_model(
    conn: sqlite3.Connection, cutoff: str, model_path: Path = MODEL_PATH,
    report_path: Path = MODEL_REPORT_PATH,
    market_db_path: Path = MARKET_DB_PATH, refresh_data: bool = True,
) -> Dict[str, Any]:
    frame = _load_training_frame(conn, cutoff)
    if len(frame) < 120 or frame["correct"].nunique() < 2:
        report = {
            "model_version": MODEL_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "active": False, "training_samples": len(frame),
            "reason": "Not enough mature non-SPY signals to train.",
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report
    fetch_report = (
        refresh_market_cache(frame, market_db_path) if refresh_data else {}
    )
    enriched, coverage = build_market_feature_frame(frame, market_db_path)
    if coverage["price_feature_coverage"] < 0.70 and model_path.exists():
        try:
            prior = joblib.load(model_path)
            prior_report = dict(prior.get("report") or {})
            if (
                prior.get("model_version") == MODEL_VERSION
                and prior_report.get("active")
            ):
                prior_report["generated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                prior_report["last_retrain_attempt"] = {
                    "status": "kept_previous_active_model",
                    "reason": "New point-in-time price coverage was below 70%.",
                    "feature_coverage": coverage,
                    "market_data_fetch": fetch_report,
                }
                report_path.write_text(
                    json.dumps(prior_report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logging.warning(
                    "Kept prior active model because market data refresh was incomplete"
                )
                return prior_report
        except Exception as exc:
            logging.warning("Could not preserve prior model: %s", exc)
    validation = _validate(enriched)
    selected = validation.get("selected_market_model")
    active = bool(validation.get("passed_activation_gate"))
    candidates = _candidates()
    if selected not in candidates:
        selected, active = "market_l2_logistic", False
    model, features = candidates[selected]
    _fit(model, enriched, features, _weights(enriched))
    if coverage["price_feature_coverage"] < 0.70:
        active = False
        validation["passed_activation_gate"] = False
        validation["failure_reason"] = (
            "Point-in-time price feature coverage is below 70%."
        )
    report = {
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_type": selected,
        "target": "Probability a directional signal beats SPY over its horizon",
        "features": features,
        "training_first_date": enriched["published_date"].min().date().isoformat(),
        "training_last_date": enriched["published_date"].max().date().isoformat(),
        "training_samples": len(enriched),
        "training_streams": int(enriched["video_id"].nunique()),
        "stream_balanced_sample_weighting": True,
        "feature_coverage": coverage, "market_data_fetch": fetch_report,
        "active": active, "validation": validation,
        "limitations": [
            "Nearby signals and repeated themes remain correlated.",
            "The dataset is concentrated in two channels and technology themes.",
            "Historical valuation uses only previously reported EPS; unavailable "
            "point-in-time fundamentals remain missing.",
            "This is a signal-reliability model, not a return or allocation model.",
            "An inactive model must not influence recommendations.",
        ],
    }
    joblib.dump({
        "model_version": MODEL_VERSION, "pipeline": model,
        "feature_columns": features, "report": report,
    }, model_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info(
        "Compared models from %d signals; selected=%s active=%s",
        len(enriched), selected, active,
    )
    return report


def load_signal_model(model_path: Path = MODEL_PATH) -> Optional[Dict[str, Any]]:
    if not model_path.exists():
        return None
    try:
        artifact = joblib.load(model_path)
        return artifact if artifact.get("model_version") == MODEL_VERSION else None
    except Exception as exc:
        logging.warning("Could not load signal model: %s", exc)
        return None


def signal_model_is_stale(
    conn: sqlite3.Connection, cutoff: str,
    report_path: Path = MODEL_REPORT_PATH, max_age_days: int = 7,
) -> bool:
    if not MODEL_PATH.exists() or not report_path.exists():
        return True
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("model_version") != MODEL_VERSION:
            return True
        generated = datetime.fromisoformat(
            str(report["generated_at"]).replace("Z", "+00:00")
        )
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - generated >= timedelta(days=max_age_days):
            return True
        count = conn.execute(
            """
            SELECT COUNT(*) FROM performance p
            JOIN signals s ON s.id=p.signal_id
            JOIN videos v ON v.video_id=s.video_id
            WHERE v.content_type='stream' AND v.transcript_status='available'
              AND v.published_date>=? AND s.stance!=0 AND UPPER(s.ticker)!='SPY'
            """,
            (cutoff,),
        ).fetchone()[0]
        return int(report.get("training_samples", -1)) != int(count)
    except Exception:
        return True


def _probability_weight(probability: float) -> float:
    if probability >= 0.65:
        return 1.0
    if probability >= 0.58:
        return 0.8
    if probability >= 0.48:
        return 0.6
    if probability >= 0.40:
        return 0.4
    return 0.25


def annotate_research_with_model(
    research: List[Dict[str, Any]], artifact: Optional[Dict[str, Any]],
    market_db_path: Path = MARKET_DB_PATH,
) -> List[Dict[str, Any]]:
    output = json.loads(json.dumps(research, ensure_ascii=False))
    if not artifact:
        return output
    report, model = artifact["report"], artifact["pipeline"]
    features = artifact["feature_columns"]
    active = bool(report.get("active"))
    validation = report.get("validation") or {}
    selected = validation.get("selected_market_model")
    records, references = [], []
    for video in output:
        for signal in video.get("signals", []):
            raw_ticker, stance = str(signal.get("ticker", "")).upper(), int(
                signal.get("stance", 0)
            )
            ticker = normalize_market_ticker(raw_ticker)
            historical = signal.get("historical_backtest") or {}
            rule_weight = float(historical.get("stream_evidence_weight", 0.6))
            if ticker is None or ticker == "SPY" or stance == 0:
                signal["trained_signal_model"] = {
                    "status": "not_scored",
                    "reason": (
                        "unsupported or invalid market ticker"
                        if ticker is None else
                        "SPY benchmark self-comparison or neutral stance"
                    ),
                    "active": active,
                }
                historical["effective_stream_evidence_weight"] = rule_weight
                continue
            records.append({
                "published_date": video.get("published_date", ""),
                "channel": video.get("channel", ""),
                "direction": "bullish" if stance > 0 else "bearish",
                "horizon_days": str(int(signal.get("horizon_days", 20))),
                "ticker": ticker,
                "confidence": float(signal.get("confidence", 0.5)),
            })
            references.append((signal, rule_weight))
    if not records:
        return output
    # Keep prediction-time trend and industry features current even when no
    # newly matured label makes the weekly training artifact stale.
    try:
        refresh_market_cache(pd.DataFrame(records), market_db_path)
    except Exception as exc:
        logging.warning("Could not refresh prediction market features: %s", exc)
    enriched, _ = build_market_feature_frame(
        pd.DataFrame(records), market_db_path
    )
    probabilities = model.predict_proba(enriched[features])[:, 1]
    chosen = (validation.get("candidates") or {}).get(selected or "", {})
    baseline = validation.get("baseline") or {}
    for (signal, rule_weight), probability in zip(references, probabilities):
        probability = float(probability)
        model_weight = _probability_weight(probability)
        effective = model_weight if active else rule_weight
        historical = signal.get("historical_backtest") or {}
        historical["effective_stream_evidence_weight"] = effective
        signal["trained_signal_model"] = {
            "status": "active" if active else "experimental_not_activated",
            "model_type": report.get("model_type"),
            "estimated_success_probability": probability,
            "model_evidence_weight": model_weight,
            "effective_stream_evidence_weight": effective,
            "validation_auc": chosen.get("auc"),
            "validation_model_log_loss": chosen.get("log_loss"),
            "validation_baseline_log_loss": baseline.get("log_loss"),
            "interpretation": (
                "Used only after passing purged walk-forward validation."
                if active else
                "Research only; it failed validation and must not affect recommendations."
            ),
        }
    return output
