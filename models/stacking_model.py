"""Stacking ensemble model for Osaka weather forecasting."""
from __future__ import annotations

import datetime
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import importlib.util
import numpy as np

# Optional imports without try/except to keep GH Actions stable
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None
XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None

if SKLEARN_AVAILABLE:
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import LinearRegression
else:  # pragma: no cover - fallback when sklearn is absent
    RandomForestRegressor = None  # type: ignore[assignment]
    StackingRegressor = None  # type: ignore[assignment]
    LinearRegression = None  # type: ignore[assignment]

if LIGHTGBM_AVAILABLE:
    from lightgbm import LGBMRegressor
else:  # pragma: no cover - fallback when LightGBM is absent
    LGBMRegressor = None  # type: ignore[assignment]

if not LIGHTGBM_AVAILABLE and XGBOOST_AVAILABLE:
    from xgboost import XGBRegressor
else:  # pragma: no cover - fallback when neither LightGBM nor XGBoost is available
    XGBRegressor = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _normalize_timestamp(ts: str) -> str:
    """Normalize timestamps to ISO without seconds for consistent alignment."""
    try:
        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%dT%H:%M")
    except Exception:  # pragma: no cover - defensive fallback
        return ts


def _convert_real_entries(real_data: Dict[str, object]) -> List[Dict[str, object]]:
    """Convert real weather payload into a list of entries if needed."""
    if "entries" in real_data and isinstance(real_data.get("entries"), list):
        return real_data.get("entries", [])  # type: ignore[return-value]

    times = real_data.get("time") or []
    temps = real_data.get("temp") or []
    rains = real_data.get("rain") or []
    codes = real_data.get("code") or []

    entries: List[Dict[str, object]] = []
    for idx, ts in enumerate(times):
        entries.append(
            {
                "time": ts,
                "temperature": float(temps[idx]) if idx < len(temps) else 0.0,
                "precipitation_probability": float(rains[idx]) if idx < len(rains) else 0.0,
                "weathercode": int(codes[idx]) if idx < len(codes) else 0,
            }
        )

    return entries


def _build_feature_matrix(entries: List[Dict[str, object]]) -> Tuple[np.ndarray, List[str]]:
    """Create a feature matrix from forecast entries."""
    features: List[List[float]] = []
    feature_names = [
        "month",
        "hour",
        "baseline_temp",
        "baseline_rain",
        "weather_code",
        "hour_sin",
        "hour_cos",
    ]

    for entry in entries:
        ts = _normalize_timestamp(str(entry.get("time", "")))
        try:
            dt = datetime.datetime.fromisoformat(ts)
        except Exception:  # pragma: no cover - fallback for malformed timestamps
            dt = datetime.datetime.now()

        hour = float(dt.hour)
        month = float(dt.month)
        baseline_temp = float(entry.get("temperature", 0.0))
        baseline_rain = float(entry.get("precipitation_probability", 0.0))
        weather_code = float(entry.get("weathercode", 0.0))

        # cyclic encodings for hour-of-day
        hour_rad = 2 * math.pi * (hour / 24.0)
        features.append(
            [
                month,
                hour,
                baseline_temp,
                baseline_rain,
                weather_code,
                math.sin(hour_rad),
                math.cos(hour_rad),
            ]
        )

    return np.array(features, dtype=float), feature_names


def load_features(
    forecast_data: Dict[str, object], real_data: Dict[str, object]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Align forecast and real observations to build supervised features."""
    forecast_entries = forecast_data.get("entries", [])
    real_entries = _convert_real_entries(real_data)

    real_map = {
        _normalize_timestamp(str(entry.get("time", ""))): entry for entry in real_entries
    }

    X_list: List[Dict[str, object]] = []
    temp_targets: List[float] = []
    rain_targets: List[float] = []

    for entry in forecast_entries:
        ts = _normalize_timestamp(str(entry.get("time", "")))
        actual = real_map.get(ts)
        if not actual:
            continue

        X_list.append(entry)
        temp_targets.append(float(actual.get("temperature", 0.0)))
        rain_targets.append(float(actual.get("precipitation_probability", 0.0)))

    X_matrix, feature_names = _build_feature_matrix(X_list)

    return X_matrix, np.array(temp_targets, dtype=float), np.array(rain_targets, dtype=float), feature_names


def _get_boosting_regressor():
    if LIGHTGBM_AVAILABLE and LGBMRegressor is not None:
        return LGBMRegressor(random_state=42)
    if XGBOOST_AVAILABLE and XGBRegressor is not None:
        return XGBRegressor(random_state=42, n_estimators=120, max_depth=4)
    if SKLEARN_AVAILABLE:
        return RandomForestRegressor(random_state=42)
    return None


def build_stacking_regressor() -> Optional[StackingRegressor]:
    """Build a stacking regressor with RF + LightGBM and LinearRegression meta-learner."""
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn is not installed; stacking model disabled")
        return None

    base_estimators = []
    if RandomForestRegressor is not None:
        base_estimators.append(("rf", RandomForestRegressor(n_estimators=150, random_state=42)))

    boosting_model = _get_boosting_regressor()
    if boosting_model is not None:
        base_estimators.append(("boost", boosting_model))

    if not base_estimators:
        logger.warning("No base estimators available; stacking model disabled")
        return None

    final_estimator = LinearRegression() if LinearRegression is not None else None
    if final_estimator is None:
        logger.warning("LinearRegression missing; stacking model disabled")
        return None

    return StackingRegressor(estimators=base_estimators, final_estimator=final_estimator)


def train_stacking(
    forecast_data: Dict[str, object], real_data: Dict[str, object]
) -> Optional[Dict[str, object]]:
    """Train stacking models for temperature and precipitation."""
    X, temp_y, rain_y, feature_names = load_features(forecast_data, real_data)

    if X.size == 0 or temp_y.size == 0 or rain_y.size == 0:
        logger.warning("Insufficient data for stacking training; falling back to baseline")
        return None

    temp_model = build_stacking_regressor()
    rain_model = build_stacking_regressor()

    if temp_model is None or rain_model is None:
        return None

    temp_model.fit(X, temp_y)
    rain_model.fit(X, rain_y)

    return {
        "temp_model": temp_model,
        "rain_model": rain_model,
        "feature_names": feature_names,
    }


def predict_stacking(
    forecast_data: Dict[str, object],
    real_data: Optional[Dict[str, object]] = None,
    hybrid_weight: Optional[float] = None,
) -> Dict[str, object]:
    """Predict forecast values using the trained stacking ensemble.

    If training fails or dependencies are missing, the baseline forecast_data is returned.
    """
    if real_data is None:
        logger.warning("No real data provided; returning baseline forecast")
        return forecast_data

    model_bundle = train_stacking(forecast_data, real_data)
    if not model_bundle:
        return forecast_data

    forecast_entries: List[Dict[str, object]] = forecast_data.get("entries", [])
    X_matrix, _ = _build_feature_matrix(forecast_entries)

    temp_preds = model_bundle["temp_model"].predict(X_matrix)
    rain_preds = model_bundle["rain_model"].predict(X_matrix)

    adjusted_entries: List[Dict[str, object]] = []
    hybrid_alpha = None
    if hybrid_weight is not None:
        hybrid_alpha = max(0.0, min(1.0, hybrid_weight))

    for entry, t_pred, r_pred in zip(forecast_entries, temp_preds, rain_preds):
        baseline_temp = float(entry.get("temperature", 0.0))
        baseline_rain = float(entry.get("precipitation_probability", 0.0))

        if hybrid_alpha is not None:
            temp_value = hybrid_alpha * baseline_temp + (1 - hybrid_alpha) * float(t_pred)
            rain_value = hybrid_alpha * baseline_rain + (1 - hybrid_alpha) * float(r_pred)
        else:
            temp_value = float(t_pred)
            rain_value = float(r_pred)

        adjusted_entry = dict(entry)
        adjusted_entry["temperature"] = round(temp_value, 2)
        adjusted_entry["precipitation_probability"] = int(round(max(0.0, min(100.0, rain_value))))
        adjusted_entries.append(adjusted_entry)

    return {
        "generated_at": forecast_data.get("generated_at"),
        "model_parameters": forecast_data.get("model_parameters", {}),
        "entries": adjusted_entries,
    }


__all__ = [
    "load_features",
    "build_stacking_regressor",
    "train_stacking",
    "predict_stacking",
]
