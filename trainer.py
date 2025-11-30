"""
trainer.py
Self-Learning Osaka Weather AI
Safely handles first-run cases / missing data.
"""

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from zoneinfo import ZoneInfo

from osaka_forecast_engine import forecast_to_json, synthesize_osaka_forecast
from predict_real_weather import fetch_real_weather
from requests import RequestException

MODEL_PATH = Path("data/model_param.json")
FORECAST_PATH = Path("data/forecast.json")
REAL_WEATHER_PATH = Path("data/real_weather.json")
LEARNING_RATE = 0.05


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_model_params() -> Dict[str, float]:
    if MODEL_PATH.exists():
        try:
            data = load_json(MODEL_PATH)
            return {
                "temp_bias": float(data.get("temp_bias", 0.0)),
                "rain_bias": float(data.get("rain_bias", 0.0)),
                "cloud_to_rain_scale": float(data.get("cloud_to_rain_scale", 1.0)),
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return {"temp_bias": 0.0, "rain_bias": 0.0, "cloud_to_rain_scale": 1.0}


def mape_style_error(forecast_values: List[float], actual_values: List[float]) -> float:
    if not forecast_values:
        return 0.0
    errors = [
        abs(f - a) / max(abs(a), 0.1) for f, a in zip(forecast_values, actual_values)
    ]
    return mean(errors)


def ensure_forecast_alignment(real_data: dict) -> dict:
    """Guarantee that forecast.json exists and overlaps with the provided real data."""

    if not FORECAST_PATH.exists():
        entries = real_data.get("entries", [])
        if not entries:
            raise ValueError("real_weather.json has no entries to align with")
        start_ts = entries[0]["time"]
        start_dt = datetime.datetime.fromisoformat(start_ts).replace(tzinfo=ZoneInfo("Asia/Tokyo"))
        forecast_entries = synthesize_osaka_forecast(start_dt, hours=len(entries))
        save_json(FORECAST_PATH, forecast_to_json(forecast_entries))

    forecast_data = load_json(FORECAST_PATH)
    return forecast_data


def main() -> None:
    if not REAL_WEATHER_PATH.exists():
        REAL_WEATHER_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("[INFO] real_weather.json missing; fetching latest observations...")
        try:
            save_json(REAL_WEATHER_PATH, fetch_real_weather())
        except RequestException as exc:
            print(f"[WARN] Failed to fetch real weather ({exc}); generating synthetic observations for training fallback.")
            tz = ZoneInfo("Asia/Tokyo")
            start_dt = datetime.datetime.now(tz).replace(minute=0, second=0, microsecond=0)
            synthetic_entries = synthesize_osaka_forecast(start_dt, hours=24)
            synthetic_real = {
                "source": "synthetic-fallback",
                "latitude": 34.6937,
                "longitude": 135.5023,
                "timezone": "Asia/Tokyo",
                "retrieved_at": start_dt.isoformat(),
                "entries": synthetic_entries,
            }
            save_json(REAL_WEATHER_PATH, synthetic_real)

    real_data = load_json(REAL_WEATHER_PATH)
    forecast_data = ensure_forecast_alignment(real_data)

    forecast_entries = {entry["time"]: entry for entry in forecast_data.get("entries", [])}
    real_entries = {entry["time"]: entry for entry in real_data.get("entries", [])}

    common_times = sorted(set(forecast_entries) & set(real_entries))
    if not common_times:
        # Regenerate a forecast aligned to real data timestamps and reload
        entries = real_data.get("entries", [])
        if not entries:
            raise ValueError("real_weather.json has no entries to align with")
        start_dt = datetime.datetime.fromisoformat(entries[0]["time"]).replace(
            tzinfo=ZoneInfo("Asia/Tokyo")
        )
        forecast_entries_new = synthesize_osaka_forecast(start_dt, hours=len(entries))
        save_json(FORECAST_PATH, forecast_to_json(forecast_entries_new))
        forecast_entries = {entry["time"]: entry for entry in forecast_entries_new}
        common_times = sorted(set(forecast_entries) & set(real_entries))
        if not common_times:
            raise ValueError("No overlapping timestamps between forecast and real data")

    temp_diffs: List[float] = []
    rain_diffs: List[float] = []
    forecast_rain: List[float] = []
    real_rain: List[float] = []

    for ts in common_times:
        f_entry = forecast_entries[ts]
        r_entry = real_entries[ts]
        temp_diffs.append(r_entry.get("temperature", 0.0) - f_entry.get("temperature", 0.0))
        rain_diffs.append(
            r_entry.get("precipitation_probability", 0.0)
            - f_entry.get("precipitation_probability", 0.0)
        )
        forecast_rain.append(f_entry.get("precipitation_probability", 0.0))
        real_rain.append(r_entry.get("precipitation_probability", 0.0))

    params = load_model_params()

    temp_bias_update = LEARNING_RATE * mean(temp_diffs)
    rain_bias_update = LEARNING_RATE * mean(rain_diffs)

    forecast_mean_rain = max(mean(forecast_rain), 1.0)
    real_mean_rain = mean(real_rain)
    rain_scale_error = (real_mean_rain / forecast_mean_rain) - 1.0
    cloud_scale_update = params["cloud_to_rain_scale"] * (1 + LEARNING_RATE * rain_scale_error)
    cloud_scale_update = max(0.2, min(3.0, cloud_scale_update))

    params["temp_bias"] += temp_bias_update
    params["rain_bias"] += rain_bias_update
    params["cloud_to_rain_scale"] = cloud_scale_update

    save_json(MODEL_PATH, params)

    temp_error_metric = mape_style_error(
        [forecast_entries[t]["temperature"] for t in common_times],
        [real_entries[t]["temperature"] for t in common_times],
    )
    rain_error_metric = mape_style_error(
        [forecast_entries[t]["precipitation_probability"] for t in common_times],
        [real_entries[t]["precipitation_probability"] for t in common_times],
    )

    print("Updated model parameters:")
    print(json.dumps(params, indent=2))
    print(f"Temperature MAPE-style error: {temp_error_metric:.3f}")
    print(f"Rain MAPE-style error: {rain_error_metric:.3f}")


if __name__ == "__main__":
    main()
