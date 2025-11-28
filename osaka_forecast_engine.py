import datetime
import json
import random
from pathlib import Path
from typing import List, Dict

# Seasonal anchors for Osaka (approximate climatology)
MONTHLY_TEMP_PROFILE = {
    1: {"avg_low": 2.5, "avg_high": 9.5, "precip_base": 20, "cloud_mean": 0.55},
    2: {"avg_low": 3.0, "avg_high": 11.0, "precip_base": 25, "cloud_mean": 0.50},
    3: {"avg_low": 5.5, "avg_high": 14.5, "precip_base": 30, "cloud_mean": 0.48},
    4: {"avg_low": 10.5, "avg_high": 20.0, "precip_base": 35, "cloud_mean": 0.42},
    5: {"avg_low": 15.5, "avg_high": 24.5, "precip_base": 40, "cloud_mean": 0.40},
    6: {"avg_low": 20.5, "avg_high": 27.5, "precip_base": 55, "cloud_mean": 0.65},
    7: {"avg_low": 24.5, "avg_high": 32.5, "precip_base": 50, "cloud_mean": 0.60},
    8: {"avg_low": 26.0, "avg_high": 33.5, "precip_base": 45, "cloud_mean": 0.58},
    9: {"avg_low": 22.5, "avg_high": 29.5, "precip_base": 55, "cloud_mean": 0.60},
    10: {"avg_low": 16.5, "avg_high": 24.0, "precip_base": 45, "cloud_mean": 0.50},
    11: {"avg_low": 10.0, "avg_high": 18.0, "precip_base": 35, "cloud_mean": 0.52},
    12: {"avg_low": 5.0, "avg_high": 12.5, "precip_base": 25, "cloud_mean": 0.55},
}

DIURNAL_CURVE = {
    0: 0.18, 1: 0.17, 2: 0.16, 3: 0.15, 4: 0.15, 5: 0.17,
    6: 0.23, 7: 0.33, 8: 0.45, 9: 0.57, 10: 0.70, 11: 0.82,
    12: 0.90, 13: 0.96, 14: 1.00, 15: 0.97, 16: 0.90, 17: 0.78,
    18: 0.62, 19: 0.50, 20: 0.38, 21: 0.30, 22: 0.24, 23: 0.20,
}

WEATHER_CODE_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    51: "Light drizzle",
    61: "Rain",
    80: "Rain showers",
    95: "Thunderstorm",
}

DEFAULT_MODEL_PARAMS = {
    "temp_bias": 0.0,
    "rain_bias": 0.0,
    "cloud_to_rain_scale": 1.0,
}


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def load_model_params(path: Path | str = Path("data/model_param.json")) -> Dict[str, float]:
    path = Path(path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
                return {
                    "temp_bias": float(data.get("temp_bias", 0.0)),
                    "rain_bias": float(data.get("rain_bias", 0.0)),
                    "cloud_to_rain_scale": float(data.get("cloud_to_rain_scale", 1.0)),
                }
        except (json.JSONDecodeError, OSError, TypeError):
            return DEFAULT_MODEL_PARAMS.copy()
    return DEFAULT_MODEL_PARAMS.copy()


def synthesize_osaka_forecast(start: datetime.datetime, hours: int = 24) -> List[Dict[str, object]]:
    """Generate a deterministic Osaka forecast sequence.

    Determinism comes from seeding the random generator with the date (YYYYMMDD).
    A lightweight seasonal model captures average Osaka temperature/precipitation,
    and the stored parameters (trained daily) add small biases.
    """

    params = load_model_params()
    rng = random.Random(int(start.strftime("%Y%m%d")))

    forecast: List[Dict[str, object]] = []

    for step in range(hours):
        current = start + datetime.timedelta(hours=step)
        month_profile = MONTHLY_TEMP_PROFILE[current.month]

        base_low = month_profile["avg_low"] + rng.uniform(-1.2, 1.2)
        base_high = month_profile["avg_high"] + rng.uniform(-1.0, 1.0)
        diurnal_factor = DIURNAL_CURVE[current.hour]

        cloud_factor = clamp(
            month_profile["cloud_mean"] + rng.uniform(-0.25, 0.35), 0.0, 1.0
        )

        precip_prob = month_profile["precip_base"]
        precip_prob += cloud_factor * 35.0 * params.get("cloud_to_rain_scale", 1.0)
        precip_prob += rng.uniform(-8, 8)
        precip_prob += params.get("rain_bias", 0.0)
        precip_prob = clamp(precip_prob, 0.0, 100.0)

        if precip_prob > 85:
            weather_code = 95 if rng.random() < 0.2 else 80
        elif precip_prob > 65:
            weather_code = 61
        elif precip_prob > 45:
            weather_code = 51
        elif cloud_factor > 0.6:
            weather_code = 3
        elif cloud_factor > 0.35:
            weather_code = 2
        elif cloud_factor > 0.2:
            weather_code = 1
        else:
            weather_code = 0

        temp_span = base_high - base_low
        temp = base_low + temp_span * diurnal_factor
        temp -= (precip_prob / 100.0) * 1.2
        temp += rng.uniform(-0.6, 0.6)
        temp += params.get("temp_bias", 0.0)
        temp = round(temp, 1)

        forecast.append(
            {
                "time": current.strftime("%Y-%m-%dT%H:%M"),
                "temperature": temp,
                "weathercode": weather_code,
                "precipitation_probability": int(round(precip_prob)),
            }
        )

    return forecast


def render_forecast_html(
    generated_at: str, forecast: List[Dict[str, object]], title: str, subtitle: str
) -> str:
    """Render a minimal HTML table for GitHub Pages output."""

    rows = []
    for entry in forecast:
        code = int(entry.get("weathercode", 0))
        desc = WEATHER_CODE_LABELS.get(code, "Unknown")
        rows.append(
            f"<tr><td>{entry['time']}</td><td>{entry['temperature']}°C"  # type: ignore[index]
            f"</td><td>{entry['precipitation_probability']}%"  # type: ignore[index]
            f"</td><td>{desc}</td></tr>"
        )

    style = """
    <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f6f9fc; color: #0f172a; }
    header { margin-bottom: 16px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #cbd5e1; padding: 8px; text-align: left; }
    th { background: #e2e8f0; }
    tr:nth-child(even) { background: #f8fafc; }
    .subtitle { color: #475569; }
    </style>
    """

    html = (
        "<html><head><meta charset='utf-8'><title>"
        f"{title}</title>{style}</head><body>"
        f"<header><h1>{title}</h1><div class='subtitle'>{subtitle}</div>"
        f"<div>Generated at: {generated_at}</div></header>"
        "<table><thead><tr><th>Time</th><th>Temperature</th><th>Precipitation Probability</th><th>Weather</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )
    return html


def forecast_to_json(forecast: List[Dict[str, object]]) -> Dict[str, object]:
    """Wrap the forecast entries with metadata for storage."""

    generated_at = datetime.datetime.now().isoformat(timespec="minutes")
    return {
        "generated_at": generated_at,
        "model_parameters": load_model_params(),
        "entries": forecast,
    }


def write_html(path: str | Path, html: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_json(path: str | Path, data: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


__all__ = [
    "synthesize_osaka_forecast",
    "render_forecast_html",
    "forecast_to_json",
    "write_html",
    "write_json",
]
