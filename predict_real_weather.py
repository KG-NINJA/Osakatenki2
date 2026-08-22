# predict_real_weather.py
# Osaka completed-weather fetcher (Open-Meteo API)

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

OUTPUT_FILE = "data/real_weather.json"

# Osaka 座標
LAT = 34.6937
LON = 135.5023
JST = ZoneInfo("Asia/Tokyo")


def as_jst_datetime(value: str) -> datetime:
    """Open-Meteoの時刻をJST aware datetimeへ変換する。"""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=JST)
    return parsed.astimezone(JST)


def as_jst_iso(value: str) -> str:
    """Open-Meteoのローカル時刻へJSTオフセットを明示する。"""
    return as_jst_datetime(value).isoformat(timespec="minutes")


def completed_hour_indices(times: list[str], now: datetime | None = None) -> list[int]:
    """現在の未完了時間と未来予報を学習入力から除外する。"""
    current = now or datetime.now(JST)
    if current.tzinfo is None:
        current = current.replace(tzinfo=JST)
    else:
        current = current.astimezone(JST)
    boundary = current.replace(minute=0, second=0, microsecond=0)
    return [idx for idx, value in enumerate(times) if as_jst_datetime(value) < boundary]


def fetch_real_weather():
    """Open-Meteoから完了済みの最近の時間帯だけを取得する。"""
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,precipitation_probability,weather_code"
        "&past_days=1"
        "&forecast_days=1"
        "&timezone=Asia%2FTokyo"
    )

    print("Fetching completed weather hours...")
    res = requests.get(url, timeout=20)
    res.raise_for_status()
    data = res.json()

    hourly = data.get("hourly") if isinstance(data, dict) else None
    if not isinstance(hourly, dict):
        reason = data.get("reason") if isinstance(data, dict) else None
        detail = f": {reason}" if reason else ""
        raise RuntimeError(f"Open-Meteo response has no hourly data{detail}")

    required = (
        "time",
        "temperature_2m",
        "precipitation_probability",
        "weather_code",
    )
    missing = [key for key in required if key not in hourly]
    if missing:
        raise RuntimeError(f"Open-Meteo hourly data is missing fields: {missing}")

    lengths = {key: len(hourly[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Open-Meteo hourly arrays have inconsistent lengths: {lengths}")

    indices = completed_hour_indices(hourly["time"])
    if not indices:
        raise RuntimeError("Open-Meteo returned no completed hourly observations")

    real = {
        "time": [as_jst_iso(hourly["time"][idx]) for idx in indices],
        "temp": [hourly["temperature_2m"][idx] for idx in indices],
        "rain": [hourly["precipitation_probability"][idx] for idx in indices],
        "code": [hourly["weather_code"][idx] for idx in indices],
    }

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(real, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {len(indices)} completed hours: {OUTPUT_FILE}")
    return real


if __name__ == "__main__":
    fetch_real_weather()
