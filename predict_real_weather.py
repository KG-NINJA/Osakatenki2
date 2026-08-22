# predict_real_weather.py
# Osaka actual weather fetcher (Open-Meteo API)

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


def as_jst_iso(value: str) -> str:
    """Open-Meteoのローカル時刻へJSTオフセットを明示する。"""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=JST)
    else:
        parsed = parsed.astimezone(JST)
    return parsed.isoformat(timespec="minutes")


def fetch_real_weather():
    """Open-Meteo の最近の観測値を1時間ごとに取得する。"""
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,precipitation_probability,weather_code"
        "&past_days=1"
        "&timezone=Asia%2FTokyo"
    )

    print("Fetching real weather...")
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

    real = {
        "time": [as_jst_iso(value) for value in hourly["time"]],
        "temp": hourly["temperature_2m"],
        "rain": hourly["precipitation_probability"],
        "code": hourly["weather_code"],
    }

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(real, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {OUTPUT_FILE}")
    return real


if __name__ == "__main__":
    fetch_real_weather()
