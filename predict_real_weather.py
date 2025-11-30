# predict_real_weather.py
# Osaka actual weather fetcher (Open-Meteo API)

import requests
import json
from datetime import datetime, timedelta, timezone
import os

OUTPUT_FILE = "data/real_weather.json"

# Osaka 座標
LAT = 34.6937
LON = 135.5023

def fetch_real_weather():
    """
    Open-Meteo の "最近の観測値" を1時間ごとに取得
    GitHub Actions 上で安定して動きます
    """
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,precipitation_probability,weathercode"
        "&past_days=1"
        "&timezone=Asia%2FTokyo"
    )

    print("Fetching real weather...")
    res = requests.get(url)
    data = res.json()

    hourly = data["hourly"]

    real = {
        "time": hourly["time"],
        "temp": hourly["temperature_2m"],
        "rain": hourly["precipitation_probability"],
        "code": hourly["weathercode"]
    }

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(real, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_real_weather()
