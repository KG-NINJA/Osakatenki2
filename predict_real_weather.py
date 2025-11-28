import json
from pathlib import Path

import requests

API_URL = "https://api.open-meteo.com/v1/forecast"
PARAMS = {
    "latitude": 34.6937,
    "longitude": 135.5023,
    "hourly": "temperature_2m,precipitation_probability,weathercode",
    "timezone": "Asia/Tokyo",
}


def fetch_real_weather() -> dict:
    response = requests.get(API_URL, params=PARAMS, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])[:24]
    temperatures = hourly.get("temperature_2m", [])[:24]
    precip_probs = hourly.get("precipitation_probability", [])[:24]
    weather_codes = hourly.get("weathercode", [])[:24]

    condensed = [
        {
            "time": t,
            "temperature": temp,
            "precipitation_probability": precip,
            "weathercode": code,
        }
        for t, temp, precip, code in zip(times, temperatures, precip_probs, weather_codes)
    ]

    return {
        "source": "open-meteo",
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": data.get("timezone"),
        "retrieved_at": data.get("generationtime_ms"),
        "entries": condensed,
    }


def main() -> None:
    Path("data").mkdir(exist_ok=True)
    real_weather = fetch_real_weather()
    with Path("data/real_weather.json").open("w", encoding="utf-8") as fp:
        json.dump(real_weather, fp, ensure_ascii=False, indent=2)
    print("[OK] Saved real Osaka weather to data/real_weather.json")


if __name__ == "__main__":
    main()
