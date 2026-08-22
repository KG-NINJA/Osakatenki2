#!/usr/bin/env python3
"""過去に発行済みの予報と完了済み観測が重なる場合だけ安全に学習する。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import trainer

REAL_PATH = Path("data/real_weather.json")
FORECAST_PATH = Path("data/forecast.json")
MIN_COMMON_HOURS = 12


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_entry_map(entries: list[dict]) -> dict[str, dict]:
    mapped = {}
    for entry in entries:
        timestamp = entry.get("time")
        if not timestamp:
            continue
        mapped[trainer.normalize_timestamp(timestamp)] = entry
    return mapped


def build_training_pair(
    real_path: Path = REAL_PATH,
    forecast_path: Path = FORECAST_PATH,
) -> tuple[list[dict], list[dict], list[str]]:
    if not real_path.exists() or real_path.stat().st_size == 0:
        raise ValueError(f"completed weather data is missing: {real_path}")

    if not forecast_path.exists() or forecast_path.stat().st_size == 0:
        return [], [], []

    real_data = trainer.normalize_real_entries(load_json(real_path))
    forecast_data = load_json(forecast_path)

    real_map = normalized_entry_map(real_data.get("entries", []))
    forecast_map = normalized_entry_map(forecast_data.get("entries", []))
    common_times = sorted(set(real_map) & set(forecast_map))

    filtered_real = [real_map[timestamp] for timestamp in common_times]
    filtered_forecast = [forecast_map[timestamp] for timestamp in common_times]
    return filtered_real, filtered_forecast, common_times


def run_training(
    real_path: Path = REAL_PATH,
    forecast_path: Path = FORECAST_PATH,
) -> bool:
    real_entries, forecast_entries, common_times = build_training_pair(
        real_path, forecast_path
    )

    if len(common_times) < MIN_COMMON_HOURS:
        # 初回移行時や予報アーカイブが古い場合は、観測後に作った合成予報で学習しない。
        print(
            "::warning title=Weather training skipped::"
            f"Prior forecast overlaps only {len(common_times)} completed hours; "
            f"at least {MIN_COMMON_HOURS} are required. Model parameters are unchanged."
        )
        return False

    with tempfile.TemporaryDirectory(prefix="osaka-weather-training-") as tmpdir:
        tmp = Path(tmpdir)
        real_tmp = tmp / "real_weather.json"
        forecast_tmp = tmp / "forecast.json"

        real_tmp.write_text(
            json.dumps({"entries": real_entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        forecast_tmp.write_text(
            json.dumps({"entries": forecast_entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        original_real = trainer.REAL_WEATHER_PATH
        original_forecast = trainer.FORECAST_PATH
        try:
            trainer.REAL_WEATHER_PATH = real_tmp
            trainer.FORECAST_PATH = forecast_tmp
            trainer.main()
        finally:
            trainer.REAL_WEATHER_PATH = original_real
            trainer.FORECAST_PATH = original_forecast

    print(f"[OK] Trained on {len(common_times)} prior-forecast observations")
    return True


def main() -> int:
    run_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
