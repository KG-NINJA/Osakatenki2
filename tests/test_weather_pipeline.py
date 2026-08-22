import json
from datetime import datetime
from pathlib import Path

import pytest
import requests

import predict_real_weather
import run_forecast
import safe_train
import trainer


class FakeResponse:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


def test_open_meteo_uses_current_weather_code_field(monkeypatch, tmp_path):
    called = {}
    payload = {
        "hourly": {
            "time": ["2026-08-22T09:00"],
            "temperature_2m": [30.5],
            "precipitation_probability": [20],
            "weather_code": [2],
        }
    }

    def fake_get(url, timeout):
        called["url"] = url
        called["timeout"] = timeout
        return FakeResponse(payload)

    monkeypatch.setattr(predict_real_weather.requests, "get", fake_get)
    monkeypatch.setattr(
        predict_real_weather, "OUTPUT_FILE", str(tmp_path / "real_weather.json")
    )
    monkeypatch.setattr(
        predict_real_weather,
        "completed_hour_indices",
        lambda times, now=None: list(range(len(times))),
    )

    result = predict_real_weather.fetch_real_weather()

    assert "weather_code" in called["url"]
    assert "weathercode" not in called["url"]
    assert "forecast_days=1" in called["url"]
    assert called["timeout"] == 20
    assert result["time"] == ["2026-08-22T09:00+09:00"]
    assert result["code"] == [2]


def test_completed_hour_filter_excludes_current_and_future_hours():
    now = datetime(2026, 8, 22, 10, 30, tzinfo=predict_real_weather.JST)
    times = [
        "2026-08-22T08:00",
        "2026-08-22T09:00",
        "2026-08-22T10:00",
        "2026-08-22T11:00",
    ]

    assert predict_real_weather.completed_hour_indices(times, now=now) == [0, 1]


def test_open_meteo_http_error_is_not_masked_as_missing_hourly(monkeypatch):
    def fake_get(url, timeout):
        return FakeResponse({}, requests.HTTPError("400 Client Error"))

    monkeypatch.setattr(predict_real_weather.requests, "get", fake_get)

    with pytest.raises(requests.HTTPError, match="400 Client Error"):
        predict_real_weather.fetch_real_weather()


def weather_entry(timestamp: str, temperature: float = 30.0) -> dict:
    return {
        "time": timestamp,
        "temperature": temperature,
        "precipitation_probability": 10.0,
        "weathercode": 1,
    }


def forecast_entry(timestamp: str, temperature: float = 29.0) -> dict:
    return {
        "time": timestamp,
        "temperature": temperature,
        "precipitation_probability": 20.0,
    }


def test_safe_train_skips_when_prior_forecast_does_not_overlap(monkeypatch, tmp_path):
    real_path = tmp_path / "real.json"
    forecast_path = tmp_path / "forecast.json"
    real_path.write_text(
        json.dumps({"entries": [weather_entry("2026-08-22T09:00+09:00")]}),
        encoding="utf-8",
    )
    forecast_path.write_text(
        json.dumps({"entries": [forecast_entry("2026-08-21T09:00+09:00")]}),
        encoding="utf-8",
    )

    called = []
    monkeypatch.setattr(trainer, "main", lambda: called.append(True))

    assert safe_train.run_training(real_path, forecast_path) is False
    assert called == []


def test_safe_train_uses_only_overlapping_prior_forecast(monkeypatch, tmp_path):
    real_path = tmp_path / "real.json"
    forecast_path = tmp_path / "forecast.json"
    timestamps = [f"2026-08-22T{hour:02d}:00+09:00" for hour in range(12)]
    real_path.write_text(
        json.dumps({"entries": [weather_entry(ts) for ts in timestamps]}),
        encoding="utf-8",
    )
    forecast_path.write_text(
        json.dumps({"entries": [forecast_entry(ts) for ts in reversed(timestamps)]}),
        encoding="utf-8",
    )

    inspected = {}

    def fake_main():
        inspected["real"] = json.loads(
            Path(trainer.REAL_WEATHER_PATH).read_text(encoding="utf-8")
        )
        inspected["forecast"] = json.loads(
            Path(trainer.FORECAST_PATH).read_text(encoding="utf-8")
        )

    monkeypatch.setattr(trainer, "main", fake_main)

    assert safe_train.run_training(real_path, forecast_path) is True
    assert len(inspected["real"]["entries"]) == 12
    assert len(inspected["forecast"]["entries"]) == 12
    assert [e["time"] for e in inspected["real"]["entries"]] == timestamps
    assert [e["time"] for e in inspected["forecast"]["entries"]] == timestamps


def test_forecast_error_matches_timestamps_not_array_positions():
    real = {
        "time": ["2026-08-22T09:00+09:00", "2026-08-22T10:00+09:00"],
        "temp": [30.0, 31.0],
    }
    prior = {
        "time": ["2026-08-22T10:00+09:00", "2026-08-22T09:00+09:00"],
        "temp": [30.0, 29.0],
    }

    mae, mape, errors = run_forecast.compute_error(real, prior)

    assert mae == pytest.approx(1.0)
    assert errors == pytest.approx([1.0, 1.0])
    assert mape > 0


def test_scheduled_workflow_runs_one_ordered_pipeline_without_force_push():
    workflow = Path(".github/workflows/weather.yml").read_text(encoding="utf-8")

    assert workflow.count("cron:") == 1
    assert 'cron: "0 0 * * *"' in workflow
    assert workflow.index("python predict_real_weather.py") < workflow.index("python safe_train.py")
    assert workflow.index("python safe_train.py") < workflow.index("python run_forecast.py")
    assert "rm -f data/forecast.json" not in workflow
    assert "push_options" not in workflow
    assert "--force" not in workflow
