from pathlib import Path

import pytest
import requests

import predict_real_weather
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

    result = predict_real_weather.fetch_real_weather()

    assert "weather_code" in called["url"]
    assert "weathercode" not in called["url"]
    assert called["timeout"] == 20
    assert result["code"] == [2]


def test_open_meteo_http_error_is_not_masked_as_missing_hourly(monkeypatch):
    def fake_get(url, timeout):
        return FakeResponse({}, requests.HTTPError("400 Client Error"))

    monkeypatch.setattr(predict_real_weather.requests, "get", fake_get)

    with pytest.raises(requests.HTTPError, match="400 Client Error"):
        predict_real_weather.fetch_real_weather()


def test_trainer_regenerates_when_stale_forecast_file_is_removed(monkeypatch, tmp_path):
    forecast_path = tmp_path / "forecast.json"
    monkeypatch.setattr(trainer, "FORECAST_PATH", forecast_path)

    calls = []
    real_data = {
        "entries": [
            {
                "time": "2026-08-22T09:00",
                "temperature": 30.0,
                "precipitation_probability": 10.0,
                "weathercode": 1,
            }
        ]
    }

    def fake_regenerate(data):
        calls.append(data)
        return {
            "entries": [
                {
                    "time": "2026-08-22T09:00",
                    "temperature": 29.0,
                    "precipitation_probability": 20.0,
                }
            ]
        }

    monkeypatch.setattr(trainer, "regenerate_forecast", fake_regenerate)

    _, _, _, common_times = trainer.ensure_forecast_alignment_improved(real_data)

    assert len(calls) == 1
    assert common_times == ["2026-08-22T09:00"]


def test_scheduled_workflow_scopes_jobs_and_never_force_pushes():
    workflow = Path(".github/workflows/weather.yml").read_text(encoding="utf-8")

    assert "github.event.schedule == '0 0 * * *'" in workflow
    assert "github.event.schedule == '0 1 * * *'" in workflow
    assert "github.event.schedule == '0 2 * * *'" in workflow
    assert "rm -f data/forecast.json" in workflow
    assert "push_options" not in workflow
    assert "--force" not in workflow
