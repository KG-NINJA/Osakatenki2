import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from osaka_forecast_engine import (
    forecast_to_json,
    render_forecast_html,
    synthesize_osaka_forecast,
    write_html,
    write_json,
)


def main() -> None:
    base_time = datetime.datetime.now(ZoneInfo("Asia/Tokyo")).replace(
        minute=0, second=0, microsecond=0
    )

    forecast = synthesize_osaka_forecast(base_time, hours=24)

    generated_at = base_time.isoformat(timespec="minutes")
    html = render_forecast_html(
        generated_at=generated_at,
        forecast=forecast,
        title="大阪 天気予報 AI",
        subtitle="24時間の生成予報 (Self-learning)",
    )

    Path("data").mkdir(exist_ok=True)
    Path("site").mkdir(exist_ok=True)

    write_json("data/forecast.json", forecast_to_json(forecast))
    write_html("site/index.html", html)

    print("[OK] Forecast saved to data/forecast.json and site/index.html")


if __name__ == "__main__":
    main()
