# run_osaka_forecast.py
import datetime
from zoneinfo import ZoneInfo
from osaka_forecast_engine import (
    synthesize_osaka_forecast,
    render_forecast_html,
    forecast_to_json,
    write_html,
    write_json,
)

def main():
    now = datetime.datetime.now(ZoneInfo("Asia/Tokyo")).replace(minute=0, second=0, microsecond=0)

    forecast = synthesize_osaka_forecast(now, hours=24)

    html = render_forecast_html(
        generated_at=now.strftime("%Y-%m-%d %H:%M"),
        forecast=forecast,
        title="大阪 天気予報",
        subtitle="24時間の簡易予測"
    )

    write_html("site/index.html", html)
    write_json("site/forecast.json", forecast_to_json(forecast))

    print("[OK] Osaka forecast generated")

if __name__ == "__main__":
    main()
