"""
run_forecast.py
-----------------------------------------
Daily HTML renderer for:
- Completed recent weather
- Today's AI forecast
- Prior-forecast error comparison
- MAPE / MAE accuracy scores
- Growth visualization
-----------------------------------------
"""

import datetime
import json
import os
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np

from osaka_forecast_engine import (
    forecast_to_json,
    render_forecast_html,
    synthesize_osaka_forecast,
    write_html,
    write_json,
)

SITE_DIR = "site"
DATA_REAL = "data/real_weather.json"
DATA_MODEL = "data/today_forecast.json"
DATA_TRAINING_FORECAST = "data/forecast.json"
DATA_ERROR_LOG = "data/error_history.json"
JST = ZoneInfo("Asia/Tokyo")


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prediction(path=DATA_MODEL):
    if not os.path.exists(path):
        raise FileNotFoundError("Prediction file not found")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "entries" in raw:
        return {
            "time": [e["time"] for e in raw["entries"]],
            "temp": [e["temperature"] for e in raw["entries"]],
            "rain": [e["precipitation_probability"] for e in raw["entries"]],
        }

    return raw


def normalize_hour(value: str) -> str:
    """時刻をJSTの時間単位キーへ正規化する。"""
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=JST)
    else:
        parsed = parsed.astimezone(JST)
    return parsed.replace(minute=0, second=0, microsecond=0).isoformat()


def generate_forecast():
    """現在JSTから24時間予報を生成し、次回学習用にも保存する。"""
    now = datetime.datetime.now(JST)
    start = now.replace(minute=0, second=0, microsecond=0)
    forecast = synthesize_osaka_forecast(start, hours=24)

    data = forecast_to_json(forecast)
    write_json(DATA_MODEL, data)
    # この予報は次回、実時間が完了してからだけsafe_train.pyが学習に使用する。
    write_json(DATA_TRAINING_FORECAST, data)
    return forecast, load_prediction(DATA_MODEL)


def load_real_weather():
    return load_json(DATA_REAL, default=None)


def compute_error(real, forecast_json):
    """同一JST時刻の完了観測と、その時刻より前に発行された予報だけを比較する。"""
    if real is None or forecast_json is None:
        return None, None, None

    real_map = {
        normalize_hour(timestamp): float(temp)
        for timestamp, temp in zip(real.get("time", []), real.get("temp", []))
    }
    forecast_map = {
        normalize_hour(timestamp): float(temp)
        for timestamp, temp in zip(
            forecast_json.get("time", []), forecast_json.get("temp", [])
        )
    }
    common = sorted(set(real_map) & set(forecast_map))
    if not common:
        return None, None, None

    real_temp = np.array([real_map[timestamp] for timestamp in common])
    pred_temp = np.array([forecast_map[timestamp] for timestamp in common])

    mae = float(np.mean(np.abs(real_temp - pred_temp)))
    mape = float(
        np.mean(np.abs(real_temp - pred_temp) / np.maximum(np.abs(real_temp), 1))
        * 100
    )
    errors = list(np.abs(real_temp - pred_temp))
    return mae, mape, errors


def update_error_history(mape):
    """同じ日の再実行では履歴を増殖させず当日値を置換する。"""
    history = load_json(DATA_ERROR_LOG, default=[])
    today = datetime.datetime.now(JST).strftime("%Y-%m-%d")
    record = {"date": today, "mape": mape}

    replaced = False
    for index, item in enumerate(history):
        if item.get("date") == today:
            history[index] = record
            replaced = True
            break
    if not replaced:
        history.append(record)

    os.makedirs("data", exist_ok=True)
    with open(DATA_ERROR_LOG, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return history


def plot_growth(history):
    if len(history) <= 1:
        return None

    dates = [h["date"] for h in history]
    mapes = [h["mape"] for h in history]

    plt.figure(figsize=(8, 4))
    plt.plot(dates, mapes, marker="o")
    plt.title("AI Accuracy Growth (MAPE %)")
    plt.xlabel("Date")
    plt.ylabel("MAPE (%)")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(SITE_DIR, exist_ok=True)
    out_path = f"{SITE_DIR}/growth.png"
    plt.savefig(out_path)
    plt.close()
    return "growth.png"


def render_full_page(html_forecast, real, pred_json, mae, mape, errors, growth_path):
    real_table = ""
    pred_table = ""
    error_table = ""

    if real is not None:
        for t, temp, rain in zip(real["time"], real["temp"], real["rain"]):
            real_table += f"<tr><td>{t[11:16]}</td><td>{temp}</td><td>{rain}%</td></tr>"

    for t, temp, rain in zip(pred_json["time"], pred_json["temp"], pred_json["rain"]):
        pred_table += f"<tr><td>{t[11:16]}</td><td>{temp}</td><td>{rain}%</td></tr>"

    if errors is not None:
        for e in errors:
            error_table += f"<tr><td>{e:.2f}</td></tr>"

    growth_img = (
        f"<img src='{growth_path}' width='600'>" if growth_path else "(評価履歴不足)"
    )
    metrics_text = (
        f"MAE: {mae:.3f}　/　MAPE: {mape:.2f}%"
        if mae is not None and mape is not None
        else "過去に発行した予報と完了観測の対応データがないため、今回は誤差を計算していません。"
    )

    html = f"""
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Self-Learning Osaka Weather AI</title>
<style>
body {{ font-family: 'Noto Sans JP', sans-serif; background:#f0f7ff; padding:20px; }}
table {{ border-collapse: collapse; margin:10px; }}
td,th {{ border:1px solid #ccc; padding:6px; }}
</style>
</head>
<body>

<h1>Self-Learning Osaka Weather AI</h1>

<h2>今日の 24時間 予報</h2>
{html_forecast}

<h2>直近の完了済み気象データ</h2>
<table>
<tr><th>時間</th><th>気温</th><th>降水</th></tr>
{real_table}
</table>

<h2>誤差（完了観測 vs 事前予測）</h2>
<p>{metrics_text}</p>
<table><tr><th>温度誤差 (°C)</th></tr>{error_table}</table>

<h2>AI 成長グラフ（MAPE履歴）</h2>
{growth_img}

</body></html>
"""

    write_html(f"{SITE_DIR}/index.html", html)


def main():
    print("== Loading completed weather ==")
    real = load_real_weather()

    print("== Loading prior forecast for evaluation ==")
    previous_prediction = (
        load_prediction(DATA_TRAINING_FORECAST)
        if os.path.exists(DATA_TRAINING_FORECAST)
        else None
    )

    print("== Computing prior-forecast error ==")
    mae, mape, errors = compute_error(real, previous_prediction)

    print("== Generating new forecast ==")
    forecast, pred_json = generate_forecast()

    print("== Saving evaluation history ==")
    if mape is not None:
        history = update_error_history(mape)
    else:
        history = load_json(DATA_ERROR_LOG, default=[])

    print("== Plotting growth ==")
    growth_path = plot_growth(history)

    print("== Rendering page ==")
    html = render_forecast_html(
        generated_at=datetime.datetime.now(JST).strftime("%Y-%m-%d %H:%M JST"),
        forecast=forecast,
        title="大阪 天気予報",
        subtitle="Self-Learning AI Model",
    )

    render_full_page(html, real, pred_json, mae, mape, errors, growth_path)
    print("Done.")


if __name__ == "__main__":
    main()
