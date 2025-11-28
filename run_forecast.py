"""
run_forecast.py
-----------------------------------------
Daily HTML renderer for:
- Yesterday real weather (actual logs)
- Today's AI forecast (model output)
- Error comparison table
- MAPE / MAE accuracy scores
- Growth visualization (error trend plot)
-----------------------------------------
"""

import json
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from osaka_forecast_engine import (
    synthesize_osaka_forecast,
    render_forecast_html,
    forecast_to_json,
    write_html,
    write_json,
)

SITE_DIR = "site"
DATA_REAL = "data/real_weather.json"
DATA_MODEL = "data/today_forecast.json"
DATA_ERROR_LOG = "data/error_history.json"


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# 修正: prediction.json の形式変換
# -----------------------------
def load_prediction(path=DATA_MODEL):
    if not os.path.exists(path):
        raise FileNotFoundError("Prediction file not found")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # open-meteoの場合 entries[] 構造を flatten する
    if "entries" in raw:
        pred = {
            "time": [e["time"] for e in raw["entries"]],
            "temp": [e["temperature"] for e in raw["entries"]],
            "rain": [e["precipitation_probability"] for e in raw["entries"]],
        }
        return pred

    # 既に整形済み
    return raw


# -----------------------------
# 1. 今日の予報を生成
# -----------------------------
def generate_forecast():
    now = datetime.datetime.now()
    start = now.replace(minute=0, second=0, microsecond=0)
    forecast = synthesize_osaka_forecast(start, hours=24)

    data = forecast_to_json(forecast)
    write_json(DATA_MODEL, data)  # そのまま保存
    return forecast, load_prediction(DATA_MODEL)  # 再読み込みして構造統一


# -----------------------------
# 2. 実測データの読み込み
# -----------------------------
def load_real_weather():
    return load_json(DATA_REAL, default=None)


# -----------------------------
# 3. 誤差計算
# -----------------------------
def compute_error(real, forecast_json):
    if real is None:
        return None, None, None

    real_temp = np.array(real["temp"])
    pred_temp = np.array(forecast_json["temp"])

    L = min(len(real_temp), len(pred_temp))
    real_temp = real_temp[:L]
    pred_temp = pred_temp[:L]

    mae = float(np.mean(np.abs(real_temp - pred_temp)))
    mape = float(np.mean(np.abs(real_temp - pred_temp) / np.maximum(real_temp, 1)) * 100)
    errors = list(np.abs(real_temp - pred_temp))

    return mae, mape, errors


# -----------------------------
# 4. 誤差履歴保存
# -----------------------------
def update_error_history(mape):
    history = load_json(DATA_ERROR_LOG, default=[])
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    history.append({"date": today, "mape": mape})

    os.makedirs("data", exist_ok=True)
    with open(DATA_ERROR_LOG, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return history


# -----------------------------
# 5. 成長グラフ
# -----------------------------
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
    return "growth.png"


# -----------------------------
# 6. HTML生成
# -----------------------------
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

    growth_img = f"<img src='{growth_path}' width='600'>" if growth_path else "(初日のためデータなし)"

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

<h2>昨日の実測データ</h2>
<table>
<tr><th>時間</th><th>気温</th><th>降水</th></tr>
{real_table}
</table>

<h2>誤差（実測 vs 予測）</h2>
<p>MAE: {mae:.3f}　/　MAPE: {mape:.2f}%</p>
<table><tr><th>温度誤差 (°C)</th></tr>{error_table}</table>

<h2>AI 成長グラフ（MAPE履歴）</h2>
{growth_img}

</body></html>
"""

    write_html(f"{SITE_DIR}/index.html", html)


# -----------------------------
# Main
# -----------------------------
def main():
    print("== Loading real weather ==")
    real = load_real_weather()

    print("== Generating forecast ==")
    forecast, pred_json = generate_forecast()

    print("== Computing error ==")
    mae, mape, errors = compute_error(real, pred_json)

    print("== Saving history ==")
    history = update_error_history(mape if mape is not None else 0)

    print("== Plotting growth ==")
    growth_path = plot_growth(history)

    print("== Rendering page ==")
    html = render_forecast_html(
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        forecast=forecast,
        title="大阪 天気予報",
        subtitle="Self-Learning AI Model"
    )

    render_full_page(html, real, pred_json, mae, mape, errors, growth_path)
    print("Done.")


if __name__ == "__main__":
    main()
