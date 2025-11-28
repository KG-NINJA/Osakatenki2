import requests
import json
import os
from datetime import datetime

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
SITE_DIR = os.path.join(BASE, "site")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SITE_DIR, exist_ok=True)

FORECAST_PATH = os.path.join(DATA_DIR, "forecast.json")
REAL_PATH = os.path.join(DATA_DIR, "real_weather.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")

# -------------------------------------------------------------------------
# 1. 予測データ（Open-Meteo）
# -------------------------------------------------------------------------
def fetch_forecast():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=34.6937&longitude=135.5022"
        "&hourly=temperature_2m,precipitation_probability,weathercode"
        "&forecast_days=2&timezone=Asia/Tokyo"
    )
    r = requests.get(url).json()

    hourly = r["hourly"]
    return {
        "time": hourly["time"],
        "temp": hourly["temperature_2m"],
        "rain": hourly["precipitation_probability"],
        "code": hourly["weathercode"],
    }


# -------------------------------------------------------------------------
# 2. 実測データ（Open-Meteo リアルタイム）
# -------------------------------------------------------------------------
def fetch_realtime():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=34.6937&longitude=135.5022"
        "&current=temperature_2m,precipitation,weathercode"
        "&timezone=Asia/Tokyo"
    )
    r = requests.get(url).json()
    current = r["current"]

    # 実測は最新1点だけだが、簡易的に forecast の先頭に合わせて複製
    temp = [current["temperature_2m"]] * 48
    rain = [current.get("precipitation", 0) * 100] * 48

    return {
        "time": r["current_units"]["time"],
        "temp": temp,
        "rain": rain,
        "code": [current["weathercode"]] * 48,
    }


# -------------------------------------------------------------------------
# 3. 精度計算（誤差 / RMSE / MAPE）
# -------------------------------------------------------------------------
def calc_errors(forecast, real):
    import math

    tempF, tempR = forecast["temp"], real["temp"]
    rainF, rainR = forecast["rain"], real["rain"]

    temp_errors = []
    rain_errors = []

    for f, r in zip(tempF, tempR):
        temp_errors.append(abs(f - r))

    for f, r in zip(rainF, rainR):
        rain_errors.append(abs(f - r))

    # RMSE
    rmse_temp = math.sqrt(sum((f - r) ** 2 for f, r in zip(tempF, tempR)) / len(tempF))
    rmse_rain = math.sqrt(sum((f - r) ** 2 for f, r in zip(rainF, rainR)) / len(rainF))

    # MAPE（降水は0割回避）
    def safe_mape(f_list, r_list):
        total = 0
        count = 0
        for f, r in zip(f_list, r_list):
            if r == 0:
                continue
            total += abs((r - f) / r)
            count += 1
        return (total / count) * 100 if count > 0 else 0

    mape_temp = safe_mape(tempF, tempR)
    mape_rain = safe_mape(rainF, rainR)

    return {
        "temp_errors": temp_errors,
        "rain_errors": rain_errors,
        "rmse_temp": rmse_temp,
        "rmse_rain": rmse_rain,
        "mape_temp": mape_temp,
        "mape_rain": mape_rain,
    }


# -------------------------------------------------------------------------
# 4. 履歴ログ保存（AIの成長を可視化）
# -------------------------------------------------------------------------
def update_history(score):
    if not os.path.exists(HISTORY_PATH):
        history = []
    else:
        history = json.load(open(HISTORY_PATH, "r", encoding="utf-8"))

    history.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "rmse_temp": score["rmse_temp"],
        "mape_temp": score["mape_temp"],
        "rmse_rain": score["rmse_rain"],
        "mape_rain": score["mape_rain"],
    })

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# -------------------------------------------------------------------------
# 5. HTML生成：index.html（予報）、compare.html（予測 vs 実測）
# -------------------------------------------------------------------------
def generate_html():
    index = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>大阪天気予報（AIモデル）</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<body>
<h1>大阪天気予報</h1>
<p>AIモデルの予測。毎日 GitHub Actions で自動更新。</p>
<a href="compare.html">予測 vs 実測（答え合わせ）</a><br>
<a href="history.html">AI 成長ログ</a><br>
</body></html>
    """

    with open(os.path.join(SITE_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(index)

    compare = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>予測 vs 実測（答え合わせ）</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<body>
<h1>予測 vs 実測</h1>
<p>誤差表、グラフ、精度スコアを表示します。</p>
<canvas id="chart" height="200"></canvas>
<table id="table"></table>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="compare.js"></script>
</body></html>
    """

    with open(os.path.join(SITE_DIR, "compare.html"), "w", encoding="utf-8") as f:
        f.write(compare)

    history_html = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>AI 成長ログ</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<body>
<h1>AI成長ログ</h1>
<p>毎日の RMSE / MAPE の推移をグラフ化</p>
<canvas id="chart" height="200"></canvas>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="history.js"></script>
</body></html>
    """

    with open(os.path.join(SITE_DIR, "history.html"), "w", encoding="utf-8") as f:
        f.write(history_html)


# -------------------------------------------------------------------------
# 6. compare.js の生成（予測と実測を並列表示）
# -------------------------------------------------------------------------
def generate_compare_js():
    script = """
async function load(path){return await (await fetch(path)).json();}

async function main(){
  const f = await load("../data/forecast.json");
  const r = await load("../data/real_weather.json");

  const labels = f.time.map(t => t.slice(11,16));

  const ctx = document.getElementById("chart");

  new Chart(ctx,{
    type:"line",
    data:{
      labels: labels,
      datasets:[
        {label:"予測 気温", data:f.temp, borderColor:"#ff6b6b"},
        {label:"実測 気温", data:r.temp, borderColor:"#4d79ff"},
        {label:"予測 降水%", data:f.rain, yAxisID:"y1", borderColor:"#ffa94d"},
        {label:"実測 降水%", data:r.rain, yAxisID:"y1", borderColor:"#4dd2c6"},
      ]
    },
    options:{
      scales:{y:{}, y1:{position:"right"}}
    }
  });

  const table=document.getElementById("table");
  table.innerHTML="<tr><th>時刻</th><th>予測気温</th><th>実測気温</th><th>誤差</th></tr>";

  for(let i=0;i<labels.length;i++){
    const tr=`<tr>
      <td>${labels[i]}</td>
      <td>${f.temp[i]}</td>
      <td>${r.temp[i]}</td>
      <td>${(Math.abs(f.temp[i]-r.temp[i])).toFixed(1)}</td>
      </tr>`;
    table.insertAdjacentHTML("beforeend",tr);
  }
}

main();
"""
    with open(os.path.join(SITE_DIR, "compare.js"), "w", encoding="utf-8") as f:
        f.write(script)


# -------------------------------------------------------------------------
# 7. history.js の生成（AIの成長グラフ）
# -------------------------------------------------------------------------
def generate_history_js():
    script = """
async function load(){return await (await fetch("../data/history.json")).json();}

async function main(){
  const h = await load();

  const labels = h.map(x => x.date);
  const rmse = h.map(x => x.rmse_temp);

  new Chart(document.getElementById("chart"),{
    type:"line",
    data:{
      labels: labels,
      datasets:[
        {label:"RMSE（気温誤差）", data:rmse, borderColor:"#ff6b6b"}
      ]
    }
  });
}

main();
"""
    with open(os.path.join(SITE_DIR, "history.js"), "w", encoding="utf-8") as f:
        f.write(script)


# -------------------------------------------------------------------------
# 実行フロー
# -------------------------------------------------------------------------
def main():
    forecast = fetch_forecast()
    real = fetch_realtime()

    with open(FORECAST_PATH, "w", encoding="utf-8") as f:
        json.dump(forecast, f, ensure_ascii=False, indent=2)

    with open(REAL_PATH, "w", encoding="utf-8") as f:
        json.dump(real, f, ensure_ascii=False, indent=2)

    score = calc_errors(forecast, real)
    update_history(score)
    generate_html()
    generate_compare_js()
    generate_history_js()

    print("更新完了：予測・実測・誤差・履歴ログ生成 → HTML更新")

if __name__ == "__main__":
    main()
