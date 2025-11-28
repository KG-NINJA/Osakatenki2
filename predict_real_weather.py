# predict_real_weather.py
# 実測を run_forecast.py が読める形式に変換して保存する

import csv
import json
import datetime

INPUT_FILE = "data/real_raw.csv"     # あなたが保存している raw データ
OUTPUT_FILE = "data/real_weather.json"

def convert():
    times = []
    temps = []
    rain = []
    codes = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(row["Time"])
            temps.append(float(row["Temperature"]))
            rain.append(int(row["Precipitation Probability"]))

            # Weather → code変換
            w = row["Weather"].lower()
            if "clear" in w or "sunny" in w:
                code = 0
            elif "cloud" in w:
                code = 3
            elif "drizzle" in w:
                code = 51
            elif "rain" in w:
                code = 61
            elif "heavy" in w:
                code = 80
            elif "storm" in w or "thunder" in w:
                code = 95
            else:
                code = 3
            codes.append(code)

    real = {
        "time": times,
        "temp": temps,
        "rain": rain,
        "code": codes,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(real, f, ensure_ascii=False, indent=2)

    print(f"[OK] Real weather saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    convert()
