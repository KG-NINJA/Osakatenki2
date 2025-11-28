"""
trainer.py
Self-Learning Osaka Weather AI
-------------------------------------
Train model parameters based on:
- real_weather.json (yesterday actual)
- today_forecast.json (AI prediction)
-------------------------------------
"""

import json
import numpy as np
import os


DATA_REAL = "data/real_weather.json"
DATA_MODEL = "data/today_forecast.json"
MODEL_PARAM = "data/model_param.json"


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs("data", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    print("== Trainer start ==")

    real = load_json(DATA_REAL)
    pred = load_json(DATA_MODEL)

    if real is None:
        raise FileNotFoundError("real_weather.json must exist for training")

    # 日次学習は24時間分データが揃っている場合にのみ行う
    if len(real["temp"]) < 12 or len(pred["temp"]) < 12:  # 初日はデータが少ないためスキップ
        print("[SKIP] not enough data for training yet")
        print("Next learning will run when 24 hrs of real weather stored.")
        return

    L = min(len(real["temp"]), len(pred["temp"]))
    real_temp = np.array(real["temp"][:L])
    pred_temp = np.array(pred["temp"][:L])

    # 誤差から係数調整
    diff = real_temp - pred_temp
    correction = float(np.mean(diff))  # 単純平均補正

    # 既存パラメータをロード
    params = load_json(MODEL_PARAM) or {"temp_bias": 0.0}
    params["temp_bias"] += correction  # 強化学習（小さな改善を積む）

    print(f"[UPDATE] temp_bias += {correction:.3f} (new: {params['temp_bias']:.3f})")

    save_json(MODEL_PARAM, params)
    print("[OK] model_param.json updated")


if __name__ == "__main__":
    main()
