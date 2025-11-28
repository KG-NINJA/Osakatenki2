"""
trainer.py
Self-Learning Osaka Weather AI
Safely handles first-run cases / missing data.
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
        print("[SKIP] real_weather.json does not exist yet")
        return

    if pred is None:
        print("[SKIP] today_forecast.json does not exist yet (run_forecast first)")
        return

    if "temp" not in real or "temp" not in pred:
        print("[SKIP] Missing 'temp' key. Data not ready.")
        return

    if len(real["temp"]) < 12 or len(pred["temp"]) < 12:
        print("[SKIP] not enough data for training yet")
        return

    # 誤差ベース学習
    L = min(len(real["temp"]), len(pred["temp"]))
    real_temp = np.array(real["temp"][:L])
    pred_temp = np.array(pred["temp"][:L])
    correction = float(np.mean(real_temp - pred_temp))

    params = load_json(MODEL_PARAM) or {"temp_bias": 0.0}
    params["temp_bias"] += correction * 0.1  # 学習レートを小さく

    print(f"[UPDATE] temp_bias += {correction:.3f} -> {params['temp_bias']:.3f}")
    save_json(MODEL_PARAM, params)

    print("[OK] training complete")


if __name__ == "__main__":
    main()
