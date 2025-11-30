"""
trainer_improved.py
Self-Learning Osaka Weather AI - Improved Version
"""

import datetime
import json
import logging
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Tuple

import numpy as np
from zoneinfo import ZoneInfo

from osaka_forecast_engine import forecast_to_json, synthesize_osaka_forecast
from predict_real_weather import fetch_real_weather
from requests import RequestException

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("data/model_param.json")
FORECAST_PATH = Path("data/forecast.json")
REAL_WEATHER_PATH = Path("data/real_weather.json")
LEARNING_RATE = 0.05


def load_json(path: Path) -> dict:
    """JSON ファイルを読み込む"""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path: Path, data: dict) -> None:
    """JSON ファイルに保存"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_model_params() -> Dict[str, float]:
    """モデルパラメータを読み込む"""
    if MODEL_PATH.exists():
        try:
            data = load_json(MODEL_PATH)
            return {
                "temp_bias": float(data.get("temp_bias", 0.0)),
                "rain_bias": float(data.get("rain_bias", 0.0)),
                "cloud_to_rain_scale": float(data.get("cloud_to_rain_scale", 1.0)),
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return {"temp_bias": 0.0, "rain_bias": 0.0, "cloud_to_rain_scale": 1.0}


def normalize_timestamp(ts: str) -> str:
    """タイムスタンプを標準形式に正規化"""
    try:
        if "+" in ts or "Z" in ts:
            ts = ts.split("+")[0].split("Z")[0]
        
        # 秒を削除（時間単位で比較）
        ts = ts.split(":")[0] + ":" + ts.split(":")[1]
        return ts
    except Exception:
        logger.warning(f"Could not normalize timestamp: {ts}")
        return ts


def calculate_mae_rmse_mape(
    forecast_values: List[float],
    actual_values: List[float]
) -> Dict[str, float]:
    """複数の誤差指標を計算"""
    if not forecast_values or len(forecast_values) != len(actual_values):
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "valid_samples": 0}
    
    # MAE
    mae = mean([abs(f - a) for f, a in zip(forecast_values, actual_values)])
    
    # RMSE
    squared_errors = [(f - a) ** 2 for f, a in zip(forecast_values, actual_values)]
    rmse = (mean(squared_errors)) ** 0.5
    
    # MAPE（ゼロ値を除外）
    mape_errors = []
    for f, a in zip(forecast_values, actual_values):
        if abs(a) > 0.5:
            mape_errors.append(abs(f - a) / abs(a))
    
    mape = (mean(mape_errors) * 100) if mape_errors else 0.0
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "valid_samples": len(mape_errors)
    }


def calculate_adaptive_learning_rate(
    errors: List[float],
    base_lr: float = LEARNING_RATE
) -> float:
    """誤差のばらつきに応じて学習率を動的に調整"""
    if len(errors) < 2:
        return base_lr
    
    variance = stdev(errors) if len(errors) > 1 else 0
    mean_abs_error = mean([abs(e) for e in errors])
    
    if mean_abs_error < 0.1:
        adaptive_lr = base_lr * 1.5
    elif mean_abs_error > 1.0:
        adaptive_lr = base_lr * 0.3
    else:
        adaptive_lr = base_lr
    
    # 分散が大きい → さらに下げる
    if variance > mean_abs_error:
        adaptive_lr *= 0.7
    
    return max(0.01, min(0.2, adaptive_lr))


def calculate_robust_update(
    diffs: List[float],
    learning_rate: float = LEARNING_RATE,
    method: str = "median"
) -> float:
    """堅牢な学習値を計算"""
    if not diffs:
        return 0.0
    
    if method == "median":
        central_value = median(diffs)
    elif method == "trimmed":
        sorted_diffs = sorted(diffs)
        trim_idx = len(diffs) // 10
        trimmed = sorted_diffs[trim_idx:-trim_idx or None]
        central_value = mean(trimmed) if trimmed else mean(diffs)
    elif method == "weighted":
        weights = np.linspace(0.5, 1.5, len(diffs))
        central_value = np.average(diffs, weights=weights)
    else:
        central_value = mean(diffs)
    
    update = max(-1.0, min(1.0, central_value * learning_rate))
    return update


def update_rain_scale_improved(
    params: Dict[str, float],
    forecast_rain: List[float],
    real_rain: List[float]
) -> Dict[str, float]:
    """改善版：降水スケール補正"""
    
    # ゼロ値を除外
    nonzero_mask = [abs(r) > 1.0 for r in real_rain]
    
    if not any(nonzero_mask):
        logger.info("No significant precipitation; skipping rain scale update")
        return params
    
    forecast_rain_filtered = [
        f for f, keep in zip(forecast_rain, nonzero_mask) if keep
    ]
    real_rain_filtered = [
        r for r, keep in zip(real_rain, nonzero_mask) if keep
    ]
    
    forecast_mean = mean(forecast_rain_filtered) if forecast_rain_filtered else 1.0
    real_mean = mean(real_rain_filtered) if real_rain_filtered else 1.0
    
    if forecast_mean < 0.1:
        logger.info("Forecast rain too low; skipping scale update")
        return params
    
    scale_ratio = real_mean / forecast_mean
    
    # 適応的学習率
    rain_diffs = [f - r for f, r in zip(forecast_rain_filtered, real_rain_filtered)]
    adaptive_lr = calculate_adaptive_learning_rate(rain_diffs)
    
    # ログスケールで補正
    log_ratio = math.log(max(scale_ratio, 0.1))
    scale_update = math.exp(adaptive_lr * log_ratio)
    
    new_scale = params["cloud_to_rain_scale"] * scale_update
    new_scale = max(0.2, min(3.0, new_scale))
    
    params["cloud_to_rain_scale"] = new_scale
    
    logger.info(f"Rain scale updated: {new_scale:.3f} (ratio: {scale_ratio:.3f})")
    
    return params


def ensure_forecast_alignment_improved(
    real_data: dict
) -> Tuple[dict, dict, dict, List[str]]:
    """改善版：タイムスタンプの確実な整合性確保"""
    
    if not FORECAST_PATH.exists():
        entries = real_data.get("entries", [])
        if not entries:
            raise ValueError("real_weather.json has no entries to align with")
        
        start_ts = entries[0]["time"]
        start_dt = datetime.datetime.fromisoformat(
            normalize_timestamp(start_ts)
        ).replace(tzinfo=ZoneInfo("Asia/Tokyo"))
        
        logger.info(f"Forecast not found; generating for {len(entries)} hours")
        forecast_entries_new = synthesize_osaka_forecast(start_dt, hours=len(entries))
        save_json(FORECAST_PATH, forecast_to_json(forecast_entries_new))
    
    forecast_data = load_json(FORECAST_PATH)
    
    # 正規化キーで再マッピング
    forecast_entries = {
        normalize_timestamp(entry["time"]): entry
        for entry in forecast_data.get("entries", [])
    }
    real_entries = {
        normalize_timestamp(entry["time"]): entry
        for entry in real_data.get("entries", [])
    }
    
    common_times = sorted(set(forecast_entries) & set(real_entries))
    
    if not common_times:
        logger.error(
            f"No overlapping timestamps!\n"
            f"  Forecast: {len(forecast_entries)} entries\n"
            f"  Real: {len(real_entries)} entries\n"
            f"  Sample forecast: {list(forecast_entries.keys())[:3]}\n"
            f"  Sample real: {list(real_entries.keys())[:3]}"
        )
        raise ValueError("Timestamp mismatch")
    
    logger.info(f"Found {len(common_times)} overlapping timestamps")
    return forecast_data, forecast_entries, real_entries, common_times


def main() -> None:
    """メイン処理"""
    logger.info("=== Trainer Start ===")
    
    # 実測データを取得
    if not REAL_WEATHER_PATH.exists():
        REAL_WEATHER_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Fetching real weather data...")
        try:
            real_weather = fetch_real_weather()
            save_json(REAL_WEATHER_PATH, real_weather)
        except RequestException as exc:
            logger.warning(f"Failed to fetch real weather: {exc}")
            logger.info("Using synthetic fallback")
            tz = ZoneInfo("Asia/Tokyo")
            start_dt = datetime.datetime.now(tz).replace(minute=0, second=0, microsecond=0)
            synthetic_entries = synthesize_osaka_forecast(start_dt, hours=24)
            synthetic_real = {
                "source": "synthetic-fallback",
                "latitude": 34.6937,
                "longitude": 135.5023,
                "timezone": "Asia/Tokyo",
                "retrieved_at": start_dt.isoformat(),
                "entries": synthetic_entries,
            }
            save_json(REAL_WEATHER_PATH, synthetic_real)
    
    real_data = load_json(REAL_WEATHER_PATH)
    logger.info(f"Real weather loaded: {len(real_data.get('entries', []))} entries")
    
    # 予報データとの整合性を確保
    forecast_data, forecast_entries, real_entries, common_times = ensure_forecast_alignment_improved(real_data)
    
    # 誤差を計算
    temp_diffs: List[float] = []
    rain_diffs: List[float] = []
    forecast_rain: List[float] = []
    real_rain: List[float] = []
    forecast_temps: List[float] = []
    real_temps: List[float] = []
    
    for ts in common_times:
        f_entry = forecast_entries[ts]
        r_entry = real_entries[ts]
        
        f_temp = f_entry.get("temperature", 0.0)
        r_temp = r_entry.get("temperature", 0.0)
        f_rain = f_entry.get("precipitation_probability", 0.0)
        r_rain = r_entry.get("precipitation_probability", 0.0)
        
        temp_diffs.append(r_temp - f_temp)
        rain_diffs.append(r_rain - f_rain)
        forecast_rain.append(f_rain)
        real_rain.append(r_rain)
        forecast_temps.append(f_temp)
        real_temps.append(r_temp)
    
    # パラメータを更新
    params = load_model_params()
    
    # 気温補正（堅牢な方法）
    temp_update = calculate_robust_update(temp_diffs, LEARNING_RATE, method="median")
    params["temp_bias"] += temp_update
    logger.info(f"Temperature bias update: +{temp_update:.6f} -> {params['temp_bias']:.6f}")
    
    # 降水補正（堅牢な方法）
    rain_update = calculate_robust_update(rain_diffs, LEARNING_RATE, method="trimmed")
    params["rain_bias"] += rain_update
    logger.info(f"Rain bias update: +{rain_update:.6f} -> {params['rain_bias']:.6f}")
    
    # 降水スケール補正（改善版）
    params = update_rain_scale_improved(params, forecast_rain, real_rain)
    
    # パラメータを保存
    save_json(MODEL_PATH, params)
    logger.info("Model parameters saved")
    
    # 精度評価
    temp_metrics = calculate_mae_rmse_mape(forecast_temps, real_temps)
    rain_metrics = calculate_mae_rmse_mape(forecast_rain, real_rain)
    
    logger.info("=" * 50)
    logger.info("Temperature Metrics:")
    logger.info(f"  MAE:  {temp_metrics['mae']:.4f}°C")
    logger.info(f"  RMSE: {temp_metrics['rmse']:.4f}°C")
    logger.info(f"  MAPE: {temp_metrics['mape']:.2f}%")
    logger.info("Precipitation Metrics:")
    logger.info(f"  MAE:  {rain_metrics['mae']:.2f}%")
    logger.info(f"  RMSE: {rain_metrics['rmse']:.2f}%")
    logger.info(f"  MAPE: {rain_metrics['mape']:.2f}%")
    logger.info(f"  Valid samples: {rain_metrics['valid_samples']}")
    logger.info("=" * 50)
    
    logger.info("=== Trainer Complete ===")


if __name__ == "__main__":
    main()
