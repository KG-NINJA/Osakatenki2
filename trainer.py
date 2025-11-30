"""
trainer_improved.py
Self-Learning Osaka Weather AI - Improved Version (Priority Fixes)
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
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Failed to load model params: {e}")
    return {"temp_bias": 0.0, "rain_bias": 0.0, "cloud_to_rain_scale": 1.0}


# ===== ① CRITICAL: normalize_timestamp を安全に書き換える =====
def normalize_timestamp(ts: str) -> str:
    """
    タイムスタンプを安全に正規化
    
    入力例：
    - "2025-01-15T10:00:00+09:00"
    - "2025-01-15T10:00:00Z"
    - "2025-01-15T10:00:00"
    
    出力：
    - "2025-01-15T10:00" (JST)
    
    絶対に壊れないISOフォーマット正規化
    """
    try:
        # Zを+00:00に置換（fromisoformatが理解する形式に）
        normalized_ts = ts.replace("Z", "+00:00")
        
        # ISOフォーマットで解析
        dt = datetime.datetime.fromisoformat(normalized_ts)
        
        # すべてを Asia/Tokyo に変換
        dt = dt.astimezone(ZoneInfo("Asia/Tokyo"))
        
        # 時間解像度で丸める（分単位まで）
        return dt.strftime("%Y-%m-%dT%H:%M")
        
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to normalize timestamp '{ts}': {e}")
        # フォールバック：そのままの形式で返す
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


# ===== ② forecast.json の生成ロジックを安定化 =====
def regenerate_forecast(real_data: dict) -> dict:
    """
    forecast.json を再生成
    
    real_weather.json のエントリ数に合わせて生成
    """
    entries = real_data.get("entries", [])
    
    if not entries:
        raise ValueError("real_weather.json has no entries to generate forecast from")
    
    # 最初のエントリのタイムスタンプから開始
    start_ts = entries[0]["time"]
    
    try:
        # normalize_timestamp を使って安全に処理
        normalized_ts = normalize_timestamp(start_ts)
        start_dt = datetime.datetime.fromisoformat(normalized_ts).replace(
            tzinfo=ZoneInfo("Asia/Tokyo")
        )
    except Exception as e:
        logger.error(f"Failed to parse start timestamp: {e}")
        # フォールバック：現在時刻から開始
        start_dt = datetime.datetime.now(ZoneInfo("Asia/Tokyo")).replace(
            minute=0, second=0, microsecond=0
        )
    
    logger.info(f"Regenerating forecast for {len(entries)} hours from {start_dt}")
    
    # real_data と同じ時間数の予報を生成
    forecast_entries = synthesize_osaka_forecast(start_dt, hours=len(entries))
    
    # 保存
    forecast_data = forecast_to_json(forecast_entries)
    save_json(FORECAST_PATH, forecast_data)
    
    logger.info(f"Forecast regenerated: {len(forecast_entries)} entries saved")
    
    return forecast_data


def ensure_forecast_alignment_improved(real_data: dict) -> Tuple[dict, dict, dict, List[str]]:
    """
    改善版：forecast.json の整合性確保
    
    ② forecast.json の生成ロジックを安定化
    - ファイルが無い場合 → 生成
    - entries 数がズレている場合 → 再生成
    """
    
    real_entries_list = real_data.get("entries", [])
    real_entries_count = len(real_entries_list)
    
    # CASE 1: forecast.json が存在しない
    if not FORECAST_PATH.exists():
        logger.info("forecast.json not found; regenerating")
        forecast_data = regenerate_forecast(real_data)
    else:
        # CASE 2: forecast.json は存在するが、エントリ数がズレている
        try:
            forecast_data = load_json(FORECAST_PATH)
            forecast_entries_count = len(forecast_data.get("entries", []))
            
            if forecast_entries_count != real_entries_count:
                logger.warning(
                    f"Forecast entries ({forecast_entries_count}) != "
                    f"Real entries ({real_entries_count}); regenerating"
                )
                forecast_data = regenerate_forecast(real_data)
            else:
                logger.info(
                    f"Forecast and Real entries match: {real_entries_count} entries"
                )
        except Exception as e:
            logger.error(f"Failed to load forecast_data: {e}; regenerating")
            forecast_data = regenerate_forecast(real_data)
    
    # タイムスタンプを正規化してマッピング
    forecast_entries = {
        normalize_timestamp(entry["time"]): entry
        for entry in forecast_data.get("entries", [])
    }
    real_entries = {
        normalize_timestamp(entry["time"]): entry
        for entry in real_entries_list
    }
    
    logger.info(f"Forecast entries: {len(forecast_entries)}")
    logger.info(f"Real entries: {len(real_entries)}")
    
    common_times = sorted(set(forecast_entries) & set(real_entries))
    
    if not common_times:
        logger.error(
            f"CRITICAL: No overlapping timestamps after normalization!\n"
            f"  Forecast times (first 3): {list(forecast_entries.keys())[:3]}\n"
            f"  Real times (first 3): {list(real_entries.keys())[:3]}"
        )
        raise ValueError("Timestamp mismatch - see logs above")
    
    logger.info(f"✓ Found {len(common_times)} overlapping timestamps")
    
    return forecast_data, forecast_entries, real_entries, common_times


# ===== ③ fetch_real_weather() の fallback を安定化 =====
def fetch_real_weather_with_fallback() -> dict:
    """
    real_weather.json を取得
    
    ③ fetch_real_weather() が取得失敗したときの fallback が危険
    改善：検証とフォールバックを強化
    """
    
    if REAL_WEATHER_PATH.exists():
        try:
            real_data = load_json(REAL_WEATHER_PATH)
            entries = real_data.get("entries", [])
            
            # 最小限の検証：エントリ数が妥当か
            if len(entries) >= 12:  # 最低12時間以上
                logger.info(f"✓ Real weather loaded: {len(entries)} entries")
                return real_data
            else:
                logger.warning(
                    f"Real weather has only {len(entries)} entries (minimum 12 required); "
                    f"fetching fresh data"
                )
        except Exception as e:
            logger.warning(f"Failed to load existing real_weather.json: {e}")
    
    # ネットワークから取得
    logger.info("Fetching real weather from API...")
    try:
        real_weather = fetch_real_weather()
        entries = real_weather.get("entries", [])
        
        if len(entries) < 12:
            logger.warning(f"API returned only {len(entries)} entries; using fallback")
            # フォールバック：合成データを生成
            real_weather = generate_synthetic_fallback()
        else:
            logger.info(f"✓ Fresh real weather: {len(entries)} entries")
            save_json(REAL_WEATHER_PATH, real_weather)
        
        return real_weather
        
    except RequestException as exc:
        logger.error(f"Failed to fetch real weather: {exc}")
        logger.info("Using synthetic fallback")
        return generate_synthetic_fallback()


def generate_synthetic_fallback() -> dict:
    """
    合成 real_weather.json を生成
    
    24時間の予測データをそのまま実測として使用（開発用）
    """
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
    
    REAL_WEATHER_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(REAL_WEATHER_PATH, synthetic_real)
    
    logger.warning(
        f"✓ Synthetic fallback generated: {len(synthetic_entries)} entries saved"
    )
    
    return synthetic_real


def main() -> None:
    """メイン処理"""
    logger.info("=" * 60)
    logger.info("=== Trainer Start ===")
    logger.info("=" * 60)
    
    # ===== ③ 実測データを取得（fallback付き） =====
    real_data = fetch_real_weather_with_fallback()
    
    # ===== ② forecast.json の整合性確保 =====
    forecast_data, forecast_entries, real_entries, common_times = ensure_forecast_alignment_improved(real_data)
    
    # 誤差を計算
    temp_diffs: List[float] = []
    rain_diffs: List[float] = []
    forecast_rain: List[float] = []
    real_rain: List[float] = []
    forecast_temps: List[float] = []
    real_temps: List[float] = []
    
    logger.info(f"Computing errors for {len(common_times)} timestamps...")
    
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
    logger.info("✓ Model parameters saved")
    
    # 精度評価
    temp_metrics = calculate_mae_rmse_mape(forecast_temps, real_temps)
    rain_metrics = calculate_mae_rmse_mape(forecast_rain, real_rain)
    
    logger.info("=" * 60)
    logger.info("Temperature Metrics:")
    logger.info(f"  MAE:  {temp_metrics['mae']:.4f}°C")
    logger.info(f"  RMSE: {temp_metrics['rmse']:.4f}°C")
    logger.info(f"  MAPE: {temp_metrics['mape']:.2f}%")
    logger.info("Precipitation Metrics:")
    logger.info(f"  MAE:  {rain_metrics['mae']:.2f}%")
    logger.info(f"  RMSE: {rain_metrics['rmse']:.2f}%")
    logger.info(f"  MAPE: {rain_metrics['mape']:.2f}%")
    logger.info(f"  Valid samples: {rain_metrics['valid_samples']}")
    logger.info("=" * 60)
    
    logger.info("=== Trainer Complete ===")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
