# --- NEW: 現実的な最低気温 (Osaka 気象庁ベース) ---
MONTHLY_MIN_TEMP = {
    1: 3.0, 2: 3.5, 3: 6.0, 4: 11.0,
    5: 16.0, 6: 21.0, 7: 25.0, 8: 26.0,
    9: 22.0, 10: 16.0, 11: 10.0, 12: 5.0,
}

# --- NEW: 日内温度カーブ（実測に基づく比率） ---
DIURNAL_CURVE = {
     0: 0.20,  1: 0.18,  2: 0.17,  3: 0.16,
     4: 0.15,  5: 0.15,  6: 0.18,  7: 0.25,
     8: 0.35,  9: 0.48, 10: 0.63, 11: 0.75,
    12: 0.85, 13: 0.90, 14: 1.00, 15: 0.95,
    16: 0.90, 17: 0.78, 18: 0.60, 19: 0.45,
    20: 0.35, 21: 0.30, 22: 0.25, 23: 0.22,
}

def synthesize_osaka_forecast(start: datetime.datetime, hours=24):
    """
    Osaka local-climate realistic forecast generator.
    """

    rng = random.Random(int(start.strftime("%Y%m%d")))
    month = start.month

    base_temp, temp_range = MONTHLY_BASE_TEMP[month]  # seasonal range
    precip_base = MONTHLY_PRECIP_BASE[month]

    # --- NEW: その日の最低気温を決める（現実的＋少しだけランダム） ---
    base_min = MONTHLY_MIN_TEMP[month]
    tmin = base_min + rng.uniform(-0.8, 0.8)
    tmax = tmin + temp_range  # 既存の「季節の幅」をそのまま使う

    storm_bias = 18 if month in {6, 7, 8, 9} else 8

    forecast = []

    for step in range(hours):
        t = start + datetime.timedelta(hours=step)
        h = t.hour

        # -----------------------------
        # 降水確率・天気コード（従来ロジック）
        # -----------------------------
        # 雲量の推定（diurnal を弱くする）
        diurnal = DIURNAL_CURVE[h]
        cloud_factor = clamp((1 - diurnal) * 0.6 + rng.uniform(0, 0.3), 0, 1)
        precip_prob = clamp(precip_base + cloud_factor * 30 + rng.uniform(-10, 15), 0, 100)

        roll = rng.uniform(0, 100)
        if roll < precip_prob * 0.6:
            code = 61 if precip_prob < storm_bias + 25 else 80
        elif roll < precip_prob:
            code = 51
        else:
            code = 3 if cloud_factor > 0.75 else 2 if cloud_factor > 0.35 else 0

        # Heavy storm condition
        if precip_prob > 85 and rng.random() < 0.15:
            code = 95

        # -----------------------------
        # 温度計算（大幅強化版）
        # -----------------------------
        # 基本カーブ
        temp = tmin + (tmax - tmin) * DIURNAL_CURVE[h]

        # 天気による温度補正（現実の物理に基づく）
        if code in {3, 45}:       # 曇り・霧
            temp -= 1.2
        elif code in {51, 61, 80}:  # 雨
            temp -= 2.0
        elif code == 95:          # 雷雨
            temp -= 3.0

        # 最終ノイズは弱めて自然に
        temp += rng.uniform(-0.4, 0.4)
        temp = round(temp, 1)

        forecast.append((
            t.strftime("%Y-%m-%dT%H:%M"),
            temp,
            code,
            int(round(precip_prob))
        ))

    return forecast
