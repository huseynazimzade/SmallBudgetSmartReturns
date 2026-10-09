# import pandas as pd
# import numpy as np
# from pathlib import Path

# BASE = Path(__file__).parent.parent
# RAW_PATH = BASE / "data" / "raw" / "Merged_AllAssets_Real_CPI.xlsx"
# OUT_DIR = BASE / "data" / "processed"
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# NOMINAL_COLS = {
#     'Gold':   'Gold_Nominal',
#     'Silver': 'Silver_Nominal',
#     'DXY':    'DXY_Nominal',
#     'EURUSD': 'EURUSD_Nominal',
#     'CHFUSD': 'CHFUSD_Nominal',
#     'ING':    'ING_Nominal',
#     'Apple':  'Apple_Nominal',
#     'Tesla':  'Tesla_Nominal',
# }


# def add_features(df):
#     df = df.copy()
#     p = df['price'].astype(float)
#     log_p = np.log(p)
#     log_ret = log_p.diff()

#     for n in [1, 3, 5, 10, 20]:
#         df[f'ret_{n}'] = p.pct_change(n)

#     df['log_ret'] = log_ret

#     for n in [5, 10, 20, 60]:
#         df[f'vol_{n}'] = log_ret.rolling(n).std()

#     df['vol_ratio_5_20'] = df['vol_5'] / (df['vol_20'] + 1e-9)
#     df['vol_ratio_10_60'] = df['vol_10'] / (df['vol_60'] + 1e-9)

#     for n in [5, 10, 20, 50, 200]:
#         ma = p.rolling(n).mean()
#         df[f'price_vs_ma{n}'] = p / ma - 1

#     df['momentum_10'] = p / p.shift(10) - 1
#     df['momentum_20'] = p / p.shift(20) - 1

#     delta = p.diff()
#     gain = delta.where(delta > 0, 0.0).rolling(14).mean()
#     loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
#     rs = gain / (loss + 1e-9)
#     df['rsi_14'] = 100 - 100 / (1 + rs)

#     ema12 = p.ewm(span=12, adjust=False).mean()
#     ema26 = p.ewm(span=26, adjust=False).mean()
#     macd = ema12 - ema26
#     macd_signal = macd.ewm(span=9, adjust=False).mean()
#     df['macd_norm'] = macd / p
#     df['macd_hist_norm'] = (macd - macd_signal) / p

#     ma20 = p.rolling(20).mean()
#     std20 = p.rolling(20).std()
#     df['bb_position'] = (p - ma20) / (2 * std20 + 1e-9)

#     df['dist_high_60'] = p / p.rolling(60).max() - 1
#     df['dist_low_60'] = p / p.rolling(60).min() - 1
#     df['dist_high_252'] = p / p.rolling(252).max() - 1
#     df['dist_low_252'] = p / p.rolling(252).min() - 1

#     for n in [20, 60]:
#         m = p.rolling(n).mean()
#         s = p.rolling(n).std()
#         df[f'zscore_{n}'] = (p - m) / (s + 1e-9)

#     df['day_of_week'] = df['date'].dt.dayofweek
#     df['month'] = df['date'].dt.month
#     df['quarter'] = df['date'].dt.quarter

#     df['target_return_1d'] = log_ret.shift(-1)
#     df['target_direction_1d'] = (df['target_return_1d'] > 0).astype(int)

#     df['target_return_5d'] = (log_p.shift(-5) - log_p)
#     df['target_direction_5d'] = (df['target_return_5d'] > 0).astype(int)

#     return df


# FEATURE_COLS = [
#     'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
#     'log_ret',
#     'vol_5', 'vol_10', 'vol_20', 'vol_60',
#     'vol_ratio_5_20', 'vol_ratio_10_60',
#     'price_vs_ma5', 'price_vs_ma10', 'price_vs_ma20',
#     'price_vs_ma50', 'price_vs_ma200',
#     'momentum_10', 'momentum_20',
#     'rsi_14',
#     'macd_norm', 'macd_hist_norm',
#     'bb_position',
#     'dist_high_60', 'dist_low_60',
#     'dist_high_252', 'dist_low_252',
#     'zscore_20', 'zscore_60',
#     'day_of_week', 'month', 'quarter',
# ]

# TARGET_COLS = [
#     'target_return_1d', 'target_direction_1d',
#     'target_return_5d', 'target_direction_5d',
# ]


# print("=" * 70)
# print("ML Data Creator — Overfitting & Leakage Protected")
# print("=" * 70)

# raw = pd.read_excel(RAW_PATH)
# raw['date'] = pd.to_datetime(raw['date'])
# raw = raw.sort_values('date').reset_index(drop=True)

# print(f"\n📥 Mənbə: {RAW_PATH.name}")
# print(f"   Sətir: {len(raw)}")
# print(f"   Dövr:  {raw['date'].min().date()} → {raw['date'].max().date()}")
# print(f"   Feature: {len(FEATURE_COLS)}")

# summary = []

# for asset, col in NOMINAL_COLS.items():
#     if col not in raw.columns:
#         print(f"⚠️  {col} tapılmadı — {asset} keçildi")
#         continue

#     df = raw[['date', col]].copy()
#     df.columns = ['date', 'price']
#     df['price'] = pd.to_numeric(df['price'], errors='coerce')
#     df = df.dropna(subset=['price']).reset_index(drop=True)

#     df = add_features(df)
#     df = df.dropna(subset=FEATURE_COLS + TARGET_COLS).reset_index(drop=True)

#     keep = ['date'] + FEATURE_COLS + TARGET_COLS
#     df = df[keep]

#     out_path = OUT_DIR / f"ml_{asset}.xlsx"
#     df.to_excel(out_path, index=False)

#     summary.append({
#         'Asset': asset,
#         'Sətir': len(df),
#         'Feature': len(FEATURE_COLS),
#         'Başlanğıc': df['date'].min().date(),
#         'Son': df['date'].max().date(),
#     })

#     print(f"   ✓ {asset:8s} → {len(df):4d} sətir")

# summary_df = pd.DataFrame(summary)
# summary_df.to_excel(OUT_DIR / "_summary.xlsx", index=False)

# print("\n" + "=" * 70)
# print("✅ HAZIR!")
# print(f"📁 {OUT_DIR.resolve()}")
# print("=" * 70)
# print("\n" + summary_df.to_string(index=False))

# print(f"\n📋 Feature siyahısı ({len(FEATURE_COLS)} ədəd):")
# for i, c in enumerate(FEATURE_COLS, 1):
#     print(f"   {i:2d}. {c}")

# print(f"\n🎯 Target siyahısı ({len(TARGET_COLS)} ədəd):")
# for c in TARGET_COLS:
#     print(f"   - {c}")