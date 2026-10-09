import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

SRC_DIR = Path(__file__).parent
DATA_DIR = SRC_DIR.parent / "data" / "processed"
RAW_PATH = SRC_DIR.parent / "data" / "raw" / "Merged_AllAssets_Real_CPI.xlsx"
OUT_DIR = SRC_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ['Gold', 'Silver', 'DXY', 'EURUSD', 'CHFUSD', 'ING', 'Apple', 'Tesla']

NOMINAL_COLS = {
    'Gold':   'Gold_Nominal',
    'Silver': 'Silver_Nominal',
    'DXY':    'DXY_Nominal',
    'EURUSD': 'EURUSD_Nominal',
    'CHFUSD': 'CHFUSD_Nominal',
    'ING':    'ING_Nominal',
    'Apple':  'Apple_Nominal',
    'Tesla':  'Tesla_Nominal',
}

HORIZON = 5
SPLIT_RATIO = 0.80
HC_THRESHOLD = 0.10
N_SELECTED = 5


raw = pd.read_excel(RAW_PATH)
raw['date'] = pd.to_datetime(raw['date'])
raw = raw.sort_values('date').reset_index(drop=True)

prices = pd.DataFrame({'date': raw['date']})
for a, c in NOMINAL_COLS.items():
    prices[a] = pd.to_numeric(raw[c], errors='coerce')
prices = prices.ffill().dropna().reset_index(drop=True)


def build_features(asset):
    ml = pd.read_excel(DATA_DIR / f"ml_{asset}.xlsx")
    ml['date'] = pd.to_datetime(ml['date'])
    ml = ml.sort_values('date').reset_index(drop=True)

    price_df = prices[['date', asset]].copy()
    price_df.columns = ['date', 'price']
    ml = ml.merge(price_df, on='date', how='left')

    for other in ASSETS:
        if other == asset:
            continue
        odf = prices[['date', other]].copy()
        odf.columns = ['date', 'p']
        ml = ml.merge(odf, on='date', how='left')
        ml[f'x_{other}_r5'] = ml['p'].pct_change(5)
        ml[f'x_{other}_r20'] = ml['p'].pct_change(20)
        ml = ml.drop(columns=['p'])

    x5 = [c for c in ml.columns if c.startswith('x_') and c.endswith('_r5')]
    ml['x_avg_r5'] = ml[x5].mean(axis=1)
    ml['x_std_r5'] = ml[x5].std(axis=1)

    own_r5 = ml['price'].pct_change(5)
    own_r20 = ml['price'].pct_change(20)
    ml['rel_strength_5'] = own_r5 - ml['x_avg_r5']
    ml['rel_strength_20'] = own_r20 - ml[x5].mean(axis=1).shift(15)

    lr = np.log(ml['price'] / ml['price'].shift(1))
    ml['vol_20'] = lr.rolling(20).std()
    ml['vol_60'] = lr.rolling(60).std()
    ml['vol_regime'] = ml['vol_20'] / (ml['vol_60'] + 1e-9)

    ma20 = ml['price'].rolling(20).mean()
    ma60 = ml['price'].rolling(60).mean()
    ma200 = ml['price'].rolling(200).mean()
    ml['trend_20_60'] = ma20 / ma60 - 1
    ml['trend_60_200'] = ma60 / ma200 - 1

    ml['future_ret'] = ml['price'].shift(-HORIZON) / ml['price'] - 1
    ml['target'] = (ml['future_ret'] > 0).astype(int)

    exclude = ['date', 'price', 'target', 'future_ret',
               'target_return_1d', 'target_direction_1d',
               'target_return_5d', 'target_direction_5d']
    feature_cols = [c for c in ml.columns if c not in exclude]

    ml = ml.dropna(subset=feature_cols + ['target', 'future_ret']).reset_index(drop=True)
    return ml, feature_cols


def train_asset(asset):
    ml, feats = build_features(asset)

    X = ml[feats].values
    y = ml['target'].values
    fut_ret = ml['future_ret'].values

    split_idx = int(len(ml) * SPLIT_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    fut_test = fut_ret[split_idx:]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=15,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        tree_method='hist',
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]

    baseline = max(y_test.mean(), 1 - y_test.mean())

    y_pred_all = (p > 0.5).astype(int)
    acc_all = accuracy_score(y_test, y_pred_all)

    conf = np.abs(p - 0.5)
    mask = conf >= HC_THRESHOLD

    if mask.sum() < 20:
        return None

    y_pred_hc = (p[mask] > 0.5).astype(int)
    acc_hc = accuracy_score(y_test[mask], y_pred_hc)
    coverage = mask.mean() * 100

    buy_mask = mask & (p > 0.5)
    avg_buy = fut_test[buy_mask].mean() * 100 if buy_mask.sum() > 0 else np.nan
    n_buys = buy_mask.sum()

    auc = roc_auc_score(y_test, p) if len(np.unique(y_test)) > 1 else np.nan

    return {
        'Asset': asset,
        'Baseline%': round(baseline * 100, 2),
        'Acc_All%': round(acc_all * 100, 2),
        'Coverage%': round(coverage, 1),
        'N_HC': mask.sum(),
        'Acc_HC%': round(acc_hc * 100, 2),
        'N_Buys': n_buys,
        'AvgRet_Buy%': round(avg_buy, 2) if not np.isnan(avg_buy) else np.nan,
        'AUC': round(auc, 3) if not np.isnan(auc) else np.nan,
    }


print("=" * 110)
print(f"Step 1: Test ALL 8 Assets (HC Threshold = {HC_THRESHOLD})")
print("=" * 110)

results = []
for asset in ASSETS:
    res = train_asset(asset)
    if res is None:
        print(f"⚠️  {asset}: kifayət qədər HC nümunə yoxdur")
        continue
    results.append(res)
    print(f"\n🔄 {asset}")
    print(f"   Baseline:      {res['Baseline%']:.2f}%")
    print(f"   Acc (all):     {res['Acc_All%']:.2f}%")
    print(f"   Coverage:      {res['Coverage%']:.1f}%  ({res['N_HC']} sətir)")
    print(f"   Acc (HC):      {res['Acc_HC%']:.2f}%")
    print(f"   Buys:          {res['N_Buys']}  | AvgRet: {res['AvgRet_Buy%']:+.2f}%")
    print(f"   AUC:           {res['AUC']}")

df = pd.DataFrame(results)
df.to_excel(OUT_DIR / "all_assets_test.xlsx", index=False)

print("\n" + "=" * 110)
print("📊 BÜTÜN AKTİVLƏR")
print("=" * 110)
print(df.to_string(index=False))

df_sorted = df.sort_values('Acc_HC%', ascending=False).reset_index(drop=True)

print("\n" + "=" * 110)
print("🏆 REYTİNQ (HC Accuracy üzrə)")
print("=" * 110)
for i, row in df_sorted.iterrows():
    print(f"   {i+1}. {row['Asset']:8s} → HC Acc: {row['Acc_HC%']:.2f}%  | "
          f"Coverage: {row['Coverage%']:.1f}%  | "
          f"AvgRet: {row['AvgRet_Buy%']:+.2f}%  | "
          f"Baseline: {row['Baseline%']:.2f}%")

top_n = df_sorted.head(N_SELECTED)

print("\n" + "=" * 110)
print(f"✅ SEÇİLƏN {N_SELECTED} AKTİV (Portfel üçün)")
print("=" * 110)
print(top_n[['Asset', 'Acc_HC%', 'Coverage%', 'AvgRet_Buy%', 'AUC']].to_string(index=False))

print(f"\n   Orta HC Accuracy:   {top_n['Acc_HC%'].mean():.2f}%")
print(f"   Orta Coverage:      {top_n['Coverage%'].mean():.2f}%")
print(f"   Orta Avg Buy ret:   {top_n['AvgRet_Buy%'].mean():+.2f}%")
print(f"   Orta AUC:           {top_n['AUC'].mean():.3f}")

print(f"\n   Baseline-dan yaxşı: "
      f"{(top_n['Acc_HC%'] > top_n['Baseline%']).sum()}/{len(top_n)}")

top_n.to_excel(OUT_DIR / "selected_assets.xlsx", index=False)

print("\n" + "=" * 110)
print("🗑️  SEÇİLMƏYƏN AKTİVLƏR (portfelə daxil edilmir)")
print("=" * 110)
not_selected = df_sorted.iloc[N_SELECTED:]
print(not_selected[['Asset', 'Acc_HC%', 'Baseline%', 'AvgRet_Buy%']].to_string(index=False))

print(f"\n✅ Saxlanıldı:")
print(f"   - all_assets_test.xlsx       (bütün 8 test)")
print(f"   - selected_assets.xlsx       (seçilən {N_SELECTED})")
print(f"📁 {OUT_DIR.resolve()}")