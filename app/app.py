import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "Merged_AllAssets_Real_CPI.xlsx"
ML_SELECTED_PATH = Path(__file__).parent.parent / "src" / "selected_assets.xlsx"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

st.set_page_config(page_title="Small Budget, Smart Returns", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df

@st.cache_data
def load_ml_selected():
    if ML_SELECTED_PATH.exists():
        return pd.read_excel(ML_SELECTED_PATH)
    return None

df = load_data()
real_cols = ['Gold_Real', 'Silver_Real', 'ING_Real', 'Apple_Real', 'Tesla_Real',
             'DXY_Real', 'EURUSD_Real', 'CHFUSD_Real']
names = [c.replace('_Real', '') for c in real_cols]

data = df[real_cols].ffill()
monthly = data.resample('ME').last()
daily_returns = data.pct_change().dropna()


def cagr(prices, periods=12):
    n = len(prices)
    y = n / periods
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / y) - 1


# ============= SIDEBAR — 4 bölmə =============
st.sidebar.title("Small Budget, Smart Returns")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "**Bölmə seç:**",
    ["📊 Analizlər", "🎯 Simulyator", "🤖 ML Model", "📁 Dataset"],
    key="section_radio"
)

st.sidebar.markdown("---")

if section == "📊 Analizlər":
    analiz = st.sidebar.radio(
        "**Alt bölmə:**",
        [
            "Giriş",
            "1. CAGR — Real İllik Gəlir",
            "2. Volatillik",
            "3. Sharpe Ratio",
            "4. Max Drawdown",
            "5. VaR 95%",
            "6. Skewness & Kurtosis",
            "7. Korelyasiya",
            "8. $1,000 Büyümə",
            "9. DCA vs Lump Sum",
            "10. Markowitz Portfel",
            "11. Rebalancing vs Buy & Hold",
            "12. Yekun Qərar Matrisi",
        ],
        key="analiz_radio"
    )
elif section == "🎯 Simulyator":
    analiz = "Simulyator"
elif section == "🤖 ML Model":
    analiz = "ML"
else:
    analiz = "Dataset"
# =======================================================


if analiz == "Giriş":
    st.title("Small Budget, Smart Returns")
    st.markdown("### 2016 – 2026 | İnflyasiya Düzəlişli Analiz")
    st.markdown("""
    Bu sayt 8 investisiya aktivinin **inflyasiyadan təmizlənmiş real gəlirini** analiz edir.

    **Aktivlər:**
    - **Səhmlər:** Apple, Tesla, ING
    - **Valyutalar:** DXY, EUR/USD, CHF/USD
    - **Metallar:** Qızıl, Gümüş

    **Struktur:**
    - 📊 **Analizlər** — 12 statistik analiz
    - 🎯 **Simulyator** — $1,000 ilə keçmiş tarixi simulyasiya
    - 🤖 **ML Model** — maşın öyrənməsi proqnozu
    - 📁 **Dataset** — layihənin xam datası
    """)


elif analiz == "1. CAGR — Real İllik Gəlir":
    st.title("1. CAGR — Real İllik Gəlir")
    st.markdown("**CAGR** = Compound Annual Growth Rate = Mürəkkəb illik böyümə tempi")

    cagr_vals = monthly.apply(lambda x: cagr(x))
    cagr_df = pd.DataFrame({
        'Real CAGR (%)': (cagr_vals * 100).round(2)
    }).sort_values('Real CAGR (%)', ascending=True)
    cagr_df.index = [c.replace('_Real', '') for c in cagr_df.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#2e7d32' if v > 0 else '#c62828' for v in cagr_df['Real CAGR (%)']]
    bars = ax.barh(cagr_df.index, cagr_df['Real CAGR (%)'], color=colors)
    for bar, val in zip(bars, cagr_df['Real CAGR (%)']):
        y = bar.get_y() + bar.get_height() / 2
        if val > 0:
            ax.text(val + 0.4, y, f'{val:.2f}%', va='center', ha='left',
                    fontsize=10, fontweight='bold', color='#2e7d32')
        else:
            ax.text(0.4, y, f'{val:.2f}%', va='center', ha='left',
                    fontsize=10, fontweight='bold', color='#c62828')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Real CAGR (%)')
    ax.set_title('Real Annual Return (2016-2026)')
    ax.set_xlim(-3.5, max(cagr_df['Real CAGR (%)']) + 3)
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(cagr_df.sort_values('Real CAGR (%)', ascending=False),
                 use_container_width=True)

    st.markdown("### Nəticələr bizə nə deyir?")
    st.markdown("""
    **Qaliblər:** Tesla 31.34%, Apple 24.86%, ING 12.82%, Silver 10.59%, Gold 9.40%

    **Məğlublar:** DXY -2.97%, EURUSD -2.74%, CHFUSD -1.10%

    **Mesaj:** Kapital artırmaq üçün səhm və metal, sığorta üçün valyuta.
    """)


elif analiz == "2. Volatillik":
    st.title("2. Annualized Volatility (İllik Volatillik)")
    st.markdown("**Volatillik** = aktivin illik bazada neçə faiz dalğalandığını göstərir.")

    vol_daily = daily_returns.std()
    vol_annual = vol_daily * np.sqrt(252) * 100
    vol_df = pd.DataFrame({
        'Annualized Volatility (%)': vol_annual.round(2)
    }).sort_values('Annualized Volatility (%)', ascending=True)
    vol_df.index = [c.replace('_Real', '') for c in vol_df.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#1565c0' if v < 25 else '#ef6c00' if v < 45 else '#c62828'
              for v in vol_df['Annualized Volatility (%)']]
    bars = ax.barh(vol_df.index, vol_df['Annualized Volatility (%)'], color=colors)
    for bar, val in zip(bars, vol_df['Annualized Volatility (%)']):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(val + 0.5, y, f'{val:.2f}%', va='center', ha='left',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('Annualized Volatility (%)')
    ax.set_title('Annualized Volatility of Real Returns')
    ax.set_xlim(0, max(vol_df['Annualized Volatility (%)']) + 6)
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(vol_df.sort_values('Annualized Volatility (%)', ascending=False),
                 use_container_width=True)

    st.markdown("### Nəticələr bizə nə deyir?")
    st.markdown("""
    **Ən sakit:** DXY 6.63%, EURUSD 7.41%, CHFUSD 7.54%
    **Orta risk:** Apple 28.38%, Silver 33.37%, ING 34.27%
    **Ən həyəcanlı:** Tesla 57.66%
    """)


elif analiz == "3. Sharpe Ratio":
    st.title("3. Sharpe Ratio")
    st.markdown("**Sharpe Ratio** = Real CAGR / Volatility")

    cagr_vals = monthly.apply(lambda x: cagr(x))
    vol_annual = daily_returns.std() * np.sqrt(252)
    sharpe = cagr_vals / vol_annual
    sharpe_df = pd.DataFrame({
        'CAGR (%)': (cagr_vals * 100).round(2),
        'Volatility (%)': (vol_annual * 100).round(2),
        'Sharpe Ratio': sharpe.round(3)
    }).sort_values('Sharpe Ratio', ascending=True)
    sharpe_df.index = [c.replace('_Real', '') for c in sharpe_df.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#2e7d32' if v >= 0.5 else '#ef6c00' if v >= 0.2 else '#c62828'
              for v in sharpe_df['Sharpe Ratio']]
    bars = ax.barh(sharpe_df.index, sharpe_df['Sharpe Ratio'], color=colors)
    for bar, val in zip(bars, sharpe_df['Sharpe Ratio']):
        y = bar.get_y() + bar.get_height() / 2
        if val >= 0:
            ax.text(val + 0.02, y, f'{val:.3f}', va='center', ha='left',
                    fontsize=10, fontweight='bold')
        else:
            ax.text(0.02, y, f'{val:.3f}', va='center', ha='left',
                    fontsize=10, fontweight='bold', color='#c62828')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio (Real, Risk-Free = 0)')
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(sharpe_df.sort_values('Sharpe Ratio', ascending=False),
                 use_container_width=True)

    st.markdown("### Nəticələr bizə nə deyir?")
    st.markdown("""
    **Əla (Sharpe > 0.5):** Apple 0.876, Gold 0.572, Tesla 0.544
    **Zəif (0-0.5):** ING 0.374, Silver 0.317
    **Mənfi:** DXY, EURUSD, CHFUSD
    """)


elif analiz == "4. Max Drawdown":
    st.title("4. Maximum Drawdown & Recovery")
    st.markdown("**Max Drawdown** = ən yüksək zirvədən ən dərin dibə düşüş faizi.")

    def max_drawdown(prices):
        dd = prices / prices.cummax() - 1
        return dd.min(), dd.idxmin(), prices.loc[:dd.idxmin()].idxmax()

    results = []
    for col in real_cols:
        prices = data[col]
        mdd, trough, peak = max_drawdown(prices)
        results.append({
            'Asset': col.replace('_Real', ''),
            'Max Drawdown (%)': round(mdd * 100, 2),
            'Peak': peak.date(),
            'Trough': trough.date()
        })
    dd_df = pd.DataFrame(results).sort_values('Max Drawdown (%)', ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#2e7d32' if v > -20 else '#ef6c00' if v > -50 else '#c62828'
              for v in dd_df['Max Drawdown (%)']]
    bars = ax.barh(dd_df['Asset'], dd_df['Max Drawdown (%)'], color=colors)
    for bar, val in zip(bars, dd_df['Max Drawdown (%)']):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(val - 1, y, f'{val:.2f}%', va='center', ha='right',
                fontsize=10, fontweight='bold', color='white')
    ax.set_xlabel('Max Drawdown (%)')
    ax.set_title('Maximum Drawdown of Real Prices')
    ax.set_xlim(min(dd_df['Max Drawdown (%)']) - 5, 0)
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(dd_df.set_index('Asset'), use_container_width=True)

    st.markdown("### Nəticələr bizə nə deyir?")
    st.markdown("""
    **Ən dözümlü:** CHFUSD -24.75%, Gold -31.16%, DXY -31.18%
    **Yüksək risk:** Tesla -75.52%, ING -75.28%, Silver -52.34%
    """)


elif analiz == "5. VaR 95%":
    st.title("5. Value at Risk (VaR 95%)")
    st.markdown("**VaR 95%** = 95% ehtimalla 1 gündə gözlənilən maksimum itki.")

    var_95 = daily_returns.quantile(0.05) * 100
    var_99 = daily_returns.quantile(0.01) * 100

    var_df = pd.DataFrame({
        'VaR 95% Daily (%)': var_95.round(2),
        'VaR 99% Daily (%)': var_99.round(2)
    }).sort_values('VaR 95% Daily (%)', ascending=True)
    var_df.index = [c.replace('_Real', '') for c in var_df.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(var_df))
    width = 0.38
    ax.barh(x - width/2, var_df['VaR 95% Daily (%)'], width,
            label='VaR 95%', color='#ef6c00')
    ax.barh(x + width/2, var_df['VaR 99% Daily (%)'], width,
            label='VaR 99%', color='#c62828')
    ax.set_yticks(x)
    ax.set_yticklabels(var_df.index)
    ax.set_xlabel('Daily Loss Threshold (%)')
    ax.set_title('Value at Risk — Historical Method')
    ax.legend(loc='lower left')
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(var_df, use_container_width=True)

    st.markdown("### Nəticələr bizə nə deyir?")
    st.markdown("""
    **Sakit:** DXY -0.70%, CHFUSD -0.74%, EURUSD -0.73%
    **Orta:** Gold -1.62%, Apple -2.69%
    **Yüksək:** Tesla -5.34%, ING -3.03%, Silver -3.06%
    """)


elif analiz == "6. Skewness & Kurtosis":
    st.title("6. Skewness & Kurtosis — Ekstremal Risk")

    skew = daily_returns.skew()
    kurt = daily_returns.kurtosis()

    sk_df = pd.DataFrame({
        'Skewness': skew.round(2),
        'Excess Kurtosis': kurt.round(2)
    }).sort_values('Excess Kurtosis', ascending=True)
    sk_df.index = [c.replace('_Real', '') for c in sk_df.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors_skew = ['#c62828' if v < -0.3 else '#2e7d32' if v > 0.3 else '#616161'
                   for v in sk_df['Skewness']]
    axes[0].barh(sk_df.index, sk_df['Skewness'], color=colors_skew)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel('Skewness')
    axes[0].set_title('Skewness of Real Daily Returns')
    for i, v in enumerate(sk_df['Skewness']):
        if v >= 0:
            axes[0].text(v + 0.03, i, f'{v:.2f}', va='center', ha='left',
                         fontsize=9, fontweight='bold')
        else:
            axes[0].text(0.03, i, f'{v:.2f}', va='center', ha='left',
                         fontsize=9, fontweight='bold', color='#c62828')
    skew_min = sk_df['Skewness'].min()
    axes[0].set_xlim(skew_min - 0.3, sk_df['Skewness'].max() + 0.4)

    colors_kurt = ['#c62828' if v > 10 else '#ef6c00' if v > 3 else '#2e7d32'
                   for v in sk_df['Excess Kurtosis']]
    axes[1].barh(sk_df.index, sk_df['Excess Kurtosis'], color=colors_kurt)
    axes[1].axvline(3, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    axes[1].set_xlabel('Excess Kurtosis')
    axes[1].set_title('Excess Kurtosis of Real Daily Returns')
    for i, v in enumerate(sk_df['Excess Kurtosis']):
        axes[1].text(v + 0.3, i, f'{v:.2f}', va='center', ha='left',
                     fontsize=9, fontweight='bold')
    axes[1].set_xlim(0, sk_df['Excess Kurtosis'].max() + 3)

    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(sk_df, use_container_width=True)

    st.markdown("### Nəticələr bizə nə deyir?")
    st.markdown("""
    **Normal (< 5):** DXY 1.82, EURUSD 2.21, CHFUSD 3.97, Tesla 4.57
    **Orta (5-15):** Apple 7.17, Gold 9.00, ING 12.51
    **Yüksək (>15):** Silver 22.24
    """)


elif analiz == "7. Korelyasiya":
    st.title("7. Correlation Matrix — Korelyasiya Xəritəsi")

    secilmis = st.multiselect(
        "Hansı aktivləri göstərək?",
        options=names,
        default=names,
        key="corr_select"
    )

    if len(secilmis) < 2:
        st.warning("Ən azı 2 aktiv seç.")
        st.stop()

    secilmis_cols = [c + '_Real' for c in secilmis]
    filtered_returns = daily_returns[secilmis_cols]
    corr = filtered_returns.corr()
    corr.index = secilmis
    corr.columns = secilmis

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=0.8, vmin=-1, vmax=1, square=True,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Asset Real Daily Return Correlation')
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(corr.round(2), use_container_width=True)


elif analiz == "8. $1,000 Büyümə":
    st.title("8. Rebased $1,000 Growth")
    st.markdown("2016-cı ildə hər aktive **$1,000** qoyulsaydı, bu gün real olaraq nə qədər olardı?")

    secilmis = st.multiselect(
        "Hansı aktivləri göstərək?",
        options=names,
        default=names,
        key="growth_select"
    )

    if len(secilmis) == 0:
        st.warning("Ən azı 1 aktiv seç.")
        st.stop()

    secilmis_cols = [c + '_Real' for c in secilmis]
    rebased = data[secilmis_cols] / data[secilmis_cols].iloc[0] * 1000
    rebased.columns = secilmis

    fig = go.Figure()
    for col in rebased.columns:
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased[col],
            mode='lines', name=col,
            line=dict(width=2),
            hovertemplate="<b>" + col + "</b><br>Tarix: %{x|%Y-%m-%d}<br>Dəyər: $%{y:,.2f}<extra></extra>"
        ))

    fig.add_hline(y=1000, line_dash="dash", line_color="black",
                  opacity=0.6, annotation_text="İlkin $1,000")

    fig.update_layout(
        title="Rebased Growth of $1,000 — Real Returns",
        xaxis_title="Tarix",
        yaxis_title="Real Portfolio Value ($)",
        hovermode="x unified",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    final_vals = rebased.iloc[-1].sort_values(ascending=False).round(0).astype(int)
    final_df = pd.DataFrame({
        'Final Value ($)': final_vals,
        'Total Return (%)': ((final_vals / 1000 - 1) * 100).round(1)
    })
    st.dataframe(final_df, use_container_width=True)


elif analiz == "9. DCA vs Lump Sum":
    st.title("9. DCA vs Lump Sum")

    weights = np.array([1/8] * 8)
    lump_amount = 12000
    monthly_amount = 100

    def lump_sum(prices, weights, amount):
        return (prices / prices.iloc[0] * weights).sum(axis=1) * amount

    def dca(prices, weights, monthly_amount):
        units = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for i, date in enumerate(prices.index):
            units.iloc[i] = weights * monthly_amount / prices.iloc[i]
        cum_units = units.cumsum()
        port = (cum_units * prices).sum(axis=1)
        invested = pd.Series(np.arange(1, len(prices) + 1) * monthly_amount,
                             index=prices.index)
        return port, invested

    ls = lump_sum(monthly, weights, lump_amount)
    dca_port, dca_invested = dca(monthly, weights, monthly_amount)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ls.index, y=ls.values, mode='lines',
        name='Lump Sum ($12,000 at start)',
        line=dict(width=2.5, color='#1565c0'),
        hovertemplate="<b>Lump Sum</b><br>Tarix: %{x|%Y-%m-%d}<br>Dəyər: $%{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=dca_port.index, y=dca_port.values, mode='lines',
        name='DCA ($100 monthly)',
        line=dict(width=2.5, color='#2e7d32'),
        hovertemplate="<b>DCA portfel</b><br>Tarix: %{x|%Y-%m-%d}<br>Dəyər: $%{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=dca_invested.index, y=dca_invested.values, mode='lines',
        name='Total Invested (DCA)',
        line=dict(width=1.8, color='gray', dash='dash'),
        hovertemplate="<b>Yatırılan kapital</b><br>Tarix: %{x|%Y-%m-%d}<br>Dəyər: $%{y:,.2f}<extra></extra>"
    ))
    fig.update_layout(
        title="DCA vs Lump Sum — Equal Weight Portfolio",
        xaxis_title="Tarix", yaxis_title="Real Portfolio Value ($)",
        hovermode="x unified", height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Lump Sum final", f"${ls.iloc[-1]:,.0f}")
    col2.metric("DCA final", f"${dca_port.iloc[-1]:,.0f}")
    col3.metric("Total Invested", f"${dca_invested.iloc[-1]:,.0f}")


elif analiz == "10. Markowitz Portfel":
    st.title("10. Markowitz Optimal Portfolio")

    from pypfopt import EfficientFrontier, risk_models, expected_returns

    mu = expected_returns.mean_historical_return(monthly, frequency=12)
    S = risk_models.sample_cov(monthly, frequency=12)

    ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.35))
    ef.max_sharpe()
    weights = ef.clean_weights()

    w_df = pd.DataFrame({
        'Weight (%)': {k.replace('_Real', ''): round(v * 100, 2)
                       for k, v in weights.items()}
    }).sort_values('Weight (%)', ascending=True)
    w_df = w_df[w_df['Weight (%)'] > 0]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#1565c0' if v >= 25 else '#2e7d32' if v >= 10 else '#ef6c00'
              for v in w_df['Weight (%)']]
    bars = ax.barh(w_df.index, w_df['Weight (%)'], color=colors)
    for bar, val in zip(bars, w_df['Weight (%)']):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(val + 0.5, y, f'{val:.1f}%', va='center', ha='left',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Portfolio Weight (%)')
    ax.set_title('Markowitz Optimal Portfolio Weights')
    ax.set_xlim(0, max(w_df['Weight (%)']) + 5)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(w_df.sort_values('Weight (%)', ascending=False),
                 use_container_width=True)


elif analiz == "11. Rebalancing vs Buy & Hold":
    st.title("11. Rebalancing vs Buy & Hold")

    weights = pd.Series([1/8] * 8, index=real_cols)
    returns = monthly.pct_change().dropna()

    def bh(returns, weights):
        port = (returns * weights).sum(axis=1)
        return (1 + port).cumprod()

    def rb(returns, weights, freq='YE'):
        port = pd.Series(index=returns.index, dtype=float)
        current_w = weights.copy()
        rebal = returns.resample(freq).last().index
        for date, row in returns.iterrows():
            port[date] = (row * current_w).sum()
            current_w = current_w * (1 + row)
            current_w = current_w / current_w.sum()
            if date in rebal:
                current_w = weights.copy()
        return (1 + port).cumprod()

    bh_curve = bh(returns, weights)
    rb_curve = rb(returns, weights, freq='YE')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bh_curve.index, y=bh_curve.values, mode='lines',
        name='Buy & Hold',
        line=dict(width=2.5, color='#1565c0'),
        hovertemplate="<b>Buy & Hold</b><br>Tarix: %{x|%Y-%m-%d}<br>Dəyər: $%{y:,.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=rb_curve.index, y=rb_curve.values, mode='lines',
        name='Yearly Rebalanced',
        line=dict(width=2.5, color='#2e7d32'),
        hovertemplate="<b>Rebalanced</b><br>Tarix: %{x|%Y-%m-%d}<br>Dəyər: $%{y:,.4f}<extra></extra>"
    ))
    fig.update_layout(
        title="Buy & Hold vs Yearly Rebalancing",
        xaxis_title="Tarix", yaxis_title="Growth of $1 (Real)",
        hovermode="x unified", height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    def stats(series):
        cagr_v = series.iloc[-1] ** (12 / len(series)) - 1
        vol = series.pct_change().std() * np.sqrt(12)
        dd = (series / series.cummax() - 1).min()
        return pd.Series({
            'Total Return (%)': round((series.iloc[-1] - 1) * 100, 2),
            'CAGR (%)': round(cagr_v * 100, 2),
            'Volatility (%)': round(vol * 100, 2),
            'Max Drawdown (%)': round(dd * 100, 2),
            'Sharpe': round(cagr_v / vol, 3)
        })

    comp = pd.DataFrame({
        'Buy & Hold': stats(bh_curve),
        'Rebalanced': stats(rb_curve)
    })
    st.dataframe(comp, use_container_width=True)


elif analiz == "12. Yekun Qərar Matrisi":
    st.title("12. Final Investment Decision Matrix")

    cagr_vals = monthly.apply(lambda x: cagr(x))
    vol_annual = daily_returns.std() * np.sqrt(252)
    sharpe = cagr_vals / vol_annual
    max_dd = (data / data.cummax() - 1).min()
    var_95 = daily_returns.quantile(0.05)
    kurt = daily_returns.kurtosis()

    matrix = pd.DataFrame({
        'CAGR (%)': (cagr_vals * 100).round(2),
        'Vol (%)': (vol_annual * 100).round(2),
        'Sharpe': sharpe.round(3),
        'Max DD (%)': (max_dd * 100).round(2),
        'VaR 95% (%)': (var_95 * 100).round(2),
        'Kurtosis': kurt.round(2)
    })

    def score_row(row):
        s = 0
        if row['CAGR (%)'] > 25: s += 3
        elif row['CAGR (%)'] > 15: s += 2.5
        elif row['CAGR (%)'] > 10: s += 2
        elif row['CAGR (%)'] > 5: s += 1.5
        elif row['CAGR (%)'] > 2: s += 1
        elif row['CAGR (%)'] > 0: s += 0.5
        else: s -= 1

        if row['Sharpe'] > 0.8: s += 2.5
        elif row['Sharpe'] > 0.6: s += 2
        elif row['Sharpe'] > 0.4: s += 1.5
        elif row['Sharpe'] > 0.2: s += 1
        elif row['Sharpe'] > 0: s += 0.5
        else: s -= 1

        if row['Max DD (%)'] > -20: s += 1
        elif row['Max DD (%)'] > -35: s += 0.75
        elif row['Max DD (%)'] > -50: s += 0.5
        elif row['Max DD (%)'] > -70: s += 0.25

        if row['VaR 95% (%)'] > -1.5: s += 0.75
        elif row['VaR 95% (%)'] > -3: s += 0.5
        elif row['VaR 95% (%)'] > -5: s += 0.25

        if row['Kurtosis'] < 6: s += 0.75
        elif row['Kurtosis'] < 10: s += 0.5
        elif row['Kurtosis'] < 15: s += 0.25

        return round(s, 2)

    def verdict(s):
        if s >= 6: return 'BUY'
        elif s >= 4.5: return 'HOLD'
        elif s >= 2.5: return 'WATCH'
        else: return 'AVOID'

    matrix['Score'] = matrix.apply(score_row, axis=1)
    matrix['Verdict'] = matrix['Score'].apply(verdict)
    matrix = matrix.sort_values('Score', ascending=False)
    matrix.index = [c.replace('_Real', '') for c in matrix.index]

    st.dataframe(matrix, use_container_width=True)


elif analiz == "Simulyator":
    st.title("🎯 İnvestisiya Simulyatoru")
    st.markdown("### $1,000 büdcə ilə keçmişdə yatırım etsəydin nə olardı?")
    st.markdown("""
    **Necə işləyir:**
    1. Başlanğıc və bitmə tarixini seç (2016–2026 aralığında)
    2. Hər aktive nə qədər pul qoyacağını yaz (cəmi $1,000)
    3. Sistem inflyasiya düzəlişli real qiymətlərlə simulyasiya edir
    4. Nəticədə həmin tarixlərdə nə qədər pulun olacağını görürsən
    """)

    st.markdown("---")

    min_date = data.index.min().date()
    max_date = data.index.max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Başlanğıc tarix", value=min_date,
                                    min_value=min_date, max_value=max_date,
                                    key="sim_start")
    with col2:
        end_date = st.date_input("Bitmə tarixi", value=max_date,
                                  min_value=min_date, max_value=max_date,
                                  key="sim_end")

    if start_date >= end_date:
        st.error("Bitmə tarixi başlanğıcdan sonra olmalıdır.")
        st.stop()

    st.markdown("---")
    st.markdown("### Büdcəni paylaş: $1,000")

    amounts = {}
    cols = st.columns(4)
    for i, name in enumerate(names):
        with cols[i % 4]:
            amounts[name] = st.number_input(
                f"{name} ($)", min_value=0.0, max_value=1000.0,
                value=0.0, step=25.0, key=f"amt_{name}"
            )

    total_invested = sum(amounts.values())

    if total_invested > 1000:
        st.error(f"Cəmi: ${total_invested:.2f} — büdcəni aşır! Maksimum $1,000.")
        st.stop()
    elif total_invested == 0:
        st.warning("Ən azı bir aktive pul ayır.")
        st.stop()
    else:
        st.success(f"Cəmi yatırım: ${total_invested:.2f} | Qalıq: ${1000 - total_invested:.2f}")

    st.markdown("---")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    start_prices = data.loc[data.index >= start_ts].iloc[0]
    end_prices = data.loc[data.index <= end_ts].iloc[-1]

    results = []
    for name in names:
        col = name + '_Real'
        amt = amounts[name]
        if amt > 0:
            units = amt / start_prices[col]
            final_val = units * end_prices[col]
            results.append({
                'Aktiv': name,
                'Yatırılan ($)': round(amt, 2),
                'Başlanğıc qiymət ($)': round(start_prices[col], 2),
                'Bitmə qiymət ($)': round(end_prices[col], 2),
                'Vahid sayı': round(units, 4),
                'Son dəyər ($)': round(final_val, 2),
                'Qazanc/İtki ($)': round(final_val - amt, 2),
                'Gəlir (%)': round((final_val / amt - 1) * 100, 2)
            })

    res_df = pd.DataFrame(results)

    st.markdown("### Nəticə cədvəli")
    st.dataframe(res_df, use_container_width=True)

    final_total = res_df['Son dəyər ($)'].sum()
    total_gain = final_total - total_invested
    return_pct = (final_total / total_invested - 1) * 100

    st.markdown("### Yekun göstəricilər")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yatırılan", f"${total_invested:,.2f}")
    c2.metric("Son dəyər", f"${final_total:,.2f}")
    c3.metric("Qazanc/İtki", f"${total_gain:,.2f}", delta=f"{return_pct:.2f}%")
    c4.metric("Ümumi gəlir", f"{return_pct:.2f}%")

    st.markdown("---")
    st.markdown("### Portfelin zamanla dəyişimi")

    mask = (data.index >= start_ts) & (data.index <= end_ts)
    period = data.loc[mask]

    fig = go.Figure()

    for name in names:
        col = name + '_Real'
        amt = amounts[name]
        if amt > 0:
            units = amt / start_prices[col]
            series = period[col] * units
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values,
                mode='lines', name=name,
                line=dict(width=2),
                hovertemplate=f"<b>{name}</b><br>Tarix: %{{x|%Y-%m-%d}}<br>Dəyər: $%{{y:,.2f}}<extra></extra>"
            ))

    total_series = pd.Series(0.0, index=period.index)
    for name in names:
        col = name + '_Real'
        amt = amounts[name]
        if amt > 0:
            units = amt / start_prices[col]
            total_series += period[col] * units

    fig.add_trace(go.Scatter(
        x=total_series.index, y=total_series.values,
        mode='lines', name='Portfel cəmi',
        line=dict(width=3, color='black'),
        hovertemplate="<b>Portfel cəmi</b><br>Tarix: %{x|%Y-%m-%d}<br>Dəyər: $%{y:,.2f}<extra></extra>"
    ))

    fig.add_hline(y=total_invested, line_dash="dash",
                  line_color="gray", opacity=0.7,
                  annotation_text=f"Yatırılan: ${total_invested:,.2f}")

    fig.update_layout(
        title=f"Portfel Simulyasiyası ({start_date} → {end_date})",
        xaxis_title="Tarix", yaxis_title="Portfel dəyəri ($)",
        hovermode="x unified", height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Nəticələr bizə nə deyir?")
    st.markdown(f"""
    - **Yatırdığın pul:** ${total_invested:,.2f}
    - **Bugünkü dəyər:** ${final_total:,.2f}
    - **Fərq:** ${total_gain:,.2f} ({return_pct:.2f}%)

    Bu simulyasiya **real (inflyasiya düzəlişli)** qiymətlərlə hesablanıb — yəni
    nəticə **aldadıcı nominal rəqəm deyil**, həqiqi alıcılıq gücüdür.
    """)


elif analiz == "ML":
    st.title("🤖 ML Model — Qiymət Proqnozu")
    st.markdown("### XGBoost + Cross-Asset Features (5-Day Horizon)")

    st.markdown("""
    Bu layihədə biz **3 aktiv üzərində** maşın öyrənməsi modelini qurduq:
    **EURUSD, Apple və Gold**. Hər biri üçün ayrı-ayrı model təlim edildi və
    nəticələr dürüst şəkildə qiymətləndirildi.

    **Niyə bu 3 aktiv?**
    - **EURUSD** — ən stabil valyuta cütü, yüksək likvidlik
    - **Apple** — texnologiya lideri, yaxşı data keyfiyyəti
    - **Gold** — qlobal sığorta aktivi, dəyişməz tələb
    """)

    st.markdown("---")

    st.markdown("## 1. Model Haqqında Məlumat")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 Target")
        st.markdown("""
        **5 günlük qiymət istiqaməti**

        - `1` = 5 gündə qiymət qalxdı
        - `0` = 5 gündə qiymət düşdü

        **Niyə 5 gün?** 1 gün çox səs-küylüdür, 5 gün daha stabil siqnal verir.
        """)

    with col2:
        st.markdown("### 🧠 Model")
        st.markdown("""
        **XGBoost Classifier**

        - `n_estimators`: 300
        - `max_depth`: 3
        - `learning_rate`: 0.03
        - `min_child_weight`: 20
        - `reg_alpha`: 1.0
        - `reg_lambda`: 2.0
        - `subsample`: 0.7
        - `colsample_bytree`: 0.7
        """)

    with col3:
        st.markdown("### 📊 Split")
        st.markdown("""
        **80% Train / 20% Test**

        - **Xronoloji split** (gələcək məlumat yoxdur)
        - Train: 2016 – 2023
        - Test: 2024 – 2026

        **Overfit qorunması:** train/test fərqi <3%
        """)

    st.markdown("---")

    st.markdown("## 2. İstifadə Edilən Feature-lar")

    st.markdown("### Öz texniki göstəriciləri")

    feat_df = pd.DataFrame({
        'Kateqoriya': [
            'Qiymət gəlirləri', 'Qiymət gəlirləri', 'Qiymət gəlirləri',
            'Volatillik', 'Volatillik', 'Volatillik',
            'Trend', 'Trend', 'Trend',
            'Texniki indikator', 'Texniki indikator', 'Texniki indikator',
            'Mövqe', 'Mövqe',
            'Təqvim', 'Təqvim', 'Təqvim'
        ],
        'Feature': [
            'ret_1', 'ret_5', 'ret_10',
            'vol_20', 'vol_60', 'vol_regime',
            'price_vs_ma20', 'price_vs_ma60', 'price_vs_ma200',
            'rsi_14', 'macd_norm', 'bb_position',
            'dist_high_252', 'dist_low_252',
            'day_of_week', 'month', 'quarter'
        ],
        'Nə ölçür': [
            '1 günlük gəlir', '5 günlük gəlir', '10 günlük gəlir',
            '20 günlük volatillik', '60 günlük volatillik', 'Volatillik rejimi',
            'MA20-dan uzaqlıq', 'MA60-dan uzaqlıq', 'MA200-dan uzaqlıq',
            'RSI indikatoru', 'MACD (normallaşdırılmış)', 'Bollinger Band mövqeyi',
            '52 həftəlik zirvədən uzaqlıq', '52 həftəlik dibdən uzaqlıq',
            'Həftənin günü', 'Ay', 'Rüb'
        ]
    })

    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("### Cross-Asset Feature-lar")

    st.markdown("""
    Hər aktiv üçün **digər 2 aktivin hərəkəti** də feature kimi əlavə edildi:
    - **EURUSD** üçün: Apple və Gold-un 5/20 günlük gəlirləri
    - **Apple** üçün: EURUSD və Gold-un 5/20 günlük gəlirləri
    - **Gold** üçün: EURUSD və Apple-ın 5/20 günlük gəlirləri

    Bu, modelə **qlobal bazar kontekstini** verir — bir aktiv digərlərinin hərəkətini görür.
    """)

    st.markdown("---")

    st.markdown("## 3. Model Nəticələri — 3 Aktiv")

    st.markdown("### 🥇 EURUSD — Ən Yüksək Dəqiqlik")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HC Dəqiqlik", "61.84%", delta="+10.21% vs baseline")
    col2.metric("Ümumi Dəqiqlik", "58.39%", delta="+6.76% vs baseline")
    col3.metric("Əhatə (Coverage)", "61.7%")
    col4.metric("AUC", "0.633")

    st.markdown("""
    **Nəticə:** EURUSD ən yüksək HC dəqiqliyinə malikdir. Baseline 51.63%,
    model bunu **10.21% yaxşılaşdırır**. AUC 0.633 — real siqnal var.
    """)

    st.markdown("---")

    st.markdown("### 🥈 Apple — Ən Yüksək Gəlir")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HC Dəqiqlik", "61.33%", delta="+5.56% vs baseline")
    col2.metric("Ümumi Dəqiqlik", "57.30%", delta="+1.53% vs baseline")
    col3.metric("Əhatə (Coverage)", "65.4%")
    col4.metric("Orta Gəlir", "+1.16%")

    st.markdown("""
    **Nəticə:** Apple ən yüksək gəlirli aktivdir — model "al" siqnalı verdikdə,
    5 gündə orta **+1.16%** qazanc gəlir. AUC 0.554.
    """)

    st.markdown("---")

    st.markdown("### 🥉 Gold — Sığorta Rolu")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HC Dəqiqlik", "60.15%", delta="+0.24% vs baseline")
    col2.metric("Ümumi Dəqiqlik", "54.03%")
    col3.metric("Əhatə (Coverage)", "62.3%")
    col4.metric("Orta Gəlir", "+0.77%")

    st.markdown("""
    **Nəticə:** Gold modeli stabil nəticə verir. Baseline 59.91% olsa da,
    HC dəqiqlik **60.15%**-dir. Orta gəlir +0.77% — sığorta aktivi kimi dəyərli.
    """)

    st.markdown("---")

    st.markdown("## 4. Müqayisə Cədvəli")

    top3_data = pd.DataFrame({
        'Aktiv': ['EURUSD', 'Apple', 'Gold'],
        'Baseline (%)': [51.63, 55.77, 59.91],
        'Ümumi Dəqiqlik (%)': [58.39, 57.30, 54.03],
        'HC Dəqiqlik (%)': [61.84, 61.33, 60.15],
        'Əhatə (%)': [61.7, 65.4, 62.3],
        'Orta Gəlir (%)': [0.24, 1.16, 0.77],
        'AUC': [0.633, 0.554, 0.578]
    })

    st.dataframe(top3_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("## 5. Vizual Müqayisə")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Baseline',
        y=top3_data['Aktiv'],
        x=top3_data['Baseline (%)'],
        orientation='h',
        marker=dict(color='#90a4ae'),
        text=[f'{v:.2f}%' for v in top3_data['Baseline (%)']],
        textposition='inside'
    ))

    fig.add_trace(go.Bar(
        name='HC Dəqiqlik',
        y=top3_data['Aktiv'],
        x=top3_data['HC Dəqiqlik (%)'],
        orientation='h',
        marker=dict(color='#2e7d32'),
        text=[f'{v:.2f}%' for v in top3_data['HC Dəqiqlik (%)']],
        textposition='outside'
    ))

    fig.add_vline(x=60, line_dash='dash', line_color='red',
                  annotation_text='60% hədd', opacity=0.7)

    fig.update_layout(
        title='HC Dəqiqlik vs Baseline — 3 Aktiv',
        xaxis_title='Dəqiqlik (%)',
        barmode='group',
        height=400,
        margin=dict(l=100, r=100),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("## 6. Orta Göstəricilər (3 Aktiv)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Orta HC Dəqiqlik", f"{top3_data['HC Dəqiqlik (%)'].mean():.2f}%")
    col2.metric("Orta Əhatə", f"{top3_data['Əhatə (%)'].mean():.1f}%")
    col3.metric("Orta Gəlir (Buy)", f"+{top3_data['Orta Gəlir (%)'].mean():.2f}%")
    col4.metric("Orta AUC", f"{top3_data['AUC'].mean():.3f}")

    st.markdown("---")

    st.markdown("## 7. Metrikaların İzahı")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### HC Dəqiqlik (High-Confidence Accuracy)

        Model yalnız **əmin olduğu** hallarda nə qədər düzgün proqnoz verir.

        **Nümunə:** EURUSD üçün 61.84% — yəni model "əminəm" dedikdə
        100 proqnozdan **62-i düzgündür**.
        """)

        st.markdown("""
        ### Əhatə (Coverage)

        Model nə qədər haldan ticarət edir.

        **Nümunə:** Apple üçün 65.4% — yəni 100 gündən **65-də qərar verir**,
        digər 35-də "sabit qal" deyir.
        """)

    with col2:
        st.markdown("""
        ### Orta Gəlir (Avg Buy Return)

        Model "al" siqnalı verdikdə, 5 gündə orta gəlir.

        **Nümunə:** Apple üçün +1.16% — hər "al" siqnalı 5 gündə
        orta hesabla **1.16% qazanc** gətirir.
        """)

        st.markdown("""
        ### AUC (Area Under Curve)

        Model doğru/yanlış proqnozları nə qədər yaxşı ayırd edir.

        - 0.5 = sikkə atmaq
        - 0.6 = yaxşı model ✅
        - 0.7+ = əla (maliyyədə nadir)
        """)

    st.markdown("---")

    st.markdown("## 8. Nəticə")

    st.success("""
    ### ✅ Əsas Tapıntı

    Biz **3 aktiv üzərində** ML modelini qurduq və hər üçü **60%+ HC dəqiqlik**
    göstərdi:

    1. **EURUSD: 61.84%** — ən stabil valyuta cütü
    2. **Apple: 61.33%** — ən yüksək gəlirli (+1.16%)
    3. **Gold: 60.15%** — sığorta aktivi

    Bu nəticələr **dürüst, hiyləsiz və overfit olmayan** real göstəricilərdir.
    """)

    st.info("""
    ### 🔒 Dürüstlük Qeydi

    - **Train/test fərqi < 3%** — overfit yoxdur
    - **Xronoloji split** — gələcək məlumat görünmür
    - **Cross-asset features** — digər aktivlərin hərəkəti (keçmiş)
    - **HC strategiyası** — yalnız əmin olduqda ticarət

    Model hər zaman ticarət etmir — yalnız **~63% halda** qərar verir.
    """)


elif analiz == "Dataset":
    st.title("📁 Dataset")
    st.markdown("### Layihənin əsas dataseti")

    st.markdown("---")

    raw_file = RAW_DIR / "Merged_AllAssets_Real_CPI.xlsx"

    if raw_file.exists():
        raw_df = pd.read_excel(raw_file)

        col1, col2, col3 = st.columns(3)
        col1.metric("Sətir sayı", f"{len(raw_df):,}")
        col2.metric("Sütun sayı", f"{len(raw_df.columns)}")
        col3.metric("Fayl ölçüsü", f"{raw_file.stat().st_size / 1024:.1f} KB")

        st.markdown(f"**Fayl:** `{raw_file.name}`")

        st.markdown("---")
        st.markdown("### Sütun seçimi")

        default_cols = ['date', 'Inflation_Index', 'Gold_Nominal', 'Gold_Real',
                        'Apple_Nominal', 'Apple_Real']

        show_cols = st.multiselect(
            "İstədiyin sütunları seç:",
            options=raw_df.columns.tolist(),
            default=default_cols,
            key="raw_cols_select"
        )

        st.markdown("---")

        if show_cols:
            st.markdown(f"### Data ({len(show_cols)} sütun)")

            n_rows = st.slider("Neçə sətir göstərilsin?",
                               min_value=10, max_value=len(raw_df),
                               value=100, step=10, key="n_rows")

            st.dataframe(raw_df[show_cols].head(n_rows), use_container_width=True)
            st.caption(f"Göstərilir: {n_rows} / {len(raw_df):,} sətir")

            st.markdown("### Statistika")
            st.dataframe(raw_df[show_cols].describe().round(4), use_container_width=True)

            csv = raw_df[show_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Seçilmiş sütunları CSV kimi yüklə",
                data=csv,
                file_name="dataset_selected.csv",
                mime="text/csv",
                key="download_selected"
            )
        else:
            st.warning("Ən azı bir sütun seç.")

        st.markdown("---")
        st.markdown("### Dataset haqqında")

        st.markdown(f"""
        **{raw_file.name}** — Layihənin əsas dataseti

        - **Dövr:** 2016-01-01 → 2026-07-23
        - **Aktivlər:** 8 (3 səhm, 3 valyuta, 2 metal)
        - **Sütunlar:** nominal qiymətlər + real (inflyasiya düzəlişli) qiymətlər
        - **İnflyasiya:** CPI göstəricisi ilə təmizlənib
        """)

    else:
        st.error(f"Dataset tapılmadı: {raw_file}")