import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

NSE_SECTORS = {
    "NIFTY_BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "AUBANK.NS", "BANDHANBNK.NS"],
    "NIFTY_PHARMA": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS", "AUROPHARMA.NS", "BIOCON.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS", "LAURUSLABS.NS", "GLENMARK.NS"],
    "NIFTY_AUTO": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BOSCHLTD.NS", "BALKRISIND.NS", "MRF.NS"],
    "NIFTY_IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "COFORGE.NS", "PERSISTENT.NS", "MPHASIS.NS", "LTTS.NS"],
    "NIFTY_METAL": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS", "NMDC.NS", "SAIL.NS", "JINDALSTEL.NS", "APLAPOLLO.NS", "HINDZINC.NS", "NATIONALUM.NS"],
    "NIFTY_FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS", "UBL.NS", "MCDOWELL-N.NS", "VBL.NS"],
    "NIFTY_REALTY": ["DLF.NS", "MACROTECH.NS", "GODREJPROP.NS", "PRESTIGE.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "BRIGADE.NS", "SOBHA.NS", "MAHLIFE.NS"]
}
TICKER_TO_SECTOR = {ticker: sector for sector, tickers in NSE_SECTORS.items() for ticker in tickers}
TICKERS = list(TICKER_TO_SECTOR.keys())

def calculate_metrics(series: pd.Series, equity: pd.Series) -> dict:
    ann_ret = series.mean() * 252
    ann_vol = series.std() * np.sqrt(252)
    return {
        "CAGR": round(float(ann_ret), 4),
        "Sharpe_Ratio": round(float(ann_ret / ann_vol if ann_vol > 0 else 0), 4),
        "Max_Drawdown": round(float(((equity / equity.expanding(1).max()) - 1).min()), 4)
    }

def main():
    try:
        # Tries to use the new multi-sector history, falls back to master history if starting fresh
        regime_df = pd.read_csv("sector_ensemble_history.csv", index_col=0, parse_dates=True)
        use_sector = True
    except FileNotFoundError:
        try:
            regime_df = pd.read_csv("master_ensemble_history.csv", index_col=0, parse_dates=True)
            use_sector = False
        except FileNotFoundError: return

    data = yf.download(TICKERS, start=regime_df.index.min(), end=regime_df.index.max(), group_by="ticker", auto_adjust=True, progress=False)
    closes, volumes = pd.DataFrame(), pd.DataFrame()
    for t in TICKERS:
        if t in data and not data[t].empty: closes[t], volumes[t] = data[t]["Close"], data[t]["Volume"]

    common_dates = regime_df.index.intersection(closes.index).sort_values()
    closes, volumes = closes.reindex(common_dates).ffill(), volumes.reindex(common_dates).ffill()
    
    returns, active, pnl = [], [], []

    for i in range(25, len(common_dates)):
        day_ret = 0.0
        surviving = []
        for pos in active:
            c, p = closes[pos["ticker"]].iloc[i], closes[pos["ticker"]].iloc[i - 1]
            day_ret += ((c / p) - 1) / 12
            ret = (c / pos["entry"]) - 1
            if ret <= -0.04 or (i - pos["idx"]) >= 5: pnl.append(ret)
            else: surviving.append(pos)
        active = surviving
        returns.append(day_ret)

        if len(active) >= 12: continue
        
        candidates = []
        for t in closes.columns:
            if any(p["ticker"] == t for p in active): continue
            
            # Fetch the specific regime for this stock's sector
            if use_sector and TICKER_TO_SECTOR[t] in regime_df.columns:
                regime = regime_df.loc[common_dates[i-1], TICKER_TO_SECTOR[t]]
            else:
                regime = regime_df.loc[common_dates[i-1], "Actionable_Regime_Name"] if not use_sector else "Choppy_or_Steady_Consolidation"
                
            if regime == "True_Bearish_Distribution": continue
            
            c_hist, v_hist = closes[t].iloc[:i], volumes[t].iloc[:i]
            if len(c_hist) < 25: continue

            fast, slow, vsma = c_hist.ewm(span=10).mean(), c_hist.ewm(span=21).mean(), v_hist.rolling(20).mean()
            band = 0.025 if regime == "Choppy_or_Steady_Consolidation" else 0.04
            
            if (abs(c_hist.iloc[-9:-1] - fast.iloc[-9:-1]) / fast.iloc[-9:-1]).max() > band: continue

            if (fast.iloc[-1] > slow.iloc[-1]) and (c_hist.iloc[-1] > fast.iloc[-1]) and (c_hist.iloc[-2] <= fast.iloc[-2]) and (v_hist.iloc[-1] > vsma.iloc[-1] * (2.0 if band == 0.025 else 1.5)):
                candidates.append((t, v_hist.iloc[-1] / (vsma.iloc[-1] + 1e-9)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        for t, _ in candidates[:12 - len(active)]:
            active.append({"ticker": t, "entry": closes[t].iloc[i], "idx": i})

    s_series = pd.Series(returns, index=common_dates[25:])
    b_series = closes.loc[common_dates[25:]].pct_change().mean(axis=1).fillna(0)
    
    s_eq, b_eq = (1 + s_series).cumprod(), (1 + b_series).cumprod()
    s_met, b_met = calculate_metrics(s_series, s_eq), calculate_metrics(b_series, b_eq)

    with open("sector_backtest_report.json", "w") as f:
        json.dump({"trades": len(pnl), "win_rate": round(len([p for p in pnl if p>0])/len(pnl)*100,2) if pnl else 0, "strategy": s_met, "benchmark": b_met}, f, indent=2)

    # Generated Plot with Matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(b_eq.index, b_eq, label=f"Equal-Weight Benchmark (Sharpe: {b_met['Sharpe_Ratio']})", color="#7f8c8d", alpha=0.6, linewidth=1.2)
    plt.plot(s_eq.index, s_eq, label=f"140-Node Strategy (Sharpe: {s_met['Sharpe_Ratio']})", color="#2ecc71", linewidth=1.8)
    plt.title("Multi-Sector Alpha: Regime-Conditioned Strategy vs Benchmark", fontsize=14, pad=12)
    plt.ylabel("Cumulative Growth (Log Scale)")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.2, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sector_backtest_chart.png", dpi=250)

if __name__ == "__main__":
    main()
