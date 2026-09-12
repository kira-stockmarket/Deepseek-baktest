import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

# Official Yahoo Finance Tickers for NSE Sectors
SECTOR_TICKERS = {
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_REALTY": "^CNXREALTY"
}

# The Core Logic: Position Sizing by AI Regime
POSITION_MAP = {
    "High_Momentum_Bull_Breakout": 1.0,      # 100% Long
    "Choppy_or_Steady_Consolidation": 0.5,   # 50% Cash
    "True_Bearish_Distribution": 0.0         # 100% Cash (Avoid Crashes)
}

def calc_metrics(returns: pd.Series) -> dict:
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    equity = (1 + returns).cumprod()
    max_dd = ((equity / equity.expanding(1).max()) - 1).min()
    return {
        "CAGR": round(float(ann_ret), 4),
        "Sharpe": round(float(sharpe), 4),
        "Max_Drawdown": round(float(max_dd), 4)
    }

def main():
    print("Initiating Pure Sector Regime Backtest...")

    # 1. Load the AI's Historical Memory
    try:
        regime_df = pd.read_csv("sector_ensemble_history.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'sector_ensemble_history.csv' not found. Run the Hive Mind first.")
        return
        
    start_date = regime_df.index.min()
    end_date = regime_df.index.max()

    # 2. Download Sector Index Prices
    tickers = list(SECTOR_TICKERS.values())
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
    
    # 3. Calculate Returns for Benchmark vs Strategy
    strat_portfolio_returns = []
    bench_portfolio_returns = []
    
    common_dates = regime_df.index.intersection(raw_data.index).sort_values()
    
    for i in range(1, len(common_dates)):
        curr_date = common_dates[i]
        prev_date = common_dates[i-1]
        
        daily_strat_ret = 0.0
        daily_bench_ret = 0.0
        
        # Loop through all 7 sectors
        for sector_name, ticker in SECTOR_TICKERS.items():
            if ticker not in raw_data.columns:
                continue
                
            prev_price = raw_data[ticker].loc[prev_date]
            curr_price = raw_data[ticker].loc[curr_date]
            
            if pd.isna(prev_price) or pd.isna(curr_price) or prev_price == 0:
                continue
                
            raw_return = (curr_price / prev_price) - 1
            
            # Get yesterday's AI signal to execute today (prevents lookahead bias)
            regime = regime_df[sector_name].loc[prev_date] if sector_name in regime_df.columns else "Choppy_or_Steady_Consolidation"
            position_size = POSITION_MAP.get(regime, 0.5)
            
            # Equal weight allocation (1/7th per sector)
            daily_bench_ret += raw_return / len(SECTOR_TICKERS)
            daily_strat_ret += (raw_return * position_size) / len(SECTOR_TICKERS)
            
        strat_portfolio_returns.append(daily_strat_ret)
        bench_portfolio_returns.append(daily_bench_ret)

    # 4. Generate Performance Curves
    perf_index = common_dates[1:]
    strat_series = pd.Series(strat_portfolio_returns, index=perf_index)
    bench_series = pd.Series(bench_portfolio_returns, index=perf_index)

    strat_eq = (1 + strat_series).cumprod()
    bench_eq = (1 + bench_series).cumprod()
    
    strat_met = calc_metrics(strat_series)
    bench_met = calc_metrics(bench_series)

    # 5. Save JSON Report
    report = {
        "AI_Regime_Strategy": strat_met,
        "Buy_and_Hold_Benchmark": bench_met,
        "Total_Days_Tested": len(perf_index)
    }
    with open("simple_sector_backtest.json", "w") as f:
        json.dump(report, f, indent=2)

    # 6. Generate Clean Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(bench_eq.index, bench_eq, label=f"Buy & Hold Sectors (Sharpe: {bench_met['Sharpe']})", color="#95a5a6", alpha=0.7, linewidth=1.5)
    plt.plot(strat_eq.index, strat_eq, label=f"AI Regime Strategy (Sharpe: {strat_met['Sharpe']})", color="#27ae60", linewidth=2.0)
    
    plt.title("Pure Sector Rotation: AI Regime vs Buy-and-Hold", fontsize=15, pad=12)
    plt.ylabel("Portfolio Value (Log Scale)", fontsize=11)
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.2, linestyle="--")
    plt.legend(loc="upper left", framealpha=0.9, fontsize=11)
    plt.tight_layout()
    plt.savefig("simple_sector_backtest_chart.png", dpi=250)
    
    print("\n--- SIMPLE SECTOR BACKTEST RESULTS ---")
    print(f"Benchmark Sharpe: {bench_met['Sharpe']} | Strategy Sharpe: {strat_met['Sharpe']}")
    print(f"Benchmark Max DD: {bench_met['Max_Drawdown'] * 100:.2f}% | Strategy Max DD: {strat_met['Max_Drawdown'] * 100:.2f}%")

if __name__ == "__main__":
    main()
