import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

NSE_SECTORS = {
    "NIFTY_BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "AUBANK.NS", "BANDHANBNK.NS"],
    "NIFTY_PHARMA": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS", "AUROPHARMA.NS", "BIOCON.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS", "LAURUSLABS.NS", "GLENMARK.NS"],
    "NIFTY_AUTO": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BOSCHLTD.NS", "BALKRISIND.NS", "MRF.NS"],
    "NIFTY_IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "COFORGE.NS", "PERSISTENT.NS", "MPHASIS.NS", "LTTS.NS"],
    "NIFTY_METAL": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS", "NMDC.NS", "SAIL.NS", "JINDALSTEL.NS", "APLAPOLLO.NS", "HINDZINC.NS", "NATIONALUM.NS"],
    "NIFTY_FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS", "UBL.NS", "MCDOWELL-N.NS", "VBL.NS"],
    "NIFTY_REALTY": ["DLF.NS", "MACROTECH.NS", "GODREJPROP.NS", "PRESTIGE.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "BRIGADE.NS", "SOBHA.NS", "MAHLIFE.NS"]
}

def calculate_ema_breakout(df: pd.DataFrame, regime: str) -> bool:
    if regime == "True_Bearish_Distribution" or len(df) < 50: return False
    
    is_high_conviction = (regime == "Choppy_or_Steady_Consolidation")
    max_band_width = 0.025 if is_high_conviction else 0.04
    vol_expansion = 2.0 if is_high_conviction else 1.5

    df['EMA_Fast'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['Vol_20SMA'] = df['Volume'].rolling(window=20).mean()

    recent_closes = df['Close'].iloc[-9:-1]
    recent_fast = df['EMA_Fast'].iloc[-9:-1]
    if (abs(recent_closes - recent_fast) / recent_fast).max() > max_band_width: return False

    return (df['EMA_Fast'].iloc[-1] > df['EMA_Slow'].iloc[-1]) and \
           (df['Close'].iloc[-1] > df['EMA_Fast'].iloc[-1]) and \
           (df['Close'].iloc[-2] <= df['EMA_Fast'].iloc[-2]) and \
           (df['Volume'].iloc[-1] > (df['Vol_20SMA'].iloc[-1] * vol_expansion))

def main():
    try:
        with open("sector_regimes.json", "r") as f:
            regimes = json.load(f)["sectors"]
    except FileNotFoundError:
        print("sector_regimes.json not found.")
        return

    all_signals = {}
    for sector, tickers in NSE_SECTORS.items():
        sector_regime = regimes.get(sector, {}).get("actionable_regime", "Choppy_or_Steady_Consolidation")
        if sector_regime == "True_Bearish_Distribution": continue

        data = yf.download(tickers, period="3mo", group_by="ticker", auto_adjust=True, progress=False)
        signals = []
        for ticker in tickers:
            try:
                df = data[ticker].dropna() if len(tickers) > 1 else data.dropna()
                if not df.empty and calculate_ema_breakout(df, sector_regime):
                    signals.append({"Ticker": ticker.replace(".NS", ""), "Close": round(df['Close'].iloc[-1], 2)})
            except Exception: pass
        
        if signals: all_signals[sector] = signals

    with open("sector_signals.json", "w") as f:
        json.dump({"scan_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), "signals": all_signals}, f, indent=2)

if __name__ == "__main__":
    main()
