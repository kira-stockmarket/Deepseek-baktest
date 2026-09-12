import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Standard Nifty Next 50 Tickers (Yahoo Finance Format)
NIFTY_NEXT_50 = [
    "ABB.NS", "ADANIENSOL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "ATGL.NS", 
    "AMBUJACEM.NS", "AWL.NS", "BAJAJHLDNG.NS", "BANKBARODA.NS", "BEL.NS", 
    "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "DLF.NS", "DABUR.NS", 
    "GAIL.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS", "HEROMOTOCO.NS", 
    "HINDALCO.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IOC.NS", "IRCTC.NS", 
    "JINDALSTEL.NS", "JIOFIN.NS", "KOTAKBANK.NS", "LICI.NS", "MARICO.NS", 
    "MUTHOOTFIN.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS", "SBICARD.NS", 
    "SRF.NS", "MOTHERSON.NS", "SHREECEM.NS", "SIEMENS.NS", "TATAELXSI.NS", 
    "TVSMOTOR.NS", "TORNTPHARM.NS", "TRENT.NS", "UBL.NS", "MCDOWELL-N.NS", 
    "VBL.NS", "VEDL.NS", "ZOMATO.NS", "ZYDUSLIFE.NS", "TATACHEM.NS"
]

def calculate_ema_breakout(df: pd.DataFrame, is_high_conviction: bool) -> bool:
    """
    Evaluates the AI-optimized EMA Consolidation Breakout Strategy.
    Adjusts parameters dynamically based on market regime.
    """
    # AI-Optimized Base Parameters
    ema_fast = 10
    ema_slow = 21
    consolidation_days = 8
    
    # Dynamic Parameters based on Regime
    max_band_width = 0.025 if is_high_conviction else 0.04  # 2.5% vs 4% consolidation band
    vol_expansion = 2.0 if is_high_conviction else 1.5      # 200% vs 150% volume breakout

    if len(df) < 50:
        return False

    df['EMA_Fast'] = df['Close'].ewm(span=ema_fast, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=ema_slow, adjust=False).mean()
    df['Vol_20SMA'] = df['Volume'].rolling(window=20).mean()

    # 1. Consolidation Check: Prices must be tightly bound around EMAs for N days
    recent_closes = df['Close'].iloc[-consolidation_days-1:-1]
    recent_fast = df['EMA_Fast'].iloc[-consolidation_days-1:-1]
    
    # Calculate how far prices strayed from the Fast EMA
    max_deviation = (abs(recent_closes - recent_fast) / recent_fast).max()
    
    if max_deviation > max_band_width:
        return False # Too volatile, no consolidation

    # 2. Breakout Check (Today's Price Action)
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    # Price must break above both EMAs, and Fast EMA must be > Slow EMA
    bullish_alignment = today['EMA_Fast'] > today['EMA_Slow']
    price_breakout = today['Close'] > today['EMA_Fast'] and yesterday['Close'] <= yesterday['EMA_Fast']
    
    # 3. Volume Confirmation
    volume_surge = today['Volume'] > (today['Vol_20SMA'] * vol_expansion)

    return bullish_alignment and price_breakout and volume_surge

def main():
    print("Initiating Hive Mind Regime-Filtered Stock Scanner...\n")
    
    # 1. Read the current Master Regime
    try:
        with open("current_regime.json", "r") as f:
            regime_data = json.load(f)
            master_regime = regime_data.get("master_actionable_regime", "Unknown")
    except FileNotFoundError:
        print("Error: current_regime.json not found. Run meta_ensemble.py first.")
        return

    print(f"Current Market Regime: {master_regime}")

    # 2. Apply Master Logic Overrides
    if master_regime == "True_Bearish_Distribution":
        print("🛑 REGIME OVERRIDE: Bearish phase detected. No buy trades permitted. Capital preservation active.")
        return

    is_high_conviction = (master_regime == "Choppy_or_Steady_Consolidation")
    
    if is_high_conviction:
        print("⚠️ REGIME OVERRIDE: Choppy market. Scanning strictly for HIGH CONVICTION setups only.\n")
    else:
        print("🟢 REGIME OVERRIDE: Bull Breakout market. Scanning with standard aggressive parameters.\n")

    # 3. Scan the Nifty Next 50 Universe
    buy_signals = []
    
    for ticker in NIFTY_NEXT_50:
        try:
            # Download recent data silently
            df = yf.download(ticker, period="3mo", progress=False)
            if df.empty:
                continue
                
            # If multi-index columns, flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            if calculate_ema_breakout(df, is_high_conviction):
                close_price = round(df['Close'].iloc[-1], 2)
                vol_surge = round(df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1], 2)
                
                buy_signals.append({
                    "Ticker": ticker.replace(".NS", ""),
                    "Close": close_price,
                    "Volume_Surge": f"{vol_surge}x"
                })
                print(f"🔥 BREAKOUT DETECTED: {ticker} (Close: ₹{close_price} | Vol: {vol_surge}x Avg)")
                
        except Exception as e:
            pass # Skip failed downloads gracefully

    # 4. Output Results
    print("\n" + "="*50)
    if not buy_signals:
        print("No stocks met the required breakout criteria today.")
    else:
        print(f"TOTAL BUY SIGNALS GENERATED: {len(buy_signals)}")
        print("="*50)
        
    # Save the signals to a JSON file for the dashboard
    output = {
        "scan_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "regime_context": master_regime,
        "high_conviction_mode": is_high_conviction,
        "buy_signals": buy_signals
    }
    
    with open("next50_signals.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
