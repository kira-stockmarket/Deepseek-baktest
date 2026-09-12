import glob
import json
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

def main():
    print("Initializing Meta-Learner Matrix Stacking...")
    
    csv_files = glob.glob("all_nodes/**/*.csv", recursive=True)
    json_files = glob.glob("all_nodes/**/*.json", recursive=True)
    
    if not csv_files:
        print("No node data found. Did matrix_workers run?")
        return

    node_dataframes = []
    for f in csv_files:
        node_name = os.path.basename(f).replace("_preds.csv", "")
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df = df.rename(columns={"P0": f"{node_name}_P0", "P1": f"{node_name}_P1", "P2": f"{node_name}_P2"})
        node_dataframes.append(df)
    
    meta_df = pd.concat(node_dataframes, axis=1, sort=False).dropna()
    target = meta_df.iloc[:, meta_df.columns.get_loc("True_Target")].iloc[:, 0] if isinstance(meta_df["True_Target"], pd.DataFrame) else meta_df["True_Target"]
    
    feature_cols = [c for c in meta_df.columns if "True_Target" not in c]
    X_meta = meta_df[feature_cols]
    y_meta = target.astype(int)
    
    stacker = LogisticRegression(max_iter=1000)
    stacker.fit(X_meta, y_meta)
    
    regime_map = {
        0: "True_Bearish_Distribution",
        1: "Choppy_or_Steady_Consolidation",
        2: "High_Momentum_Bull_Breakout"
    }

    # --- NEW: GENERATE HISTORICAL BACKTEST DATABASE ---
    print("Generating historical predictions for backtesting...")
    historical_preds = stacker.predict(X_meta)
    
    history_df = pd.DataFrame(index=X_meta.index)
    history_df["Master_Regime_ID"] = historical_preds
    history_df["Actionable_Regime_Name"] = history_df["Master_Regime_ID"].map(regime_map)
    
    # Download Nifty returns to align with our predictions
    nifty = yf.download("^NSEI", period="max", auto_adjust=True, progress=False)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty["Returns"] = nifty["Close"].pct_change()
    
    # Merge and save
    backtest_df = history_df.join(nifty[["Close", "Returns"]]).dropna()
    backtest_df.to_csv("master_ensemble_history.csv")
    # --------------------------------------------------

    today_votes = {}
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            node_name = f"{data['model']}_{data['lookback']}"
            today_votes[f"{node_name}_P0"] = data["today_probs"][0]
            today_votes[f"{node_name}_P1"] = data["today_probs"][1]
            today_votes[f"{node_name}_P2"] = data["today_probs"][2]
            
    today_df = pd.DataFrame([today_votes])[feature_cols]
    master_probs = stacker.predict_proba(today_df)[0]
    master_class = int(np.argmax(master_probs))
    
    output = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "matrix_nodes_reporting": len(csv_files),
        "master_actionable_regime": regime_map[master_class],
        "master_confidence": round(float(master_probs[master_class]), 4),
        "probabilities": {
            regime_map[0]: round(float(master_probs[0]), 4),
            regime_map[1]: round(float(master_probs[1]), 4),
            regime_map[2]: round(float(master_probs[2]), 4)
        }
    }

    with open("current_regime.json", "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Master Actionable Signal: {regime_map[master_class]}")

if __name__ == "__main__":
    main()
