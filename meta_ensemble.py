import json
import glob
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SECTORS = [
    "NIFTY_BANK", "NIFTY_PHARMA", "NIFTY_AUTO", 
    "NIFTY_IT", "NIFTY_METAL", "NIFTY_FMCG", "NIFTY_REALTY"
]
MODELS = ["gru", "xgboost", "lightgbm", "catboost", "random_forest"]
LOOKBACKS = [5, 14, 21, 63]

REGIME_NAMES = {
    0: "True_Bearish_Distribution",
    1: "Choppy_or_Steady_Consolidation",
    2: "High_Momentum_Bull_Breakout"
}

def process_sector(sector: str, base_path: str = "all_nodes"):
    dfs = []
    today_features = []

    # 1. Gather all out-of-sample predictions and today's probabilities
    for model in MODELS:
        for lb in LOOKBACKS:
            pattern = f"{base_path}/**/{sector}_{model}_{lb}_preds.csv"
            files = glob.glob(pattern, recursive=True) or glob.glob(f"output/{sector}_{model}_{lb}_preds.csv")
            if not files: continue

            df = pd.read_csv(files[0], index_col=0, parse_dates=True)
            cols = [f"{sector}_{model}_{lb}_P0", f"{sector}_{model}_{lb}_P1", f"{sector}_{model}_{lb}_P2"]
            df = df.rename(columns={"P0": cols[0], "P1": cols[1], "P2": cols[2]})
            dfs.append(df[cols])

            meta_pattern = f"{base_path}/**/{sector}_{model}_{lb}_meta.json"
            meta_files = glob.glob(meta_pattern, recursive=True) or glob.glob(f"output/{sector}_{model}_{lb}_meta.json")
            if meta_files:
                with open(meta_files[0], "r") as mf:
                    today_features.extend(json.load(mf)["today_probs"])

    if not dfs:
        return None, None

    merged = pd.concat(dfs, axis=1, join="inner").dropna()
    sample_file = glob.glob(f"{base_path}/**/{sector}_*_preds.csv", recursive=True) or glob.glob(f"output/{sector}_*_preds.csv")
    targets = pd.read_csv(sample_file[0], index_col=0, parse_dates=True)["True_Target"]
    y = targets.loc[merged.index].astype(int)

    # 2. Fit Meta-Learner
    meta_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    meta_clf.fit(merged, y)

    # 3. Predict Historical Set
    historical_preds = meta_clf.predict(merged)
    historical_regimes = pd.Series([REGIME_NAMES[p] for p in historical_preds], index=merged.index, name=sector)

    # 4. Predict Today's Regime with 5% Filter
    today_regime = None
    if len(today_features) == len(MODELS) * len(LOOKBACKS) * 3:
        today_input = np.array(today_features).reshape(1, -1)
        probs = meta_clf.predict_proba(today_input)[0]
        
        sorted_probs = np.sort(probs)
        pred_class = int(np.argmax(probs))
        if (sorted_probs[-1] - sorted_probs[-2]) < 0.05 and pred_class != 1:
            pred_class = 1 # Default to Choppy if margin is tight

        today_regime = {
            "regime_id": pred_class,
            "actionable_regime": REGIME_NAMES[pred_class],
            "confidence": round(float(probs[pred_class]), 4),
            "class_probabilities": {
                "Bearish": round(float(probs[0]), 4),
                "Choppy": round(float(probs[1]), 4),
                "Bullish": round(float(probs[2]), 4)
            }
        }

    return today_regime, historical_regimes

def main():
    today_sectors = {}
    sector_history_series = []

    for sector in SECTORS:
        today_regime, hist_series = process_sector(sector)
        if today_regime:
            today_sectors[sector] = today_regime
        if hist_series is not None:
            sector_history_series.append(hist_series)

    # Output 1: Live JSON for Web Dashboard
    with open("sector_regimes.json", "w") as f:
        json.dump({"updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), "sectors": today_sectors}, f, indent=2)

    # Output 2: Append-Only History CSV for Backtester
    hist_file = "sector_ensemble_history.csv"
    if sector_history_series:
        new_history_df = pd.concat(sector_history_series, axis=1, join="inner").dropna()
        new_history_df.index.name = "Date"
        
        if os.path.exists(hist_file):
            old_history_df = pd.read_csv(hist_file, index_col="Date", parse_dates=True)
            combined_history = pd.concat([old_history_df, new_history_df])
            combined_history = combined_history[~combined_history.index.duplicated(keep='last')].sort_index()
        else:
            combined_history = new_history_df
            
        today_date = pd.to_datetime("today").normalize()
        today_row = {sector: data["actionable_regime"] for sector, data in today_sectors.items()}
        today_df = pd.DataFrame([today_row], index=[today_date])
        today_df.index.name = "Date"
        
        final_history = pd.concat([combined_history, today_df])
        final_history = final_history[~final_history.index.duplicated(keep='last')].sort_index()
        final_history.to_csv(hist_file)
    else:
        if not os.path.exists(hist_file):
            pd.DataFrame(columns=SECTORS).to_csv(hist_file, index_label="Date")

if __name__ == "__main__":
    main()
