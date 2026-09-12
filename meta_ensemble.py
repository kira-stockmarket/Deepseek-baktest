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
    print(f"Aggregating 20 nodes for {sector}...")
    dfs = []
    today_features = []

    for model in MODELS:
        for lb in LOOKBACKS:
            # 1. Look for predictions CSV
            pattern = f"{base_path}/**/{sector}_{model}_{lb}_preds.csv"
            files = glob.glob(pattern, recursive=True) or glob.glob(f"output/{sector}_{model}_{lb}_preds.csv")
            if not files:
                continue

            df = pd.read_csv(files[0], index_col=0, parse_dates=True)
            cols = [f"{sector}_{model}_{lb}_P0", f"{sector}_{model}_{lb}_P1", f"{sector}_{model}_{lb}_P2"]
            df = df.rename(columns={"P0": cols[0], "P1": cols[1], "P2": cols[2]})
            dfs.append(df[cols])

            # 2. Look for today's meta JSON
            meta_pattern = f"{base_path}/**/{sector}_{model}_{lb}_meta.json"
            meta_files = glob.glob(meta_pattern, recursive=True) or glob.glob(f"output/{sector}_{model}_{lb}_meta.json")
            if meta_files:
                with open(meta_files[0], "r") as mf:
                    today_features.extend(json.load(mf)["today_probs"])

    if not dfs:
        print(f"Warning: No node data found for {sector}")
        return None, None

    # Merge out-of-sample predictions across all 20 nodes
    merged = pd.concat(dfs, axis=1, join="inner").dropna()

    # Retrieve true target from one of the prediction files
    sample_file = glob.glob(f"{base_path}/**/{sector}_*_preds.csv", recursive=True) or glob.glob(f"output/{sector}_*_preds.csv")
    targets = pd.read_csv(sample_file[0], index_col=0, parse_dates=True)["True_Target"]
    y = targets.loc[merged.index].astype(int)
    X = merged

    # Fit the Meta-Learner on out-of-sample features
    meta_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    meta_clf.fit(X, y)

    # --- 1. GENERATE HISTORICAL REGIMES FOR BACKTESTING ---
    # Predict the regime for every single historical day in the test period
    historical_preds = meta_clf.predict(X)
    historical_regimes = pd.Series(
        [REGIME_NAMES[p] for p in historical_preds],
        index=X.index,
        name=sector
    )

    # --- 2. GENERATE TODAY'S REGIME FOR LIVE SCANNING ---
    today_regime = None
    expected_feature_len = len(MODELS) * len(LOOKBACKS) * 3
    if len(today_features) == expected_feature_len:
        today_input = np.array(today_features).reshape(1, -1)
        probs = meta_clf.predict_proba(today_input)[0]
        pred_class = int(np.argmax(probs))
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

    # 1. Save Today's Live Regimes (Used by Stock Scanner)
    master_live_output = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_nodes_evaluated": 140,
        "sectors": today_sectors
    }
    with open("sector_regimes.json", "w") as f:
        json.dump(master_live_output, f, indent=2)
    print("Saved 'sector_regimes.json' for scanner.")

    # 2. Save Complete Multi-Year History (Used by Backtest Engine)
    if sector_history_series:
        history_df = pd.concat(sector_history_series, axis=1, join="inner").dropna()
        history_df.index.name = "Date"
        history_df.to_csv("sector_ensemble_history.csv")
        print(f"Generated 'sector_ensemble_history.csv' with {len(history_df)} historical trading days.")
    else:
        print("Error: Could not generate historical regime CSV.")


if __name__ == "__main__":
    main()
