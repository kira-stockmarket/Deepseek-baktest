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

def train_sector_meta_learner(sector: str, base_path: str = "all_nodes"):
    print(f"\n--- Stacking 20 Nodes for Sector: {sector} ---")
    
    # 1. Collect out-of-sample prediction CSVs
    dfs = []
    today_features = []
    
    for model in MODELS:
        for lb in LOOKBACKS:
            # Pattern matches artifacts downloaded by GitHub Actions
            pattern = f"{base_path}/**/{sector}_{model}_{lb}_preds.csv"
            files = glob.glob(pattern, recursive=True)
            if not files:
                # Local fallback check
                files = glob.glob(f"output/{sector}_{model}_{lb}_preds.csv")
            
            if not files:
                print(f"Warning: Missing predictions for {sector}_{model}_{lb}")
                continue
                
            df = pd.read_csv(files[0], index_col=0, parse_dates=True)
            prob_cols = [f"{sector}_{model}_{lb}_P0", f"{sector}_{model}_{lb}_P1", f"{sector}_{model}_{lb}_P2"]
            df = df.rename(columns={"P0": prob_cols[0], "P1": prob_cols[1], "P2": prob_cols[2]})
            dfs.append(df[prob_cols])

            # Get today's probability vector from metadata JSON
            meta_pattern = f"{base_path}/**/{sector}_{model}_{lb}_meta.json"
            meta_files = glob.glob(meta_pattern, recursive=True) or glob.glob(f"output/{sector}_{model}_{lb}_meta.json")
            if meta_files:
                with open(meta_files[0], "r") as mf:
                    meta = json.load(mf)
                    today_features.extend(meta["today_probs"])

    if not dfs or len(today_features) < (len(MODELS) * len(LOOKBACKS) * 3):
        print(f"Insufficient node files to fit Meta-Learner for {sector}.")
        return None

    merged = pd.concat(dfs, axis=1, join="inner").dropna()
    
    # Get True Target from first available file
    sample_file = glob.glob(f"{base_path}/**/{sector}_*_preds.csv", recursive=True) or glob.glob(f"output/{sector}_*_preds.csv")
    targets = pd.read_csv(sample_file[0], index_col=0, parse_dates=True)["True_Target"]
    merged["Target"] = targets.loc[merged.index]

    X = merged.drop(columns=["Target"])
    y = merged["Target"].astype(int)

    # 2. Fit Sector-Level Meta-Learner
    meta_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    meta_clf.fit(X, y)

    # 3. Infer Today's Regime for this sector
    today_input = np.array(today_features).reshape(1, -1)
    probs = meta_clf.predict_proba(today_input)[0]
    predicted_class = int(np.argmax(probs))

    regime_info = {
        "regime_id": predicted_class,
        "actionable_regime": REGIME_NAMES[predicted_class],
        "confidence": round(float(probs[predicted_class]), 4),
        "class_probabilities": {
            "Bearish": round(float(probs[0]), 4),
            "Choppy": round(float(probs[1]), 4),
            "Bullish": round(float(probs[2]), 4)
        }
    }
    print(f"[{sector}] Consensus: {regime_info['actionable_regime']} ({regime_info['confidence']*100:.1f}% Confidence)")
    return regime_info

def main():
    master_sector_regimes = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_nodes_evaluated": 140,
        "sectors": {}
    }

    for sector in SECTORS:
        res = train_sector_meta_learner(sector)
        if res:
            master_sector_regimes["sectors"][sector] = res

    # Output consolidated master dictionary
    with open("sector_regimes.json", "w") as f:
        json.dump(master_sector_regimes, f, indent=2)
        
    print("\nSaved sector consensus to 'sector_regimes.json'.")

if __name__ == "__main__":
    main()
