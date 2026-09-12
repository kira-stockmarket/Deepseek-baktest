import glob
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression

def main():
    print("Initializing Meta-Learner Matrix Stacking...")
    
    # 1. Gather all Node Data
    csv_files = glob.glob("all_nodes/**/*.csv", recursive=True)
    json_files = glob.glob("all_nodes/**/*.json", recursive=True)
    
    if not csv_files:
        print("No node data found. Did matrix_workers run?")
        return

    # 2. Build the Meta-Training Dataset
    node_dataframes = []
    for f in csv_files:
        node_name = os.path.basename(f).replace("_preds.csv", "")
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        # Rename columns to identify which node voted for what
        df = df.rename(columns={"P0": f"{node_name}_P0", "P1": f"{node_name}_P1", "P2": f"{node_name}_P2"})
        node_dataframes.append(df)
    
    # Merge all 20 nodes together by date
    meta_df = pd.concat(node_dataframes, axis=1).dropna()
    
    # The true outcome is identical across nodes, we just need one column of it
    target = meta_df.iloc[:, meta_df.columns.get_loc("True_Target")].iloc[:, 0] if isinstance(meta_df["True_Target"], pd.DataFrame) else meta_df["True_Target"]
    
    # Features for Meta-Learner = The probabilities output by the 20 worker models
    feature_cols = [c for c in meta_df.columns if "True_Target" not in c]
    X_meta = meta_df[feature_cols]
    y_meta = target.astype(int)
    
    # 3. Train the Meta-Learner (Logistic Regression prevents overfitting at the ensemble layer)
    stacker = LogisticRegression(max_iter=1000, multi_class='multinomial')
    stacker.fit(X_meta, y_meta)
    
    print(f"Meta-Learner trained on {len(X_meta)} Out-Of-Sample days across {len(csv_files)} models.")

    # 4. Gather Today's Votes
    today_votes = {}
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            node_name = f"{data['model']}_{data['lookback']}"
            today_votes[f"{node_name}_P0"] = data["today_probs"][0]
            today_votes[f"{node_name}_P1"] = data["today_probs"][1]
            today_votes[f"{node_name}_P2"] = data["today_probs"][2]
            
    # Format today's row to match the exact column order the Stacker was trained on
    today_df = pd.DataFrame([today_votes])[feature_cols]
    
    # 5. Master Prediction
    master_probs = stacker.predict_proba(today_df)[0]
    master_class = int(np.argmax(master_probs))
    
    regime_map = {
        0: "True_Bearish_Distribution",
        1: "Choppy_or_Steady_Consolidation",
        2: "High_Momentum_Bull_Breakout"
    }
    
    # 6. Output the Master Consensus
    output = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "matrix_nodes_reporting": len(csv_files),
        "master_actionable_regime": regime_map[master_class],
        "master_confidence": round(float(master_probs[master_class]), 4),
        "probabilities": {
            regime_map[0]: round(float(master_probs[0]), 4),
            regime_map[1]: round(float(master_probs[1]), 4),
            regime_map[2]: round(float(master_probs[2]), 4)
        },
        "system_status": "20-Node Hive Mind Optimal"
    }

    with open("current_regime.json", "w") as f:
        json.dump(output, f, indent=2)
        
    print(json.dumps(output, indent=2))
    print(f"Master Actionable Signal: {regime_map[master_class]}")

if __name__ == "__main__":
    main()
