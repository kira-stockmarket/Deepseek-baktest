import argparse
import json
import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")

def fetch_and_prep_data(lookback: int):
    nifty_raw = yf.download("^NSEI", period="max", auto_adjust=True, progress=False)
    vix_raw = yf.download("^INDIAVIX", period="max", auto_adjust=True, progress=False)

    # 1. Safely handle new yfinance MultiIndex output (The Fix)
    if isinstance(nifty_raw.columns, pd.MultiIndex):
        nifty_raw.columns = nifty_raw.columns.get_level_values(0)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)

    # 2. Extract and rename as 1D Series
    nifty = nifty_raw["Close"].rename("NIFTY")
    vix = vix_raw["Close"].rename("VIX")
    
    df = pd.concat([nifty, vix], axis=1).dropna()

    # Features tailored to this node's specific lookback window
    df["Returns"] = df["NIFTY"].pct_change(1)
    df[f"Vol_{lookback}"] = df["Returns"].rolling(lookback).std() * np.sqrt(252)
    df[f"VIX_SMA_{lookback}"] = df["VIX"].rolling(lookback).mean()
    df[f"VIX_Ratio_{lookback}"] = df["VIX"] / (df[f"VIX_SMA_{lookback}"] + 1e-9)
    
    df[f"Momentum_{lookback}"] = df["NIFTY"].pct_change(lookback)
    df[f"SMA_{lookback}"] = df["NIFTY"].rolling(lookback).mean()
    df[f"Dist_SMA_{lookback}"] = (df["NIFTY"] / df[f"SMA_{lookback}"]) - 1

    # TARGET: 3-Class Forward 5-Day Horizon
    # 0 = True Bearish (< -1%), 1 = Chop/Steady (-1% to 1.5%), 2 = Bull Breakout (> 1.5%)
    fwd_ret = df["NIFTY"].shift(-5) / df["NIFTY"] - 1
    df["Target"] = np.where(fwd_ret < -0.01, 0, np.where(fwd_ret > 0.015, 2, 1))

    df = df.dropna()
    features = [c for c in df.columns if c not in ["NIFTY", "VIX", "Target"]]
    return df, features

class FastGRUClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, 32, batch_first=True)
        self.fc = nn.Linear(32, 3)
    def forward(self, x):
        _, h = self.gru(x)
        return torch.softmax(self.fc(h[-1]), dim=1)

def train_gru(X_train, y_train, X_test):
    # Reshape for GRU (Batch, Seq=1, Features)
    X_tr_t = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
    y_tr_t = torch.tensor(y_train.values, dtype=torch.long)
    X_te_t = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
    
    model = FastGRUClassifier(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(15): # Fast training for Matrix Worker
        optimizer.zero_grad()
        out = model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).numpy()
    return preds, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lookback", type=int, required=True)
    args = parser.parse_args()

    df, feature_cols = fetch_and_prep_data(args.lookback)
    
    # Time-series split (Train on past, Test on recent 20% to generate Out-of-Sample predictions for Meta-Learner)
    X = df[feature_cols]
    y = df["Target"]
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training {args.model} with lookback {args.lookback}...")

    if args.model == "xgboost":
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
    elif args.model == "lightgbm":
        clf = lgb.LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
    elif args.model == "catboost":
        clf = cb.CatBoostClassifier(iterations=100, depth=4, random_state=42, verbose=0)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
    elif args.model == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
    elif args.model == "gru":
        preds, clf = train_gru(X_train, y_train, X_test)

    # 1. Save Out-Of-Sample Predictions for the Meta-Learner
    out_df = pd.DataFrame(preds, index=X_test.index, columns=[f"P0", f"P1", f"P2"])
    out_df["True_Target"] = y_test
    
    os.makedirs("output", exist_ok=True)
    out_df.to_csv(f"output/{args.model}_{args.lookback}_preds.csv")

    # 2. Predict Today
    if args.model == "gru":
        with torch.no_grad():
            today_pred = clf(torch.tensor(X.iloc[-1:].values, dtype=torch.float32).unsqueeze(1)).numpy()[0]
    else:
        today_pred = clf.predict_proba(X.iloc[-1:])[-1]

    meta = {
        "model": args.model,
        "lookback": args.lookback,
        "today_probs": [round(float(p), 4) for p in today_pred],
        "val_accuracy": round(accuracy_score(y_test, np.argmax(preds, axis=1)), 4)
    }
    
    with open(f"output/{args.model}_{args.lookback}_meta.json", "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    main()
