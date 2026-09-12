import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
from torch import nn

warnings.filterwarnings("ignore")


def calculate_technicals(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Calculates an institutional feature suite tailored to the node's lookback."""
    # 1. Price Returns & Momentum
    df["Returns"] = df["NIFTY"].pct_change(1)
    df[f"Momentum_{lookback}"] = df["NIFTY"].pct_change(lookback)
    
    # 2. Moving Average Distance & Volatility
    sma = df["NIFTY"].rolling(lookback).mean()
    df[f"Dist_SMA_{lookback}"] = (df["NIFTY"] / (sma + 1e-9)) - 1
    df[f"Vol_{lookback}"] = df["Returns"].rolling(lookback).std() * np.sqrt(252)

    # 3. High/Low Volatility proxy (True Range Approximation)
    rolling_max = df["NIFTY"].rolling(lookback).max()
    rolling_min = df["NIFTY"].rolling(lookback).min()
    df[f"Channel_Width_{lookback}"] = (rolling_max - rolling_min) / (sma + 1e-9)

    # 4. VIX Term & Velocity
    vix_sma = df["VIX"].rolling(lookback).mean()
    df[f"VIX_Ratio_{lookback}"] = df["VIX"] / (vix_sma + 1e-9)
    df[f"VIX_Velocity_{lookback}"] = df["VIX"].pct_change(max(2, lookback // 3))

    # 5. Trend Strength Proxy (ADX-style trend vs chop filter)
    abs_momentum = df[f"Momentum_{lookback}"].abs()
    path_length = df["Returns"].abs().rolling(lookback).sum()
    df[f"Trend_Efficiency_{lookback}"] = abs_momentum / (path_length + 1e-9)

    return df


def fetch_and_prep_data(lookback: int):
    nifty_raw = yf.download("^NSEI", period="max", auto_adjust=True, progress=False)
    vix_raw = yf.download("^INDIAVIX", period="max", auto_adjust=True, progress=False)

    if isinstance(nifty_raw.columns, pd.MultiIndex):
        nifty_raw.columns = nifty_raw.columns.get_level_values(0)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)

    nifty = nifty_raw["Close"].rename("NIFTY")
    vix = vix_raw["Close"].rename("VIX")
    df = pd.concat([nifty, vix], axis=1).dropna()

    # Apply expanded feature pipeline
    df = calculate_technicals(df, lookback)

    # TARGET: Balanced Volatility-Responsive Thresholds
    # Using 0.8% threshold for 5-day horizon to capture realistic tradable swings
    fwd_ret = df["NIFTY"].shift(-5) / df["NIFTY"] - 1
    df["Target"] = np.where(fwd_ret < -0.008, 0, np.where(fwd_ret > 0.008, 2, 1))

    df = df.dropna()
    features = [c for c in df.columns if c not in ["NIFTY", "VIX", "Target"]]
    return df, features


class EnhancedGRU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        _, h = self.gru(x)
        return torch.softmax(self.fc(h[-1]), dim=1)


def train_gru(X_train, y_train, X_test, weights):
    X_tr_t = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
    y_tr_t = torch.tensor(y_train.values, dtype=torch.long)
    X_te_t = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)

    model = EnhancedGRU(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    model.train()
    for _ in range(25):
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

    X = df[feature_cols]
    y = df["Target"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Calculate Balanced Class Weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes, weights))

    print(f"Training Upgraded {args.model} ({args.lookback}D) | Class Weights: {np.round(weights, 2)}")

    if args.model == "xgboost":
        sample_weights = y_train.map(weight_dict)
        clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        preds = clf.predict_proba(X_test)

    elif args.model == "lightgbm":
        clf = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1
        )
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)

    elif args.model == "catboost":
        clf = cb.CatBoostClassifier(
            iterations=150,
            depth=5,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_state=42,
            verbose=0
        )
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)

    elif args.model == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            class_weight="balanced_subsample",
            random_state=42
        )
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)

    elif args.model == "gru":
        preds, clf = train_gru(X_train, y_train, X_test, weights)

    # 1. Save Out-Of-Sample Predictions
    out_df = pd.DataFrame(preds, index=X_test.index, columns=["P0", "P1", "P2"])
    out_df["True_Target"] = y_test

    os.makedirs("output", exist_ok=True)
    out_df.to_csv(f"output/{args.model}_{args.lookback}_preds.csv")

    # 2. Predict Today
    if args.model == "gru":
        with torch.no_grad():
            today_input = torch.tensor(X.iloc[-1:].values, dtype=torch.float32).unsqueeze(1)
            today_pred = clf(today_input).numpy()[0]
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
