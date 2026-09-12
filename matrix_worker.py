import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
from torch import nn
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING) # Keep logs clean

SECTOR_TICKERS = {
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_REALTY": "^CNXREALTY"
}

# MACRO DATA TICKERS
MACRO_TICKERS = {
    "VIX": "^INDIAVIX",
    "USDINR": "INR=X",
    "SP500": "^GSPC",
    "CRUDE": "CL=F",
    "US10Y": "^TNX"
}

def calculate_macro_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Calculates cross-asset features to give the AI global context."""
    # Sector features
    df["Returns"] = df["INDEX"].pct_change(1)
    df[f"Momentum_{lookback}"] = df["INDEX"].pct_change(lookback)
    sma = df["INDEX"].rolling(lookback).mean()
    df[f"Dist_SMA_{lookback}"] = (df["INDEX"] / (sma + 1e-9)) - 1
    df[f"Vol_{lookback}"] = df["Returns"].rolling(lookback).std() * np.sqrt(252)
    
    # Intermarket Macro Features
    df[f"USD_Trend_{lookback}"] = df["USDINR"].pct_change(lookback)
    df[f"SP500_Corr_{lookback}"] = df["INDEX"].pct_change(1).rolling(lookback).corr(df["SP500"].pct_change(1))
    df[f"Crude_Shock"] = df["CRUDE"].pct_change(3) # 3-day oil shock
    df[f"Yield_Spread"] = df["US10Y"] - df["US10Y"].rolling(lookback).mean()
    
    return df

def fetch_and_prep_data(sector_name: str, lookback: int):
    ticker = SECTOR_TICKERS.get(sector_name, "^NSEBANK")
    print(f"[{sector_name}] Fetching Sector and Global Macro Data...")
    
    all_tickers = [ticker] + list(MACRO_TICKERS.values())
    raw_data = yf.download(all_tickers, period="10y", auto_adjust=True, progress=False)["Close"]
    
    # Rename columns to match our feature pipeline
    raw_data = raw_data.rename(columns={ticker: "INDEX", MACRO_TICKERS["VIX"]: "VIX", 
                                        MACRO_TICKERS["USDINR"]: "USDINR", MACRO_TICKERS["SP500"]: "SP500", 
                                        MACRO_TICKERS["CRUDE"]: "CRUDE", MACRO_TICKERS["US10Y"]: "US10Y"})
    
    # Forward fill macro data (handles different global market holidays)
    df = raw_data.ffill().dropna()
    df = calculate_macro_features(df, lookback)

    # TARGET: Institutional Breakout Thresholds
    fwd_ret = df["INDEX"].shift(-5) / df["INDEX"] - 1
    df["Target"] = np.where(fwd_ret < -0.008, 0, np.where(fwd_ret > 0.008, 2, 1))
    df = df.dropna()
    
    features = [c for c in df.columns if c not in ["INDEX", "VIX", "USDINR", "SP500", "CRUDE", "US10Y", "Target"]]
    return df, features

# ==========================================
# OPTUNA DYNAMIC HYPERPARAMETER TUNING
# ==========================================
def optimize_tree_model(X_train, y_train, model_type, n_trials=15):
    """Uses TimeSeriesSplit to find the exact perfect parameters without lookahead bias."""
    tscv = TimeSeriesSplit(n_splits=3)
    
    def objective(trial):
        if model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42
            }
            clf = xgb.XGBClassifier(**params)
            
        elif model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "class_weight": "balanced",
                "random_state": 42,
                "verbose": -1
            }
            clf = lgb.LGBMClassifier(**params)
            
        elif model_type == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 300),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "auto_class_weights": "Balanced",
                "random_state": 42,
                "verbose": 0
            }
            clf = cb.CatBoostClassifier(**params)
            
        elif model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "class_weight": "balanced_subsample",
                "random_state": 42
            }
            clf = RandomForestClassifier(**params)

        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            if model_type == "xgboost":
                weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
                weight_dict = dict(zip(np.unique(y_tr), weights))
                clf.fit(X_tr, y_tr, sample_weight=y_tr.map(weight_dict))
            else:
                clf.fit(X_tr, y_tr)
                
            preds = clf.predict(X_val)
            scores.append(balanced_accuracy_score(y_val, preds))
            
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# ==========================================
# DEEP LSTM NEURAL NETWORK WITH EARLY STOPPING
# ==========================================
class DeepLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, num_layers=3, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.softmax(self.fc(h[-1]), dim=1)

def train_lstm(X_train, y_train, X_test, weights):
    # Time-based Validation split for Early Stopping
    val_split = int(len(X_train) * 0.8)
    X_tr_t = torch.tensor(X_train.iloc[:val_split].values, dtype=torch.float32).unsqueeze(1)
    y_tr_t = torch.tensor(y_train.iloc[:val_split].values, dtype=torch.long)
    
    X_val_t = torch.tensor(X_train.iloc[val_split:].values, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_train.iloc[val_split:].values, dtype=torch.long)
    
    X_te_t = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)

    model = DeepLSTM(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    best_val_loss = float('inf')
    patience, patience_counter = 10, 0
    best_model_state = None

    # Train for up to 100 epochs, but stop early if validation loss increases
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()

        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"LSTM Early Stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).numpy()
    return preds, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lookback", type=int, required=True)
    args = parser.parse_args()

    df, feature_cols = fetch_and_prep_data(args.sector, args.lookback)
    X, y = df[feature_cols], df["Target"]

    # 80/20 Chronological Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    
    print(f"[{args.sector}] Training {args.model.upper()} ({args.lookback}D). Executing Optuna Search...")

    # Route logic based on model type
    if args.model == "xgboost":
        best_params = optimize_tree_model(X_train, y_train, "xgboost")
        clf = xgb.XGBClassifier(**best_params)
        weight_dict = dict(zip(classes, weights))
        clf.fit(X_train, y_train, sample_weight=y_train.map(weight_dict))
        preds = clf.predict_proba(X_test)
        
    elif args.model == "lightgbm":
        best_params = optimize_tree_model(X_train, y_train, "lightgbm")
        clf = lgb.LGBMClassifier(**best_params)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
        
    elif args.model == "catboost":
        best_params = optimize_tree_model(X_train, y_train, "catboost")
        clf = cb.CatBoostClassifier(**best_params)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
        
    elif args.model == "random_forest":
        best_params = optimize_tree_model(X_train, y_train, "random_forest")
        clf = RandomForestClassifier(**best_params)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
        
    elif args.model == "gru":
        # Note: Renamed to deep_lstm internally but kept 'gru' string for pipeline compatibility
        preds, clf = train_lstm(X_train, y_train, X_test, weights)

    # Save Outputs
    os.makedirs("output", exist_ok=True)
    out_df = pd.DataFrame(preds, index=X_test.index, columns=["P0", "P1", "P2"])
    out_df["True_Target"] = y_test
    out_df.to_csv(f"output/{args.sector}_{args.model}_{args.lookback}_preds.csv")

    if args.model == "gru":
        with torch.no_grad():
            today_input = torch.tensor(X.iloc[-1:].values, dtype=torch.float32).unsqueeze(1)
            today_pred = clf(today_input).numpy()[0]
    else:
        today_pred = clf.predict_proba(X.iloc[-1:])[-1]

    meta = {
        "sector": args.sector,
        "model": args.model,
        "lookback": args.lookback,
        "today_probs": [round(float(p), 4) for p in today_pred],
        "val_accuracy": round(accuracy_score(y_test, np.argmax(preds, axis=1)), 4),
        "val_balanced_accuracy": round(balanced_accuracy_score(y_test, np.argmax(preds, axis=1)), 4)
    }

    with open(f"output/{args.sector}_{args.model}_{args.lookback}_meta.json", "w") as f:
        json.dump(meta, f)
        
    print(f"[{args.sector}] {args.model.upper()} ({args.lookback}D) Complete. Balanced Accuracy: {meta['val_balanced_accuracy']}")

if __name__ == "__main__":
    main()
