import json
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
DEVICE = torch.device("cpu")


class FastGRUAutoencoder(nn.Module):
    """1-layer GRU Autoencoder: Highly optimized for CPU GitHub Actions runners."""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, h = self.encoder(x)
        latent = self.fc_enc(h[-1])
        dec_in = self.fc_dec(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in)
        return self.out(dec_out), latent


def fetch_and_prep_data():
    print("1. Fetching Nifty 50 and India VIX data...")
    # Fetch both Nifty and VIX to align dates perfectly
    nifty = yf.download("^NSEI", period="max", auto_adjust=True, progress=False)
    vix = yf.download("^INDIAVIX", period="max", auto_adjust=True, progress=False)
    
    # Handle yfinance multi-index formats dynamically
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
        
    nifty = nifty[["Close", "High", "Low"]]
    vix = vix[["Close"]].rename(columns={"Close": "VIX"})
    
    # Inner join guarantees we only train on days where VIX data exists
    df = nifty.join(vix, how="inner").dropna()

    # Base Price Features
    df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Vol_10"] = df["Returns"].rolling(10).std() * np.sqrt(252)
    df["Vol_21"] = df["Returns"].rolling(21).std() * np.sqrt(252)
    df["Vol_63"] = df["Returns"].rolling(63).std() * np.sqrt(252)

    # India VIX Features (Forward-looking risk)
    df["VIX_SMA_10"] = df["VIX"].rolling(10).mean()
    df["VIX_Ratio"] = df["VIX"] / (df["VIX_SMA_10"] + 1e-9)

    # Native Indicators
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df = df.dropna()
    
    # Added VIX and VIX_Ratio to the ML compression layer
    features = ["Returns", "Vol_10", "Vol_21", "Vol_63", "RSI", "MACD", "ATR", "VIX", "VIX_Ratio"]
    scaled_data = StandardScaler().fit_transform(df[features].values)
    return df, scaled_data, features


def create_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    return np.array([data[i:(i + seq_len)] for i in range(len(data) - seq_len)])


def run_optuna_study(scaled_data: np.ndarray, n_trials: int = 15) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        seq_len = trial.suggest_int("seq_len", 10, 20)
        hidden_dim = trial.suggest_int("hidden_dim", 24, 48)
        latent_dim = trial.suggest_int("latent_dim", 4, 8)
        lr = trial.suggest_float("lr", 1e-3, 5e-3, log=True)
        n_clusters = trial.suggest_int("n_clusters", 3, 4)

        X_seq = create_sequences(scaled_data, seq_len)
        tscv = TimeSeriesSplit(n_splits=2)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_seq):
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=256, shuffle=True)
            model = FastGRUAutoencoder(scaled_data.shape[1], hidden_dim, latent_dim).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            model.train()
            for _ in range(6):  
                for batch_x, in loader:
                    optimizer.zero_grad()
                    recon, _ = model(batch_x)
                    loss = criterion(recon, batch_x)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                _, latents_val = model(torch.tensor(X_val, dtype=torch.float32))
            latents_np = latents_val.cpu().numpy()

            try:
                gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=42)
                labels = gmm.fit_predict(latents_np)
                if len(set(labels)) > 1:
                    sample_size = min(500, len(latents_np))
                    idx = np.random.choice(len(latents_np), sample_size, replace=False)
                    cv_scores.append(silhouette_score(latents_np[idx], labels[idx]))
                else:
                    cv_scores.append(-1.0)
            except Exception:
                cv_scores.append(-1.0)

        return float(np.mean(cv_scores))

    print(f"\n2. Starting VIX-Enhanced Auto-Tuning ({n_trials} Trials)...")
    study = optuna.create_study(direction="maximize")

    def print_progress(study, trial):
        print(f"   [Trial {trial.number + 1:02d}/{n_trials}] Score: {trial.value:.4f} | Best: {study.best_value:.4f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[print_progress])
    print(f"\nOptimal Setup: {study.best_params}\n")
    return study.best_params


def train_final_model(scaled_data: np.ndarray, best_params: dict):
    print("3. Training final model on full historical dataset (30 epochs)...")
    seq_len = best_params["seq_len"]
    X_seq = create_sequences(scaled_data, seq_len)
    tensor_X = torch.tensor(X_seq, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_X), batch_size=128, shuffle=True)

    model = FastGRUAutoencoder(scaled_data.shape[1], best_params["hidden_dim"], best_params["latent_dim"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(30):
        for batch_x, in loader:
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _, latents = model(tensor_X)
    latents_np = latents.cpu().numpy()

    gmm = GaussianMixture(n_components=best_params["n_clusters"], covariance_type="diag", random_state=42)
    labels = gmm.fit_predict(latents_np)
    probs = gmm.predict_proba(latents_np)

    return labels, probs, seq_len


def apply_whipsaw_filter(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.80) -> np.ndarray:
    """
    Prevents algo from over-trading on choppy days.
    Regime only changes if it persists for 2 days OR if the ML confidence > 80%.
    """
    actionable = np.copy(labels)
    current_regime = labels[0]
    
    for i in range(len(labels)):
        proposed = labels[i]
        confidence = probs[i][proposed]
        
        if proposed == current_regime:
            actionable[i] = current_regime
        else:
            # Overriding condition 1: Overwhelming mathematical conviction
            if confidence >= threshold:
                current_regime = proposed
            # Overriding condition 2: Signal persists for 2 consecutive days
            elif i > 0 and proposed == labels[i-1]:
                current_regime = proposed
            
            actionable[i] = current_regime
            
    return actionable


def main():
    start_time = datetime.now()
    df, scaled_data, _ = fetch_and_prep_data()

    best_params = run_optuna_study(scaled_data, n_trials=15)
    raw_labels, probs, seq_len = train_final_model(scaled_data, best_params)
    
    # 4. Apply Whipsaw Filter
    actionable_labels = apply_whipsaw_filter(raw_labels, probs)

    df_regime = df.iloc[seq_len:].copy()
    df_regime["Raw_Regime"] = raw_labels
    df_regime["Actionable_Regime"] = actionable_labels

    # 5. Calculate Profiles
    summary = []
    for state in range(best_params["n_clusters"]):
        rets = df_regime[df_regime["Raw_Regime"] == state]["Returns"]
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0
        summary.append({"state": state, "sharpe": sharpe, "ret": ann_ret, "vol": ann_vol})

    # Rename regimes logically based on Sharpe
    summary_sorted = sorted(summary, key=lambda x: x["sharpe"], reverse=True)
    # Note: Sideways is renamed to Pre_Breakout_Accumulation based on backtest data
    names = ["High_Conviction_Bull", "Steady_Uptrend", "Pre_Breakout_Accumulation", "Bearish_Distribution"]
    state_mapping = {s["state"]: names[i] if i < len(names) else f"Regime_{i}" for i, s in enumerate(summary_sorted)}
    
    # Empirical Capital Allocation Map
    allocation_map = {
        "High_Conviction_Bull": 1.0,
        "Steady_Uptrend": 0.8,
        "Pre_Breakout_Accumulation": 1.0,
        "Bearish_Distribution": 0.0
    }

    current_raw = int(raw_labels[-1])
    current_actionable = int(actionable_labels[-1])
    
    current_raw_name = state_mapping[current_raw]
    current_actionable_name = state_mapping[current_actionable]

    output = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "latest_close": float(df_regime["Close"].iloc[-1]),
        "latest_vix": float(df_regime["VIX"].iloc[-1]),
        "regime_data": {
            "raw_regime_today": current_raw_name,
            "actionable_regime_today": current_actionable_name,
            "recommended_capital_allocation": allocation_map.get(current_actionable_name, 0.5),
            "probabilities": {state_mapping[i]: round(float(probs[-1][i]), 4) for i in range(best_params["n_clusters"])}
        },
        "runtime_seconds": round((datetime.now() - start_time).total_seconds(), 2)
    }

    with open("current_regime.json", "w") as f:
        json.dump(output, f, indent=2)

    df_regime["Raw_Regime_Name"] = df_regime["Raw_Regime"].map(state_mapping)
    df_regime["Actionable_Regime_Name"] = df_regime["Actionable_Regime"].map(state_mapping)
    df_regime[["Close", "VIX", "Returns", "Actionable_Regime_Name", "Raw_Regime_Name"]].to_csv("regime_history.csv")

    print(f"4. Success! Completed in {output['runtime_seconds']}s.")
    print(f"   -> Raw Signal: {current_raw_name}")
    print(f"   -> Actionable Signal (Whipsaw Filtered): {current_actionable_name}")
    print(f"   -> Live API Updated: current_regime.json")


if __name__ == "__main__":
    main()
