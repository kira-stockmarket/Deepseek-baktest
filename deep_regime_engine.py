import json
import os
import torch
import optuna
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime

warnings.filterwarnings("ignore")

# Force CPU for GitHub Actions to avoid CUDA overhead
DEVICE = torch.device("cpu") 

class LSTMAutoencoder(nn.Module):
    """Deep sequence-to-sequence autoencoder to extract non-linear market features."""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)
        
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        latent = self.fc_enc(h[-1]) 
        # Repeat latent space for sequence decoding
        dec_in = self.fc_dec(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in)
        reconstructed = self.out(dec_out)
        return reconstructed, latent


def fetch_and_prep_data(ticker="^NSEI"):
    """Fetch Nifty 50 data and engineer technical features."""
    df = yf.download(ticker, period="max", interval="1d", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Vol_10"] = df["Returns"].rolling(10).std() * np.sqrt(252)
    df["Vol_21"] = df["Returns"].rolling(21).std() * np.sqrt(252)
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]
    
    df = df.dropna()
    features = ["Returns", "Vol_10", "Vol_21", "Range"]
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features].values)
    return df, scaled_data, features


def create_sequences(data, seq_len):
    """Convert flat time series into rolling windows for the LSTM."""
    X = [data[i:(i + seq_len)] for i in range(len(data) - seq_len)]
    return np.array(X)


def run_optuna_study(scaled_data, n_trials=15):
    """Bayesian optimization to find the best network architecture and regime count."""
    
    def objective(trial):
        seq_len = trial.suggest_int("seq_len", 5, 21)
        hidden_dim = trial.suggest_int("hidden_dim", 16, 64)
        latent_dim = trial.suggest_int("latent_dim", 4, 12)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        n_clusters = trial.suggest_int("n_clusters", 3, 5)
        
        X_seq = create_sequences(scaled_data, seq_len)
        tensor_X = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
        loader = DataLoader(TensorDataset(tensor_X), batch_size=256, shuffle=True)
        
        model = LSTMAutoencoder(scaled_data.shape[1], hidden_dim, latent_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Fast train for evaluation (10 epochs)
        model.train()
        for _ in range(10):
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
        
        try:
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
            labels = gmm.fit_predict(latents_np)
            return silhouette_score(latents_np, labels) if len(set(labels)) > 1 else -1.0
        except:
            return -1.0

    print("Initiating Heavy ML Auto-Tuning via Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Optimal Parameters Found: {study.best_params}")
    return study.best_params


def train_final_model(scaled_data, best_params):
    """Train the final Deep LSTM model and fit GMM on the extracted latent space."""
    seq_len = best_params["seq_len"]
    X_seq = create_sequences(scaled_data, seq_len)
    tensor_X = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    loader = DataLoader(TensorDataset(tensor_X), batch_size=128, shuffle=True)
    
    model = LSTMAutoencoder(
        scaled_data.shape[1], 
        best_params["hidden_dim"], 
        best_params["latent_dim"]
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()
    
    print("Training final deep latent representation...")
    model.train()
    for _ in range(40):  # Deep training
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
    
    gmm = GaussianMixture(n_components=best_params["n_clusters"], covariance_type='full', random_state=42)
    labels = gmm.fit_predict(latents_np)
    probs = gmm.predict_proba(latents_np)
    
    return labels, probs, seq_len


def main():
    df, scaled_data, feature_cols = fetch_and_prep_data("^NSEI")
    
    # Run bayesian optimization (limits trials to ~5-10 min runtime on GH Actions)
    best_params = run_optuna_study(scaled_data, n_trials=15)
    
    labels, probs, seq_len = train_final_model(scaled_data, best_params)
    
    # Align labels with original dataframe (dropping first `seq_len` rows)
    df_regime = df.iloc[seq_len:].copy()
    df_regime["Regime"] = labels
    
    # Profiling regimes to map semantic names
    summary = []
    n_clusters = best_params["n_clusters"]
    for state in range(n_clusters):
        state_returns = df_regime[df_regime["Regime"] == state]["Returns"]
        annual_return = state_returns.mean() * 252
        annual_vol = state_returns.std() * np.sqrt(252)
        sharpe = (annual_return / annual_vol) if annual_vol != 0 else 0
        summary.append({
            "state": state, "annual_return": annual_return,
            "annual_vol": annual_vol, "sharpe": sharpe
        })
        
    summary_sorted = sorted(summary, key=lambda x: x["sharpe"], reverse=True)
    names = ["High_Conviction_Bull", "Grinding_Uptrend", "Sideways_Chop", "Bearish_Distribution", "Crash_Volatile"]
    
    state_mapping = {s["state"]: names[i] for i, s in enumerate(summary_sorted)}
    
    current_regime_id = int(labels[-1])
    current_regime_name = state_mapping[current_regime_id]
    current_probs = probs[-1]
    
    output_payload = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "latest_close": float(df_regime["Close"].iloc[-1]),
        "latest_date": str(df_regime.index[-1].strftime("%Y-%m-%d")),
        "ml_hyperparameters": best_params,
        "current_regime_id": current_regime_id,
        "current_regime_name": current_regime_name,
        "state_probabilities": {
            state_mapping[i]: round(float(current_probs[i]), 4)
            for i in range(n_clusters)
        },
        "regime_profiles": [
            {
                "regime_name": state_mapping[s["state"]],
                "expected_annual_return": round(float(s["annual_return"]), 4),
                "expected_annual_vol": round(float(s["annual_vol"]), 4),
                "sharpe_ratio": round(float(s["sharpe"]), 4)
            }
            for s in summary_sorted
        ]
    }
    
    with open("current_regime.json", "w") as f:
        json.dump(output_payload, f, indent=2)

    df_regime["Regime_Name"] = df_regime["Regime"].map(state_mapping)
    df_regime[["Close", "Returns", "Regime", "Regime_Name"]].to_csv("regime_history.csv")
    print(f"Update Complete. Current Regime: {current_regime_name}")

if __name__ == "__main__":
    main()
