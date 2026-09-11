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
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Force CPU execution for fast, stable runs on GitHub Actions runners
DEVICE = torch.device("cpu")


class BiLSTMAutoencoder(nn.Module):
    """Deep Bi-Directional LSTM Autoencoder for temporal feature compression."""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        # 2-layer Bi-Directional LSTM Encoder
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        # BiLSTM outputs 2 * hidden_dim
        self.fc_enc = nn.Linear(hidden_dim * 2, latent_dim)

        # 2-layer Bi-Directional LSTM Decoder
        self.fc_dec = nn.Linear(latent_dim, hidden_dim * 2)
        self.decoder = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.out = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        # Concatenate forward and backward hidden states from the top layer
        h_concat = torch.cat((h[-2], h[-1]), dim=1)
        latent = self.fc_enc(h_concat)

        # Reconstruct sequence
        dec_in = self.fc_dec(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in)
        reconstructed = self.out(dec_out)
        return reconstructed, latent


def fetch_and_prep_data(ticker: str = "^NSEI"):
    """Fetch all historical Nifty 50 data and compute features with native Pandas."""
    print(f"Fetching historical data for {ticker}...")
    df = yf.download(ticker, period="max", interval="1d", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. Price Action & Realized Volatility
    df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Vol_10"] = df["Returns"].rolling(10).std() * np.sqrt(252)
    df["Vol_21"] = df["Returns"].rolling(21).std() * np.sqrt(252)
    df["Vol_63"] = df["Returns"].rolling(63).std() * np.sqrt(252)

    # 2. Native Technical Indicators (Zero third-party library dependencies)
    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (Moving Average Convergence Divergence)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2

    # ATR (Average True Range)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()

    df = df.dropna()
    features = ["Returns", "Vol_10", "Vol_21", "Vol_63", "RSI", "MACD", "ATR"]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features].values)
    return df, scaled_data, features


def create_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    """Create sliding temporal windows for sequence-to-sequence modeling."""
    X = [data[i:(i + seq_len)] for i in range(len(data) - seq_len)]
    return np.array(X)


def run_optuna_study(scaled_data: np.ndarray, n_trials: int = 60) -> dict:
    """Walk-forward cross-validated Bayesian optimization for hyperparameters."""
    def objective(trial):
        seq_len = trial.suggest_int("seq_len", 10, 30)
        hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
        latent_dim = trial.suggest_int("latent_dim", 4, 12)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)
        n_clusters = trial.suggest_int("n_clusters", 3, 5)

        X_seq = create_sequences(scaled_data, seq_len)
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_seq):
            X_train = X_seq[train_idx]
            X_val = X_seq[val_idx]

            tensor_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
            tensor_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)

            loader = DataLoader(TensorDataset(tensor_train), batch_size=256, shuffle=True)
            model = BiLSTMAutoencoder(scaled_data.shape[1], hidden_dim, latent_dim, dropout).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            model.train()
            for _ in range(12):
                for batch_x, in loader:
                    optimizer.zero_grad()
                    recon, _ = model(batch_x)
                    loss = criterion(recon, batch_x)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                _, latents_val = model(tensor_val)
            latents_np = latents_val.cpu().numpy()

            try:
                gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
                labels = gmm.fit_predict(latents_np)
                if len(set(labels)) > 1:
                    cv_scores.append(silhouette_score(latents_np, labels))
                else:
                    cv_scores.append(-1.0)
            except Exception:
                cv_scores.append(-1.0)

        return float(np.mean(cv_scores))

    print(f"Initiating Bayesian Optimization ({n_trials} trials, Walk-Forward CV)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Optimal Parameters Selected: {study.best_params}")
    return study.best_params


def train_final_model(scaled_data: np.ndarray, best_params: dict):
    """Train the optimized BiLSTM Autoencoder and cluster latent market states with GMM."""
    seq_len = best_params["seq_len"]
    X_seq = create_sequences(scaled_data, seq_len)
    tensor_X = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    loader = DataLoader(TensorDataset(tensor_X), batch_size=128, shuffle=True)

    model = BiLSTMAutoencoder(
        scaled_data.shape[1],
        best_params["hidden_dim"],
        best_params["latent_dim"],
        best_params["dropout"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()

    print("Training final deep network (80 epochs)...")
    model.train()
    for _ in range(80):
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

    # Fit Gaussian Mixture Model with multiple restarts
    gmm = GaussianMixture(
        n_components=best_params["n_clusters"],
        covariance_type="full",
        n_init=10,
        random_state=42
    )
    labels = gmm.fit_predict(latents_np)
    probs = gmm.predict_proba(latents_np)

    return labels, probs, seq_len


def main():
    df, scaled_data, _ = fetch_and_prep_data("^NSEI")

    best_params = run_optuna_study(scaled_data, n_trials=60)
    labels, probs, seq_len = train_final_model(scaled_data, best_params)

    # Synchronize labels with dataframe
    df_regime = df.iloc[seq_len:].copy()
    df_regime["Regime"] = labels

    # Profile clusters by Sharpe Ratio
    summary = []
    n_clusters = best_params["n_clusters"]
    for state in range(n_clusters):
        state_returns = df_regime[df_regime["Regime"] == state]["Returns"]
        annual_return = state_returns.mean() * 252
        annual_vol = state_returns.std() * np.sqrt(252)
        sharpe = (annual_return / annual_vol) if annual_vol != 0 else 0
        summary.append({
            "state": state,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "samples": int(len(state_returns))
        })

    summary_sorted = sorted(summary, key=lambda x: x["sharpe"], reverse=True)
    regime_names = [
        "High_Conviction_Bull",
        "Steady_Uptrend",
        "Sideways_Consolidation",
        "High_Vol_Bear",
        "Market_Crash"
    ]

    state_mapping = {
        s["state"]: regime_names[i] if i < len(regime_names) else f"Regime_{i}"
        for i, s in enumerate(summary_sorted)
    }

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
                "sharpe_ratio": round(float(s["sharpe"]), 4),
                "historical_days_in_regime": s["samples"]
            }
            for s in summary_sorted
        ]
    }

    # Save outputs
    with open("current_regime.json", "w") as f:
        json.dump(output_payload, f, indent=2)

    df_regime["Regime_Name"] = df_regime["Regime"].map(state_mapping)
    df_regime[["Close", "Returns", "Regime", "Regime_Name"]].to_csv("regime_history.csv")

    print("\n--- Model Execution Complete ---")
    print(f"Current Market Regime: {current_regime_name}")
    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
