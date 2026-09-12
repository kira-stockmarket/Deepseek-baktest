import json
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DEVICE = torch.device("cpu")


class FastGRUAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 6):
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


def fetch_leading_features():
    """Engineers LEADING distribution indicators instead of lagging crash indicators."""
    print("1. Fetching Nifty 50 and India VIX data...")
    nifty = yf.download("^NSEI", period="max", auto_adjust=True, progress=False)
    vix = yf.download("^INDIAVIX", period="max", auto_adjust=True, progress=False)

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    nifty = nifty[["Close", "High", "Low"]]
    vix = vix[["Close"]].rename(columns={"Close": "VIX"})
    df = nifty.join(vix, how="inner").dropna()

    # Returns
    df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))

    # 1. DISTRIBUTION FEATURE: Trend Breakdown (Distance to 20 & 50 EMAs)
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["Dist_EMA_20"] = (df["Close"] / df["EMA_20"]) - 1
    df["EMA_20_50_Spread"] = (df["EMA_20"] / df["EMA_50"]) - 1

    # 2. DISTRIBUTION FEATURE: Momentum Rollover (MACD Slope & RSI Slope)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Slope_5"] = df["MACD"] - df["MACD"].shift(5)

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df["RSI"] = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))
    df["RSI_Slope_5"] = df["RSI"] - df["RSI"].shift(5)

    # 3. DISTRIBUTION FEATURE: VIX Divergence (VIX rising from low levels while price stalls)
    df["VIX_5D_Change"] = df["VIX"].diff(5)
    df["VIX_Level"] = df["VIX"]

    # Forward 5-day return for EMPIRICAL validation
    df["Fwd_5D_Ret"] = df["Close"].shift(-5) / df["Close"] - 1

    df = df.dropna()

    features = [
        "Returns", "Dist_EMA_20", "EMA_20_50_Spread",
        "MACD", "MACD_Slope_5", "RSI", "RSI_Slope_5",
        "VIX_5D_Change", "VIX_Level"
    ]
    scaled_data = StandardScaler().fit_transform(df[features].values)
    return df, scaled_data, features


def create_sequences(data: np.ndarray, seq_len: int = 10) -> np.ndarray:
    return np.array([data[i:(i + seq_len)] for i in range(len(data) - seq_len)])


def main():
    df, scaled_data, features = fetch_leading_features()
    seq_len = 10

    # Train Fast GRU Autoencoder
    X_seq = create_sequences(scaled_data, seq_len)
    tensor_X = torch.tensor(X_seq, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_X), batch_size=128, shuffle=True)

    print("2. Training neural compression network...")
    model = FastGRUAutoencoder(input_dim=scaled_data.shape[1], hidden_dim=32, latent_dim=6).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(25):
        for bx, in loader:
            optimizer.zero_grad()
            recon, _ = model(bx)
            loss = criterion(recon, bx)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _, latents = model(tensor_X)
    latents_np = latents.cpu().numpy()

    # 4 Clusters: Strong Bull, Weak Bull, Distribution/Top, Capitulation/Selloff
    gmm = GaussianMixture(n_components=4, covariance_type="diag", random_state=42)
    labels = gmm.fit_predict(latents_np)
    probs = gmm.predict_proba(latents_np)

    df_regime = df.iloc[seq_len:].copy()
    df_regime["Regime"] = labels

    # 3. TRUTH-BASED REGIME NAMING (Based on actual forward returns, NOT Sharpe)
    cluster_stats = []
    for c in range(4):
        subset = df_regime[df_regime["Regime"] == c]
        mean_fwd_5d = subset["Fwd_5D_Ret"].mean() * 100
        ann_vol = subset["Returns"].std() * np.sqrt(252) * 100
        cluster_stats.append({
            "cluster_id": c,
            "mean_fwd_5d": mean_fwd_5d,
            "ann_vol": ann_vol,
            "count": len(subset)
        })

    # Sort strictly by forward 5-day return (Ascending: Worst to Best)
    cluster_stats_sorted = sorted(cluster_stats, key=lambda x: x["mean_fwd_5d"])

    state_mapping = {}
    for rank, item in enumerate(cluster_stats_sorted):
        cid = item["cluster_id"]
        fwd_ret = item["mean_fwd_5d"]

        if rank == 0:
            # If the worst cluster is negative, it's a true Bearish regime
            if fwd_ret < 0:
                state_mapping[cid] = "True_Bearish_Distribution"
            else:
                # If even the worst cluster is slightly positive, name it honestly
                state_mapping[cid] = "Low_Return_Exhaustion"
        elif rank == 1:
            state_mapping[cid] = "Choppy_Consolidation"
        elif rank == 2:
            state_mapping[cid] = "Steady_Uptrend"
        else:
            state_mapping[cid] = "High_Momentum_Bull"

    current_state = int(labels[-1])
    current_name = state_mapping[current_state]

    output = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "current_regime": current_name,
        "empirical_cluster_validation": [
            {
                "regime_name": state_mapping[item["cluster_id"]],
                "historical_5D_forward_return": f"{item['mean_fwd_5d']:.3f}%",
                "annualized_vol": f"{item['ann_vol']:.1f}%",
                "days_in_regime": item["count"]
            }
            for item in cluster_stats_sorted
        ],
        "latest_probabilities": {
            state_mapping[i]: round(float(probs[-1][i]), 4)
            for i in range(4)
        }
    }

    with open("current_regime.json", "w") as f:
        json.dump(output, f, indent=2)

    df_regime["Regime_Name"] = df_regime["Regime"].map(state_mapping)
    df_regime["Actionable_Regime_Name"] = df_regime["Regime_Name"]
    df_regime.to_csv("regime_history.csv")

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
