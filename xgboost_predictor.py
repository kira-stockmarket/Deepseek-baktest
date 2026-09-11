import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score

def engineer_predictive_features(df):
    """Create features based on TODAY to predict TOMORROW."""
    # Price Momentum
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["Distance_to_SMA"] = (df["Close"] / df["SMA_10"]) - 1
    
    # Short-term Lags (How did we move yesterday vs the day before?)
    df["Ret_Lag_1"] = df["Returns"].shift(1)
    df["Ret_Lag_2"] = df["Returns"].shift(2)
    df["Ret_Lag_3"] = df["Returns"].shift(3)
    
    # Short-term Volatility
    df["Vol_5D"] = df["Returns"].rolling(5).std() * np.sqrt(252)
    df["Vol_15D"] = df["Returns"].rolling(15).std() * np.sqrt(252)
    
    # Volatility Trend (Is volatility expanding or shrinking?)
    df["Vol_Ratio"] = df["Vol_5D"] / (df["Vol_15D"] + 1e-9)
    
    return df

def main():
    print("Loading regime history...")
    try:
        df = pd.read_csv("regime_history.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'regime_history.csv' not found. Run the Deep ML engine first.")
        return

    # 1. Feature Engineering
    df = engineer_predictive_features(df)
    
    # 2. Define the Target (Shift Regime by -1 to predict TOMORROW)
    df["Target_Regime"] = df["Regime"].shift(-1)
    
    # Drop rows with NaNs (created by rolling windows and shifting)
    df = df.dropna()
    
    features = [
        "Returns", "Distance_to_SMA", "Ret_Lag_1", "Ret_Lag_2", "Ret_Lag_3", 
        "Vol_5D", "Vol_15D", "Vol_Ratio", "Regime" # Including today's regime to predict tomorrow's
    ]
    
    X = df[features]
    y = df["Target_Regime"].astype(int)
    
    # 3. Train/Test Split (Strictly Chronological to prevent Lookahead Bias)
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training on {len(X_train)} days, Testing on {len(X_test)} days...")
    
    # 4. Train XGBoost Classifier
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 5. Evaluate Accuracy on Unseen Future Data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOut-of-Sample Prediction Accuracy: {accuracy * 100:.2f}%\n")
    
    # Map feature importance
    importance = dict(zip(features, model.feature_importances_))
    sorted_importance = {k: round(float(v), 4) for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
    
    # 6. Predict Tomorrow's Regime
    latest_features = X.iloc[-1:]
    tomorrow_probs = model.predict_proba(latest_features)[0]
    tomorrow_pred_id = int(np.argmax(tomorrow_probs))
    
    # Create mapping dictionary from the CSV
    regime_map = dict(zip(df["Regime"], df["Regime_Name"]))
    
    output = {
        "prediction_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_accuracy": round(accuracy, 4),
        "predicted_regime_tomorrow": regime_map.get(tomorrow_pred_id, f"Regime_{tomorrow_pred_id}"),
        "confidence_probabilities": {
            regime_map.get(i, f"Regime_{i}"): round(float(prob), 4) 
            for i, prob in enumerate(tomorrow_probs)
        },
        "top_predictive_features": sorted_importance
    }
    
    with open("tomorrow_prediction.json", "w") as f:
        json.dump(output, f, indent=2)
        
    print("Prediction complete. Output saved to 'tomorrow_prediction.json'")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
