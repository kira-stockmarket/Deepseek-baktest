import pandas as pd
import numpy as np
import json
from datetime import datetime

def grade_prediction(row, forward_col="Fwd_5D_Ret"):
    """Grades if the forward market action matched the regime's profile."""
    regime = str(row["Regime_Name"])
    ret = row[forward_col]
    
    if pd.isna(ret):
        return None
        
    if regime in ["High_Conviction_Bull", "Steady_Uptrend"]:
        return 1 if ret > 0 else 0
    elif regime in ["High_Vol_Bear", "Market_Crash"]:
        return 1 if ret < 0 else 0
    elif regime in ["Sideways_Consolidation"]:
        return 1 if -0.015 <= ret <= 0.015 else 0
    else:
        # Fallback for generic unnamed regimes (assuming positive drift)
        return 1 if ret > 0 else 0

def main():
    print("Loading regime history for accuracy testing...")
    try:
        df = pd.read_csv("regime_history.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'regime_history.csv' not found.")
        return

    # Calculate Forward Returns (Lookahead into the future)
    # 1D = Tomorrow, 5D = Next Week, 20D = Next Month
    df["Fwd_1D_Ret"] = df["Close"].shift(-1) / df["Close"] - 1
    df["Fwd_5D_Ret"] = df["Close"].shift(-5) / df["Close"] - 1
    df["Fwd_20D_Ret"] = df["Close"].shift(-20) / df["Close"] - 1

    # Grade the predictions based on 5-day forward action
    df["Is_Correct_5D"] = df.apply(lambda row: grade_prediction(row, "Fwd_5D_Ret"), axis=1)
    
    # Drop rows at the very end where we don't have future data yet
    df_eval = df.dropna(subset=["Fwd_20D_Ret", "Is_Correct_5D"])

    # Aggregate statistics per regime
    report_data = []
    total_predictions = len(df_eval)
    overall_correct = df_eval["Is_Correct_5D"].sum()
    
    regime_groups = df_eval.groupby("Regime_Name")
    
    for regime_name, group in regime_groups:
        count = len(group)
        correct_count = group["Is_Correct_5D"].sum()
        hit_rate = correct_count / count if count > 0 else 0
        
        # Mean forward returns
        mean_1d = group["Fwd_1D_Ret"].mean()
        mean_5d = group["Fwd_5D_Ret"].mean()
        mean_20d = group["Fwd_20D_Ret"].mean()
        
        report_data.append({
            "regime_name": regime_name,
            "total_occurrences": count,
            "prediction_hit_rate_5D": round(hit_rate * 100, 2),
            "average_forward_1_day_return": round(mean_1d * 100, 3),
            "average_forward_5_day_return": round(mean_5d * 100, 3),
            "average_forward_20_day_return": round(mean_20d * 100, 3)
        })

    # Sort report by occurrence count
    report_data = sorted(report_data, key=lambda x: x["total_occurrences"], reverse=True)

    final_report = {
        "evaluation_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_days_evaluated": total_predictions,
        "overall_model_accuracy_5D": round((overall_correct / total_predictions) * 100, 2),
        "definition_of_correct": {
            "Bull_Regimes": "Market went UP over the next 5 days",
            "Bear_Regimes": "Market went DOWN over the next 5 days",
            "Sideways": "Market stayed within a +/- 1.5% range over the next 5 days"
        },
        "regime_performance_breakdown": report_data
    }

    with open("regime_accuracy_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
        
    print("Accuracy testing complete. Results saved to 'regime_accuracy_report.json'")
    print(f"Overall 5-Day Accuracy: {final_report['overall_model_accuracy_5D']}%")

if __name__ == "__main__":
    main()
