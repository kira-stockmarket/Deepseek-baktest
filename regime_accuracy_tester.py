import json
from datetime import datetime
import numpy as np
import pandas as pd


def grade_prediction(row, forward_col: str = "Fwd_5D_Ret", regime_col: str = "Actionable_Regime_Name"):
    """
    Grades whether forward market performance matched the regime's statistical expectation.
    - Bullish & Pre-Breakout: Market went UP over the forward window.
    - Bearish: Market went DOWN over the forward window.
    """
    regime = str(row[regime_col])
    ret = row[forward_col]

    if pd.isna(ret):
        return None

    if regime in ["High_Conviction_Bull", "Steady_Uptrend", "Pre_Breakout_Accumulation"]:
        return 1 if ret > 0 else 0
    elif regime in ["Bearish_Distribution", "Market_Crash"]:
        return 1 if ret < 0 else 0
    elif regime == "Sideways_Consolidation":
        return 1 if -0.015 <= ret <= 0.015 else 0
    else:
        return 1 if ret > 0 else 0


def main():
    print("Loading regime history for accuracy testing...")
    try:
        df = pd.read_csv("regime_history.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'regime_history.csv' not found. Run deep_regime_engine.py first.")
        return

    # Detect the appropriate regime column
    if "Actionable_Regime_Name" in df.columns:
        regime_col = "Actionable_Regime_Name"
    elif "Regime_Name" in df.columns:
        regime_col = "Regime_Name"
    elif "Raw_Regime_Name" in df.columns:
        regime_col = "Raw_Regime_Name"
    else:
        raise KeyError("No valid regime column found in regime_history.csv")

    print(f"Testing accuracy using column: '{regime_col}'")

    # Forward returns (lookahead for evaluation only)
    df["Fwd_1D_Ret"] = df["Close"].shift(-1) / df["Close"] - 1
    df["Fwd_5D_Ret"] = df["Close"].shift(-5) / df["Close"] - 1
    df["Fwd_20D_Ret"] = df["Close"].shift(-20) / df["Close"] - 1

    # Grade accuracy across forward horizons
    df["Is_Correct_5D"] = df.apply(lambda r: grade_prediction(r, "Fwd_5D_Ret", regime_col), axis=1)

    # Drop latest rows that do not have forward lookahead outcomes yet
    df_eval = df.dropna(subset=["Fwd_20D_Ret", "Is_Correct_5D"])

    report_data = []
    total_eval = len(df_eval)
    total_correct = int(df_eval["Is_Correct_5D"].sum())

    for regime_name, group in df_eval.groupby(regime_col):
        count = len(group)
        correct_count = int(group["Is_Correct_5D"].sum())
        hit_rate = (correct_count / count) * 100 if count > 0 else 0.0

        report_data.append({
            "regime_name": regime_name,
            "total_occurrences": count,
            "prediction_hit_rate_5D": round(hit_rate, 2),
            "average_forward_1_day_return": round(float(group["Fwd_1D_Ret"].mean() * 100), 3),
            "average_forward_5_day_return": round(float(group["Fwd_5D_Ret"].mean() * 100), 3),
            "average_forward_20_day_return": round(float(group["Fwd_20D_Ret"].mean() * 100), 3)
        })

    # Sort report by sample size
    report_data = sorted(report_data, key=lambda x: x["total_occurrences"], reverse=True)

    final_report = {
        "evaluation_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "tested_regime_column": regime_col,
        "total_days_evaluated": total_eval,
        "overall_model_accuracy_5D": round((total_correct / total_eval) * 100, 2),
        "definition_of_correct": {
            "Bull_and_Accumulation": "Market moved POSITIVE over the next 5 days",
            "Bearish_Distribution": "Market moved NEGATIVE over the next 5 days"
        },
        "regime_performance_breakdown": report_data
    }

    with open("regime_accuracy_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    print("\nAccuracy audit complete. Saved to 'regime_accuracy_report.json'.")
    print(f"Overall 5-Day Directional Hit Rate: {final_report['overall_model_accuracy_5D']}%")


if __name__ == "__main__":
    main()
