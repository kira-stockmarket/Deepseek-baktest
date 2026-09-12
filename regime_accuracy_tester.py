import json
import os
from datetime import datetime
import numpy as np
import pandas as pd


def grade_prediction(row, regime_col: str) -> int:
    """
    Grades accuracy based on the strict institutional thresholds
    the 20-Node Meta-Learner was trained on (Forward 5-Day Horizon):
      - High Momentum Bull: Market moved UP > 1.5% in 5 days
      - True Bearish Distribution: Market moved DOWN < -1.0% in 5 days
      - Choppy / Consolidation: Market remained between -1.0% and +1.5%
    """
    regime = str(row[regime_col])
    ret = row["Fwd_5D_Ret"]

    if pd.isna(ret):
        return None

    # 3-Class Stacking Ensemble mapping
    if regime == "True_Bearish_Distribution":
        return 1 if ret < -0.01 else 0
    elif regime == "High_Momentum_Bull_Breakout":
        return 1 if ret > 0.015 else 0
    elif regime == "Choppy_or_Steady_Consolidation":
        return 1 if -0.01 <= ret <= 0.015 else 0

    # Backward compatibility for legacy labels (if testing older datasets)
    elif regime in ["High_Conviction_Bull", "Steady_Uptrend", "Pre_Breakout_Accumulation"]:
        return 1 if ret > 0 else 0
    elif regime in ["Bearish_Distribution", "Market_Crash"]:
        return 1 if ret < 0 else 0
    elif regime == "Choppy_Consolidation":
        return 1 if -0.015 <= ret <= 0.015 else 0

    return 0


def main():
    print("Loading historical data for regime accuracy testing...")

    # Look for the master ensemble history first, then fall back to single-model history
    file_path = None
    if os.path.exists("master_ensemble_history.csv"):
        file_path = "master_ensemble_history.csv"
    elif os.path.exists("regime_history.csv"):
        file_path = "regime_history.csv"
    else:
        print("Error: Neither 'master_ensemble_history.csv' nor 'regime_history.csv' found.")
        print("Please run meta_ensemble.py or your pipeline workflow first.")
        return

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Loaded dataset: '{file_path}' ({len(df)} total rows)")

    # Identify the correct regime column
    regime_col = None
    for candidate in ["Actionable_Regime_Name", "Regime_Name", "Raw_Regime_Name"]:
        if candidate in df.columns:
            regime_col = candidate
            break

    if not regime_col:
        raise KeyError("Could not find a valid regime column in the CSV.")

    print(f"Evaluating accuracy using column: '{regime_col}'")

    # Calculate forward returns across multiple horizons (forward lookahead strictly for evaluation)
    df["Fwd_1D_Ret"] = df["Close"].shift(-1) / df["Close"] - 1
    df["Fwd_5D_Ret"] = df["Close"].shift(-5) / df["Close"] - 1
    df["Fwd_20D_Ret"] = df["Close"].shift(-20) / df["Close"] - 1

    # Apply strict 5-day grading
    df["Is_Correct_5D"] = df.apply(lambda r: grade_prediction(r, regime_col), axis=1)

    # Exclude recent rows that don't have complete forward outcome windows yet
    df_eval = df.dropna(subset=["Fwd_20D_Ret", "Is_Correct_5D"]).copy()

    total_eval = len(df_eval)
    total_correct = int(df_eval["Is_Correct_5D"].sum())

    report_breakdown = []
    for regime_name, group in df_eval.groupby(regime_col):
        count = len(group)
        correct_count = int(group["Is_Correct_5D"].sum())
        hit_rate = (correct_count / count) * 100 if count > 0 else 0.0

        report_breakdown.append({
            "regime_name": str(regime_name),
            "total_occurrences": count,
            "strict_prediction_hit_rate_5D": round(hit_rate, 2),
            "average_forward_1_day_return": round(float(group["Fwd_1D_Ret"].mean() * 100), 3),
            "average_forward_5_day_return": round(float(group["Fwd_5D_Ret"].mean() * 100), 3),
            "average_forward_20_day_return": round(float(group["Fwd_20D_Ret"].mean() * 100), 3)
        })

    # Sort breakdown by occurrence count
    report_breakdown = sorted(report_breakdown, key=lambda x: x["total_occurrences"], reverse=True)

    final_report = {
        "evaluation_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "evaluated_file": file_path,
        "evaluated_column": regime_col,
        "total_days_evaluated": total_eval,
        "overall_strict_accuracy_5D": round((total_correct / total_eval) * 100, 2) if total_eval > 0 else 0.0,
        "grading_criteria": {
            "High_Momentum_Bull_Breakout": "Market moved UP > 1.5% over 5 days",
            "True_Bearish_Distribution": "Market moved DOWN < -1.0% over 5 days",
            "Choppy_or_Steady_Consolidation": "Market stayed bounded between -1.0% and +1.5% over 5 days"
        },
        "regime_performance_breakdown": report_breakdown
    }

    # Save report
    with open("regime_accuracy_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    print("\nAccuracy test completed. Saved output to 'regime_accuracy_report.json'.")
    print(f"Total Days Audited: {total_eval}")
    print(f"Overall Strict 5-Day Prediction Accuracy: {final_report['overall_strict_accuracy_5D']}%\n")
    for r in report_breakdown:
        print(f"  - {r['regime_name']} ({r['total_occurrences']} days): "
              f"Hit Rate = {r['strict_prediction_hit_rate_5D']}% | "
              f"Avg 5D Ret = {r['average_forward_5_day_return']}%")


if __name__ == "__main__":
    main()
