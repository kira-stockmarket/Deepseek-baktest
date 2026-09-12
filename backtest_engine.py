import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Prevent GUI errors on headless GitHub Actions Ubuntu runners
plt.switch_backend("Agg")


def calculate_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum peak-to-trough drawdown."""
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve / peak) - 1
    return float(drawdown.min())


def get_performance_metrics(returns_series: pd.Series, equity_curve: pd.Series) -> dict:
    """Calculate annualized return, volatility, Sharpe ratio, and Max Drawdown."""
    ann_return = returns_series.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0
    max_dd = calculate_drawdown(equity_curve)

    return {
        "Annualized_Return": round(float(ann_return), 4),
        "Annualized_Volatility": round(float(ann_vol), 4),
        "Sharpe_Ratio": round(float(sharpe), 4),
        "Max_Drawdown": round(float(max_dd), 4)
    }


def main():
    print("Loading historical data for backtesting...")
    
    # Check for ensemble history first, fall back to single-model history if needed
    try:
        df = pd.read_csv("master_ensemble_history.csv", index_col=0, parse_dates=True)
        print("Loaded 'master_ensemble_history.csv'")
    except FileNotFoundError:
        try:
            df = pd.read_csv("regime_history.csv", index_col=0, parse_dates=True)
            print("Fallback: Loaded 'regime_history.csv'")
        except FileNotFoundError:
            print("Error: Neither 'master_ensemble_history.csv' nor 'regime_history.csv' found.")
            return

    # Find the regime name column
    regime_col = None
    for candidate in ["Actionable_Regime_Name", "Regime_Name", "Raw_Regime_Name"]:
        if candidate in df.columns:
            regime_col = candidate
            break

    if not regime_col:
        raise KeyError("Could not find a valid regime column in CSV.")

    print(f"Testing performance using regime column: '{regime_col}'")

    # Production allocation map for 3-class Ensemble and 4-class legacy
    allocation_map = {
        # 3-Class Ensemble Targets
        "High_Momentum_Bull_Breakout": 1.0,     # 100% Long
        "Choppy_or_Steady_Consolidation": 0.5,  # 50% Capital (Defensive)
        "True_Bearish_Distribution": 0.0,       # 100% Cash (Avoid Crash)

        # Legacy 4-class names (for backward compatibility)
        "High_Conviction_Bull": 1.0,
        "Steady_Uptrend": 0.8,
        "Pre_Breakout_Accumulation": 1.0,
        "Choppy_Consolidation": 0.5
    }

    # Map allocations (default 0.5 if unrecognized)
    df["Target_Position"] = df[regime_col].map(allocation_map).fillna(0.5)

    # 1-day lag to eliminate lookahead bias (execute tomorrow on today's signal)
    df["Actual_Position"] = df["Target_Position"].shift(1).fillna(0.5)

    # Calculate returns
    df["Strategy_Returns"] = df["Actual_Position"] * df["Returns"]
    df["Benchmark_Equity"] = (1 + df["Returns"]).cumprod()
    df["Strategy_Equity"] = (1 + df["Strategy_Returns"]).cumprod()

    # Drop starting NaN from shifts
    df = df.dropna(subset=["Strategy_Returns", "Returns"])

    bench_metrics = get_performance_metrics(df["Returns"], df["Benchmark_Equity"])
    strat_metrics = get_performance_metrics(df["Strategy_Returns"], df["Strategy_Equity"])

    report = {
        "backtest_run_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_trading_days": len(df),
        "benchmark_buy_and_hold": bench_metrics,
        "ensemble_strategy": strat_metrics,
        "strategy_edge": {
            "cagr_difference": round(strat_metrics["Annualized_Return"] - bench_metrics["Annualized_Return"], 4),
            "max_drawdown_reduction": round(abs(bench_metrics["Max_Drawdown"]) - abs(strat_metrics["Max_Drawdown"]), 4)
        }
    }

    with open("backtest_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Generate visual performance comparison chart
    plt.figure(figsize=(12, 6))
    plt.plot(
        df.index,
        df["Benchmark_Equity"],
        label=f"Nifty 50 Buy & Hold (Sharpe: {bench_metrics['Sharpe_Ratio']}, Max DD: {bench_metrics['Max_Drawdown']*100:.1f}%)",
        color="#7f8c8d",
        alpha=0.65,
        linewidth=1.2
    )
    plt.plot(
        df.index,
        df["Strategy_Equity"],
        label=f"Hive Mind Strategy (Sharpe: {strat_metrics['Sharpe_Ratio']}, Max DD: {strat_metrics['Max_Drawdown']*100:.1f}%)",
        color="#2ecc71",
        linewidth=1.8
    )

    plt.title("Nifty 50: Hive Mind Ensemble Strategy vs. Buy & Hold", fontsize=14, pad=12)
    plt.ylabel("Cumulative Growth (Log Scale)", fontsize=11)
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.2, linestyle="--")
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()

    plt.savefig("backtest_chart.png", dpi=250)
    print("Backtest complete! Created 'backtest_report.json' and 'backtest_chart.png'.")
    print(f"Strategy Sharpe: {strat_metrics['Sharpe_Ratio']} vs Benchmark: {bench_metrics['Sharpe_Ratio']}")
    print(f"Drawdown reduction: {report['strategy_edge']['max_drawdown_reduction']*100:.1f}% lower.")


if __name__ == "__main__":
    main()
