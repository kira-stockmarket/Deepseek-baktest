import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Prevent GUI backend crashes on headless GitHub Actions runners
plt.switch_backend("Agg")


def calculate_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum peak-to-trough drawdown."""
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve / peak) - 1
    return float(drawdown.min())


def get_performance_metrics(returns_series: pd.Series, equity_curve: pd.Series) -> dict:
    """Compute annualized returns, volatility, Sharpe ratio, and Max Drawdown."""
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
    print("Loading historical regime data for backtesting...")
    try:
        df = pd.read_csv("regime_history.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'regime_history.csv' not found. Run deep_regime_engine.py first.")
        return

    # Backwards-compatible column detection (prefer actionable signal)
    if "Actionable_Regime_Name" in df.columns:
        regime_col = "Actionable_Regime_Name"
    elif "Regime_Name" in df.columns:
        regime_col = "Regime_Name"
    else:
        raise KeyError("Could not find a valid regime column in regime_history.csv")

    print(f"Running backtest against signal column: '{regime_col}'")

    # Production allocation map based on your empirical test results:
    # 1.0 = 100% Long, 0.8 = 80% Long / 20% Cash, 0.0 = 100% Cash / Defensive
    allocation_map = {
        "High_Conviction_Bull": 1.0,
        "Steady_Uptrend": 0.8,
        "Pre_Breakout_Accumulation": 1.0,
        "Sideways_Consolidation": 1.0,  # Legacy alias
        "Bearish_Distribution": 0.0
    }

    # Map target allocation; default to 0.5 if an unknown state appears
    df["Target_Position"] = df[regime_col].map(allocation_map).fillna(0.5)

    # 1-day lag to eliminate lookahead bias (execute tomorrow on today's signal)
    df["Actual_Position"] = df["Target_Position"].shift(1).fillna(0.5)

    # Strategy vs Benchmark performance
    df["Strategy_Returns"] = df["Actual_Position"] * df["Returns"]
    df["Benchmark_Equity"] = (1 + df["Returns"]).cumprod()
    df["Strategy_Equity"] = (1 + df["Strategy_Returns"]).cumprod()

    bench_metrics = get_performance_metrics(df["Returns"], df["Benchmark_Equity"])
    strat_metrics = get_performance_metrics(df["Strategy_Returns"], df["Strategy_Equity"])

    report = {
        "backtest_run_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "tested_regime_column": regime_col,
        "total_trading_days": len(df),
        "benchmark_buy_and_hold": bench_metrics,
        "ml_regime_strategy": strat_metrics,
        "strategy_edge": {
            "cagr_difference": round(strat_metrics["Annualized_Return"] - bench_metrics["Annualized_Return"], 4),
            "max_drawdown_reduction": round(abs(bench_metrics["Max_Drawdown"]) - abs(strat_metrics["Max_Drawdown"]), 4)
        }
    }

    with open("backtest_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Generate performance comparison chart
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
        label=f"ML Whipsaw-Filtered Strategy (Sharpe: {strat_metrics['Sharpe_Ratio']}, Max DD: {strat_metrics['Max_Drawdown']*100:.1f}%)",
        color="#2ecc71",
        linewidth=1.8
    )

    plt.title("Nifty 50: VIX-Enhanced Regime Strategy vs. Buy & Hold", fontsize=14, pad=12)
    plt.ylabel("Cumulative Growth (Log Scale)", fontsize=11)
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.2, linestyle="--")
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()

    plt.savefig("backtest_chart.png", dpi=250)
    print("Backtest complete. Saved 'backtest_report.json' and 'backtest_chart.png'.")
    print(f"Benchmark Sharpe: {bench_metrics['Sharpe_Ratio']} | ML Strategy Sharpe: {strat_metrics['Sharpe_Ratio']}")
    print(f"Max DD Improvement: {report['strategy_edge']['max_drawdown_reduction']*100:.1f}% lower drawdown.")


if __name__ == "__main__":
    main()
