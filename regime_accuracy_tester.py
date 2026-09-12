import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

plt.switch_backend("Agg")

def calculate_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve / peak) - 1
    return float(drawdown.min())

def get_performance_metrics(returns_series: pd.Series, equity_curve: pd.Series) -> dict:
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
    print("Loading 20-Node Ensemble historical data...")
    try:
        df = pd.read_csv("master_ensemble_history.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'master_ensemble_history.csv' not found. Run meta_ensemble.py first.")
        return

    # Production allocation map for the Meta-Learner outputs:
    allocation_map = {
        "High_Momentum_Bull_Breakout": 1.0,     # 100% Long
        "Choppy_or_Steady_Consolidation": 0.5,  # 50% Capital (Risk reduction)
        "True_Bearish_Distribution": 0.0        # 100% Cash (Avoid the crash)
    }

    df["Target_Position"] = df["Actionable_Regime_Name"].map(allocation_map).fillna(0.5)

    # 1-day lag to eliminate lookahead bias
    df["Actual_Position"] = df["Target_Position"].shift(1).fillna(0.5)

    df["Strategy_Returns"] = df["Actual_Position"] * df["Returns"]
    df["Benchmark_Equity"] = (1 + df["Returns"]).cumprod()
    df["Strategy_Equity"] = (1 + df["Strategy_Returns"]).cumprod()

    bench_metrics = get_performance_metrics(df["Returns"], df["Benchmark_Equity"])
    strat_metrics = get_performance_metrics(df["Strategy_Returns"], df["Strategy_Equity"])

    report = {
        "backtest_run_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_trading_days": len(df),
        "benchmark_buy_and_hold": bench_metrics,
        "ensemble_matrix_strategy": strat_metrics,
        "strategy_edge": {
            "cagr_difference": round(strat_metrics["Annualized_Return"] - bench_metrics["Annualized_Return"], 4),
            "max_drawdown_reduction": round(abs(bench_metrics["Max_Drawdown"]) - abs(strat_metrics["Max_Drawdown"]), 4)
        }
    }

    with open("backtest_report.json", "w") as f:
        json.dump(report, f, indent=2)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Benchmark_Equity"], label=f"Nifty Buy & Hold (Sharpe: {bench_metrics['Sharpe_Ratio']})", color="#7f8c8d", alpha=0.65, linewidth=1.2)
    plt.plot(df.index, df["Strategy_Equity"], label=f"20-Node Ensemble (Sharpe: {strat_metrics['Sharpe_Ratio']})", color="#2ecc71", linewidth=1.8)

    plt.title("Nifty 50: 20-Node Meta-Ensemble Strategy vs. Buy & Hold", fontsize=14, pad=12)
    plt.ylabel("Cumulative Growth (Log Scale)", fontsize=11)
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.2, linestyle="--")
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()

    plt.savefig("backtest_chart.png", dpi=250)
    print("Backtest complete! Chart saved to 'backtest_chart.png'.")
    print(f"Max Drawdown Reduction: {report['strategy_edge']['max_drawdown_reduction']*100:.1f}%")

if __name__ == "__main__":
    main()
