import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

TICKERS = [
    "ABB.NS", "ADANIENSOL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "ATGL.NS",
    "AMBUJACEM.NS", "AWL.NS", "BAJAJHLDNG.NS", "BANKBARODA.NS", "BEL.NS",
    "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "DLF.NS", "DABUR.NS",
    "GAIL.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IOC.NS", "IRCTC.NS",
    "JINDALSTEL.NS", "JIOFIN.NS", "KOTAKBANK.NS", "LICI.NS", "MARICO.NS",
    "MUTHOOTFIN.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS", "SBICARD.NS",
    "SRF.NS", "MOTHERSON.NS", "SHREECEM.NS", "SIEMENS.NS", "TATAELXSI.NS",
    "TVSMOTOR.NS", "TORNTPHARM.NS", "TRENT.NS", "UBL.NS", "MCDOWELL-N.NS",
    "VBL.NS", "VEDL.NS", "ZOMATO.NS", "ZYDUSLIFE.NS", "TATACHEM.NS"
]

HOLDING_PERIOD = 5
MAX_CONCURRENT_POSITIONS = 10
STOP_LOSS = -0.04  # 4% structural stop-loss


def calculate_metrics(returns_series: pd.Series, equity_curve: pd.Series) -> dict:
    ann_return = returns_series.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0
    peak = equity_curve.expanding(min_periods=1).max()
    max_dd = ((equity_curve / peak) - 1).min()

    return {
        "Annualized_Return": round(float(ann_return), 4),
        "Annualized_Volatility": round(float(ann_vol), 4),
        "Sharpe_Ratio": round(float(sharpe), 4),
        "Max_Drawdown": round(float(max_dd), 4)
    }


def main():
    print("Initializing Regime-Conditioned Stock Strategy Backtest...")

    try:
        regime_df = pd.read_csv("master_ensemble_history.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'master_ensemble_history.csv' not found. Run meta_ensemble.py first.")
        return

    regime_col = "Actionable_Regime_Name"
    if regime_col not in regime_df.columns:
        raise KeyError(f"'{regime_col}' column missing from master_ensemble_history.csv.")

    start_date = regime_df.index.min().strftime("%Y-%m-%d")
    end_date = regime_df.index.max().strftime("%Y-%m-%d")

    print(f"Downloading historical data for {len(TICKERS)} tickers ({start_date} to {end_date})...")
    data = yf.download(TICKERS, start=start_date, end=end_date, group_by="ticker", auto_adjust=True, progress=False)

    stock_closes = pd.DataFrame()
    stock_volumes = pd.DataFrame()

    for ticker in TICKERS:
        try:
            sub = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
            if not sub.empty and "Close" in sub.columns:
                stock_closes[ticker] = sub["Close"]
                stock_volumes[ticker] = sub["Volume"]
        except Exception:
            continue

    common_dates = regime_df.index.intersection(stock_closes.index).sort_values()
    stock_closes = stock_closes.reindex(common_dates).ffill()
    stock_volumes = stock_volumes.reindex(common_dates).ffill()
    regime_series = regime_df.loc[common_dates, regime_col]

    daily_portfolio_returns = []
    active_positions = []  # list of dicts: {ticker, entry_price, entry_idx, stop_price}
    all_trade_pnl = []

    for i in range(25, len(common_dates)):
        current_date = common_dates[i]
        regime = regime_series.iloc[i - 1]  # Signal generated at prior close to avoid lookahead

        # 1. Update & manage active positions
        realized_return_today = 0.0
        open_positions_count = len(active_positions)
        surviving_positions = []

        for pos in active_positions:
            t = pos["ticker"]
            entry_px = pos["entry_price"]
            curr_px = stock_closes[t].iloc[i]
            prev_px = stock_closes[t].iloc[i - 1]

            pos_daily_ret = (curr_px / prev_px) - 1
            realized_return_today += pos_daily_ret / MAX_CONCURRENT_POSITIONS

            unrealized_total_ret = (curr_px / entry_px) - 1
            days_held = i - pos["entry_idx"]

            # Exit criteria: Stop-loss hit or holding duration reached
            if unrealized_total_ret <= STOP_LOSS or days_held >= HOLDING_PERIOD:
                all_trade_pnl.append(unrealized_total_ret)
            else:
                surviving_positions.append(pos)

        active_positions = surviving_positions
        daily_portfolio_returns.append(realized_return_today)

        # 2. Entry Logic (Regime gated)
        if regime == "True_Bearish_Distribution":
            continue

        if len(active_positions) >= MAX_CONCURRENT_POSITIONS:
            continue

        is_high_conviction = (regime == "Choppy_or_Steady_Consolidation")
        band_threshold = 0.025 if is_high_conviction else 0.04
        vol_multiple = 2.0 if is_high_conviction else 1.5

        # Scan for candidates
        available_slots = MAX_CONCURRENT_POSITIONS - len(active_positions)
        candidates = []

        for ticker in stock_closes.columns:
            if any(p["ticker"] == ticker for p in active_positions):
                continue

            closes = stock_closes[ticker].iloc[:i]
            volumes = stock_volumes[ticker].iloc[:i]
            if len(closes) < 25:
                continue

            ema_fast = closes.ewm(span=10, adjust=False).mean()
            ema_slow = closes.ewm(span=21, adjust=False).mean()
            vol_sma = volumes.rolling(20).mean()

            # Consolidation check over last 8 bars
            recent_closes = closes.iloc[-9:-1]
            recent_fast = ema_fast.iloc[-9:-1]
            max_dev = (abs(recent_closes - recent_fast) / recent_fast).max()

            if max_dev > band_threshold:
                continue

            # Breakout check
            today_close = closes.iloc[-1]
            yesterday_close = closes.iloc[-2]
            today_vol = volumes.iloc[-1]

            bullish_alignment = ema_fast.iloc[-1] > ema_slow.iloc[-1]
            price_breakout = (today_close > ema_fast.iloc[-1]) and (yesterday_close <= ema_fast.iloc[-2])
            volume_surge = today_vol > (vol_sma.iloc[-1] * vol_multiple)

            if bullish_alignment and price_breakout and volume_surge:
                candidates.append((ticker, today_vol / (vol_sma.iloc[-1] + 1e-9)))

        # Prioritize setups with strongest relative volume
        candidates.sort(key=lambda x: x[1], reverse=True)
        for ticker, _ in candidates[:available_slots]:
            active_positions.append({
                "ticker": ticker,
                "entry_price": stock_closes[ticker].iloc[i],
                "entry_idx": i
            })

    # 3. Performance Aggregation
    perf_index = common_dates[25:]
    strat_series = pd.Series(daily_portfolio_returns, index=perf_index)
    bench_series = stock_closes.loc[perf_index].pct_change().mean(axis=1).fillna(0)

    strat_equity = (1 + strat_series).cumprod()
    bench_equity = (1 + bench_series).cumprod()

    strat_metrics = calculate_metrics(strat_series, strat_equity)
    bench_metrics = calculate_metrics(bench_series, bench_equity)

    win_trades = [p for p in all_trade_pnl if p > 0]
    win_rate = (len(win_trades) / len(all_trade_pnl) * 100) if all_trade_pnl else 0.0

    report = {
        "backtest_run_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_trading_days": len(perf_index),
        "total_trades_executed": len(all_trade_pnl),
        "trade_win_rate_pct": round(win_rate, 2),
        "benchmark_equal_weighted_next50": bench_metrics,
        "regime_filtered_stock_strategy": strat_metrics
    }

    with open("stock_backtest_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # 4. Chart Generation
    plt.figure(figsize=(12, 6))
    plt.plot(perf_index, bench_equity, label=f"Next 50 Equal Weight (Sharpe: {bench_metrics['Sharpe_Ratio']})", color="#7f8c8d", alpha=0.6, linewidth=1.2)
    plt.plot(perf_index, strat_equity, label=f"Regime Stock Strategy (Sharpe: {strat_metrics['Sharpe_Ratio']})", color="#2ed573", linewidth=1.8)
    plt.title("Nifty Next 50: Regime-Conditioned EMA Breakout Strategy vs Benchmark", fontsize=14, pad=12)
    plt.ylabel("Portfolio Cumulative Return", fontsize=11)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.2)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("stock_backtest_chart.png", dpi=250)

    print(f"Stock Backtest Complete. Total Trades: {len(all_trade_pnl)} | Win Rate: {win_rate:.1f}%")
    print(f"Strategy Sharpe: {strat_metrics['Sharpe_Ratio']} | Benchmark Sharpe: {bench_metrics['Sharpe_Ratio']}")


if __name__ == "__main__":
    main()
