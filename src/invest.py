import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta

# Configuration Constants
LOOKBACK_DAYS = 365  # Number of days to analyze
INITIAL_CAPITAL = 100000  # Initial investment in USD
BUY_RATIO = 0.3  # Percentage of available capital to invest in each buy signal
SELL_RATIO = 0.3  # Percentage of BTC holdings to sell in each sell signal


class InvestmentAnalyzer:
    def __init__(self, rules_file, patterns_file, price_file, initial_capital=100000):
        """
        Initialize with necessary data files and initial capital
        """
        self.rules_df = pd.read_csv(rules_file)
        self.patterns_df = pd.read_csv(patterns_file)
        self.price_df = pd.read_csv(price_file)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.btc_holdings = 0
        self.trades = []

        # Convert patterns string to list
        self.patterns_df["patterns"] = self.patterns_df["patterns"].apply(eval)

        # Prepare price data for plotting
        self.price_df["date"] = pd.to_datetime(self.price_df["date"])

        # Filter last year's data
        last_date = self.price_df["date"].max()
        start_date = last_date - timedelta(days=LOOKBACK_DAYS)
        self.price_df = self.price_df[self.price_df["date"] >= start_date]
        self.patterns_df = self.patterns_df[
            pd.to_datetime(self.patterns_df["date"]) >= start_date
        ]

        # Set date as index for mplfinance
        self.price_df.set_index("date", inplace=True)

    def identify_signals(self):
        """
        Identify both buy and sell signals
        """
        signals = []

        for idx, row in self.patterns_df.iterrows():
            date = row["date"]
            current_patterns = set(row["patterns"])
            signal_type = None
            signal_strength = 0

            # Check buy conditions
            if "RSI_OVERSOLD" in current_patterns and "BB_BELOW" in current_patterns:
                signal_type = "BUY"
                signal_strength = 1.0
            elif (
                "PRICE_STRONG_DOWN" in current_patterns
                and "RSI_OVERSOLD" in current_patterns
            ):
                signal_type = "BUY"
                signal_strength = 0.8

            # Check sell conditions
            elif (
                "RSI_OVERBOUGHT" in current_patterns and "BB_ABOVE" in current_patterns
            ):
                signal_type = "SELL"
                signal_strength = 1.0
            elif (
                "PRICE_STRONG_UP" in current_patterns
                and "RSI_OVERBOUGHT" in current_patterns
            ):
                signal_type = "SELL"
                signal_strength = 0.8

            if signal_type:
                close_price = self.price_df.loc[pd.to_datetime(date)]["close"]
                signals.append(
                    {
                        "date": pd.to_datetime(date),
                        "type": signal_type,
                        "strength": signal_strength,
                        "patterns": list(current_patterns),
                        "price": float(close_price),
                    }
                )

        return pd.DataFrame(signals)

    def plot_signals_with_performance(self):
        """
        Plot candlestick chart with buy/sell signals and performance
        """
        signals = self.identify_signals()
        final_value, capital_history, trades = self.backtest_strategy()

        # Prepare the candlestick data
        df_plot = self.price_df.copy()

        # Create the figure
        kwargs = dict(
            type="candle",
            volume=True,
            title="Bitcoin Price with Trading Signals",
            ylabel="Price (USD)",
            ylabel_lower="Volume",
            figratio=(15, 10),
            figscale=1,
        )

        # Prepare the buy signals
        buy_signals = signals[signals["type"] == "BUY"]
        if not buy_signals.empty:
            buy_markers = pd.DataFrame(index=df_plot.index, columns=["buy"])
            buy_markers.loc[buy_signals["date"], "buy"] = (
                df_plot.loc[buy_signals["date"], "low"] * 0.99
            )
            ap_buy = mpf.make_addplot(
                buy_markers, type="scatter", markersize=100, marker="^", color="g"
            )
            kwargs["addplot"] = [ap_buy]

        # Prepare the sell signals
        sell_signals = signals[signals["type"] == "SELL"]
        if not sell_signals.empty:
            sell_markers = pd.DataFrame(index=df_plot.index, columns=["sell"])
            sell_markers.loc[sell_signals["date"], "sell"] = (
                df_plot.loc[sell_signals["date"], "high"] * 1.01
            )
            ap_sell = mpf.make_addplot(
                sell_markers, type="scatter", markersize=100, marker="v", color="r"
            )
            kwargs["addplot"] = kwargs.get("addplot", []) + [ap_sell]

        # Plot the candlestick chart
        mpf.plot(
            df_plot, **kwargs, savefig="outputs/trading_signals_with_performance.png"
        )

        # Create portfolio value plot
        plt.figure(figsize=(15, 5))
        plt.plot(
            capital_history["date"],
            capital_history["total_value"],
            label="Portfolio Value",
            color="blue",
        )
        plt.axhline(
            y=self.initial_capital, color="r", linestyle="--", label="Initial Capital"
        )
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("USD")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("outputs/portfolio_value.png")
        plt.close()

        return final_value, trades

    def backtest_strategy(self):
        """
        Backtest the trading strategy with 30% allocation
        """
        signals = self.identify_signals()
        capital_history = []
        current_capital = self.initial_capital
        btc_holdings = 0

        for _, signal in signals.iterrows():
            if signal["type"] == "BUY" and current_capital > 0:
                # Use 30% of current capital to buy BTC
                investment = current_capital * BUY_RATIO
                btc_amount = investment / signal["price"]
                btc_holdings += btc_amount
                current_capital -= investment

                self.trades.append(
                    {
                        "date": signal["date"],
                        "type": "BUY",
                        "price": signal["price"],
                        "btc_amount": btc_amount,
                        "investment": investment,
                        "current_capital": current_capital,
                        "btc_holdings": btc_holdings,
                        "total_value": current_capital
                        + (btc_holdings * signal["price"]),
                    }
                )

            elif signal["type"] == "SELL" and btc_holdings > 0:
                # Sell 30% of BTC holdings
                btc_to_sell = btc_holdings * SELL_RATIO
                sale_value = btc_to_sell * signal["price"]
                current_capital += sale_value
                btc_holdings -= btc_to_sell

                self.trades.append(
                    {
                        "date": signal["date"],
                        "type": "SELL",
                        "price": signal["price"],
                        "btc_amount": btc_to_sell,
                        "sale_value": sale_value,
                        "current_capital": current_capital,
                        "btc_holdings": btc_holdings,
                        "total_value": current_capital
                        + (btc_holdings * signal["price"]),
                    }
                )

            capital_history.append(
                {
                    "date": signal["date"],
                    "total_value": current_capital + (btc_holdings * signal["price"]),
                }
            )

        # Calculate final portfolio value
        if btc_holdings > 0:
            final_btc_value = btc_holdings * self.price_df.iloc[-1]["close"]
            final_value = current_capital + final_btc_value
        else:
            final_value = current_capital

        return final_value, pd.DataFrame(capital_history), pd.DataFrame(self.trades)

    def generate_performance_report(self, final_value, trades):
        """
        Generate a detailed performance report
        """
        total_return = (
            (final_value - self.initial_capital) / self.initial_capital
        ) * 100
        total_trades = len(trades)

        # Calculate additional statistics
        buy_trades = trades[trades["type"] == "BUY"]
        sell_trades = trades[trades["type"] == "SELL"]

        report = [
            "Trading Performance Report",
            "=" * 50,
            f"\nInitial Capital: ${self.initial_capital:,.2f}",
            f"Final Portfolio Value: ${final_value:,.2f}",
            f"Total Return: {total_return:.2f}%",
            f"Total Number of Trades: {total_trades}",
            f"Number of Buy Trades: {len(buy_trades)}",
            f"Number of Sell Trades: {len(sell_trades)}",
            f"\nProfit/Loss: ${(final_value - self.initial_capital):,.2f}",
            "\nDetailed Trade History:",
            "-" * 30,
        ]

        for _, trade in trades.iterrows():
            if trade["type"] == "BUY":
                report.append(f"\nBUY at {trade['date']}")
                report.append(f"Price: ${trade['price']:,.2f}")
                report.append(f"BTC Amount: {trade['btc_amount']:.6f}")
                report.append(f"Investment: ${trade['investment']:,.2f}")
            else:
                report.append(f"\nSELL at {trade['date']}")
                report.append(f"Price: ${trade['price']:,.2f}")
                report.append(f"BTC Amount: {trade['btc_amount']:.6f}")
                report.append(f"Sale Value: ${trade['sale_value']:,.2f}")
            report.append(f"Remaining BTC: {trade['btc_holdings']:.6f}")
            report.append(f"Cash Balance: ${trade['current_capital']:,.2f}")
            report.append(f"Total Portfolio Value: ${trade['total_value']:,.2f}")

        with open("outputs/performance_report.txt", "w") as f:
            f.write("\n".join(report))


def main():
    rules_file = "outputs/BINANCE_BTCUSDT_D1_rules.csv"
    patterns_file = "outputs/BINANCE_BTCUSDT_D1_with_patterns.csv"
    price_file = "datasets/BINANCE_BTCUSDT_D1.csv"

    print("Starting investment analysis...")
    analyzer = InvestmentAnalyzer(
        rules_file, patterns_file, price_file, initial_capital=100000
    )

    print("Running backtest and generating visualizations...")
    final_value, trades = analyzer.plot_signals_with_performance()

    print("Generating performance report...")
    analyzer.generate_performance_report(final_value, trades)

    profit_loss = final_value - 100000
    profit_loss_percent = (profit_loss / 100000) * 100

    print("\nBacktest Results:")
    print(f"Lookback Period: {LOOKBACK_DAYS} days")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Buy Ratio: {BUY_RATIO*100}%")
    print(f"Sell Ratio: {SELL_RATIO*100}%")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Profit/Loss: ${profit_loss:,.2f} ({profit_loss_percent:.2f}%)")
    print("\nDetailed report saved in outputs/performance_report.txt")


if __name__ == "__main__":
    main()
