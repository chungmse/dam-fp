import pandas as pd
import numpy as np


def add_moving_averages(df):
    """
    Add various Moving Averages: MA7, MA14, MA21, MA50, MA200
    """
    for period in [7, 14, 21, 50, 200]:
        df[f"MA{period}"] = df["close"].rolling(window=period).mean()
    return df


def add_rsi(df, period=14):
    """
    Add Relative Strength Index (RSI)
    """
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df):
    """
    Add Moving Average Convergence Divergence (MACD)
    """
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(df, period=20, std_dev=2):
    """
    Add Bollinger Bands
    """
    df["BB_Middle"] = df["close"].rolling(window=period).mean()
    bb_std = df["close"].rolling(window=period).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * std_dev)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * std_dev)
    return df


def add_volume_indicators(df):
    """
    Add Volume-based indicators
    """
    # Volume Moving Average
    df["Volume_MA20"] = df["volume"].rolling(window=20).mean()
    # Volume Ratio
    df["Volume_Ratio"] = df["volume"] / df["Volume_MA20"]
    return df


def main():
    # Read the original data
    input_file = "outputs/BINANCE_BTCUSDT_D1_preprocessed.csv"
    output_file = "outputs/BINANCE_BTCUSDT_D1_with_indicators.csv"

    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)

    # Convert date column to datetime if it's not already
    df["date"] = pd.to_datetime(df["date"])

    # Add technical indicators
    print("Adding technical indicators...")
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_indicators(df)

    # Drop any rows with NaN values that resulted from calculations
    df = df.dropna()

    # Save to new file
    print(f"Saving data with indicators to {output_file}")
    df.to_csv(output_file, index=False)
    print("Done!")

    # Print summary of added indicators
    print("\nAdded indicators:")
    new_columns = set(df.columns) - set(
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "number_of_trades",
            "taker_volume",
            "taker_quote_volume",
            "symbol",
        ]
    )
    for col in sorted(new_columns):
        print(f"- {col}")


if __name__ == "__main__":
    main()
