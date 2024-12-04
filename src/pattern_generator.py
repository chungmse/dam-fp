import pandas as pd
import numpy as np


def classify_price_movement(row):
    """
    Classify daily price movement based on close price change
    """
    daily_return = ((row["close"] - row["open"]) / row["open"]) * 100

    if daily_return > 5:
        return "PRICE_STRONG_UP"
    elif daily_return > 2:
        return "PRICE_UP"
    elif daily_return < -5:
        return "PRICE_STRONG_DOWN"
    elif daily_return < -2:
        return "PRICE_DOWN"
    else:
        return "PRICE_SIDEWAYS"


def classify_rsi(value):
    """
    Classify RSI values into categories
    """
    if value >= 70:
        return "RSI_OVERBOUGHT"
    elif value >= 60:
        return "RSI_HIGH"
    elif value <= 30:
        return "RSI_OVERSOLD"
    elif value <= 40:
        return "RSI_LOW"
    else:
        return "RSI_NEUTRAL"


def classify_macd(row):
    """
    Classify MACD patterns
    """
    patterns = []
    if row["MACD"] > row["MACD_Signal"]:
        patterns.append("MACD_BULLISH")
    else:
        patterns.append("MACD_BEARISH")

    if row["MACD_Histogram"] > 0:
        patterns.append("MACD_HIST_POSITIVE")
    else:
        patterns.append("MACD_HIST_NEGATIVE")

    return patterns


def classify_bollinger_bands(row):
    """
    Classify price position relative to Bollinger Bands
    """
    price = row["close"]
    if price > row["BB_Upper"]:
        return "BB_ABOVE"
    elif price < row["BB_Lower"]:
        return "BB_BELOW"
    else:
        return "BB_INSIDE"


def classify_volume(row):
    """
    Classify volume based on Volume Ratio
    """
    volume_ratio = row["Volume_Ratio"]
    if volume_ratio > 2:
        return "VOL_VERY_HIGH"
    elif volume_ratio > 1.5:
        return "VOL_HIGH"
    elif volume_ratio < 0.5:
        return "VOL_VERY_LOW"
    elif volume_ratio < 0.75:
        return "VOL_LOW"
    else:
        return "VOL_NORMAL"


def generate_patterns(df):
    """
    Generate all patterns for each day
    """
    patterns_df = pd.DataFrame()
    patterns_df["date"] = df["date"]

    # Generate different patterns
    patterns_df["price_pattern"] = df.apply(classify_price_movement, axis=1)
    patterns_df["rsi_pattern"] = df["RSI"].apply(classify_rsi)
    patterns_df["bb_pattern"] = df.apply(classify_bollinger_bands, axis=1)
    patterns_df["volume_pattern"] = df.apply(classify_volume, axis=1)

    # Handle MACD patterns (returns list)
    macd_patterns = df.apply(classify_macd, axis=1)
    patterns_df["macd_patterns"] = macd_patterns

    # Create transaction-like format (combine all patterns into a list)
    def combine_patterns(row):
        patterns = [
            row["price_pattern"],
            row["rsi_pattern"],
            row["bb_pattern"],
            row["volume_pattern"],
        ]
        patterns.extend(row["macd_patterns"])
        return patterns

    patterns_df["patterns"] = patterns_df.apply(combine_patterns, axis=1)

    return patterns_df


def main():
    # Read data with indicators
    input_file = "outputs/BINANCE_BTCUSDT_D1_with_indicators.csv"
    output_file = "outputs/BINANCE_BTCUSDT_D1_with_patterns.csv"

    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)
    df["date"] = pd.to_datetime(df["date"])

    # Generate patterns
    print("Generating patterns...")
    patterns_df = generate_patterns(df)

    # Save to new file
    print(f"Saving patterns to {output_file}")
    patterns_df.to_csv(output_file, index=False)

    # Print summary
    print("\nPattern categories generated:")
    summary_columns = ["price_pattern", "rsi_pattern", "bb_pattern", "volume_pattern"]
    for column in summary_columns:
        unique_patterns = patterns_df[column].unique()
        print(f"\n{column}:")
        for pattern in sorted(unique_patterns):
            print(f"- {pattern}")

    # Print MACD patterns separately
    print("\nMACD patterns:")
    print("- MACD_BULLISH")
    print("- MACD_BEARISH")
    print("- MACD_HIST_POSITIVE")
    print("- MACD_HIST_NEGATIVE")

    print("\nDone!")


if __name__ == "__main__":
    main()
