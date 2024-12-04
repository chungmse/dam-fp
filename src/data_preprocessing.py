import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        """
        print("Checking for missing values...")
        missing_count = df.isnull().sum()
        if missing_count.any():
            print("Found missing values:")
            print(missing_count[missing_count > 0])
            # Fill missing values using forward fill method
            df = df.fillna(method="ffill")
            # If there are still missing values at the start, use backward fill
            df = df.fillna(method="bfill")
        else:
            print("No missing values found.")
        return df

    def remove_outliers(
        self, df, columns=["open", "high", "low", "close", "volume"], n_std=3
    ):
        """
        Remove outliers using z-score method
        """
        print("\nChecking for outliers...")
        df_clean = df.copy()
        total_outliers = 0

        for column in columns:
            if column in df.columns:
                z_scores = np.abs(
                    (df_clean[column] - df_clean[column].mean())
                    / df_clean[column].std()
                )
                outliers_mask = z_scores > n_std
                outliers_count = outliers_mask.sum()
                if outliers_count > 0:
                    df_clean = df_clean[~outliers_mask]
                    total_outliers += outliers_count
                    print(f"Removed {outliers_count} outliers from {column}")

        print(f"Total outliers removed: {total_outliers}")
        return df_clean

    def add_derived_features(self, df):
        """
        Add derived features that might be useful for analysis
        """
        print("\nAdding derived features...")

        # Price changes
        df["price_change"] = df["close"] - df["open"]
        df["price_change_pct"] = (df["close"] - df["open"]) / df["open"] * 100

        # Trading ranges
        df["daily_range"] = df["high"] - df["low"]
        df["daily_range_pct"] = (df["high"] - df["low"]) / df["open"] * 100

        # Volume analysis
        df["volume_price_ratio"] = df["volume"] / df["close"]

        # Average price
        df["avg_price"] = (df["high"] + df["low"] + df["close"]) / 3

        # Price momentum (rate of change)
        df["price_momentum"] = df["close"].pct_change()

        print(
            "Added new features:",
            [
                "price_change",
                "price_change_pct",
                "daily_range",
                "daily_range_pct",
                "volume_price_ratio",
                "avg_price",
                "price_momentum",
            ],
        )
        return df

    def normalize_features(self, df, columns=["volume", "daily_range"]):
        """
        Normalize selected features using StandardScaler
        """
        print("\nNormalizing features...")
        df_normalized = df.copy()
        for column in columns:
            if column in df.columns:
                df_normalized[f"{column}_normalized"] = self.scaler.fit_transform(
                    df[[column]]
                )
                print(f"Normalized {column}")
        return df_normalized

    def process_data(self, input_file, output_file):
        """
        Main preprocessing pipeline
        """
        print(f"\nReading data from {input_file}")
        df = pd.read_csv(input_file)

        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Apply preprocessing steps
        df = self.handle_missing_values(df)
        df = self.remove_outliers(df)
        df = self.add_derived_features(df)
        df = self.normalize_features(df)

        # Save preprocessed data
        print(f"\nSaving preprocessed data to {output_file}")
        df.to_csv(output_file, index=False)

        print("\nPreprocessing complete!")
        print(f"Original shape: {len(df)} rows, {len(df.columns)} columns")
        print(
            "New features added:",
            sorted(set(df.columns) - set(pd.read_csv(input_file).columns)),
        )

        return df


def main():
    input_file = "datasets/BINANCE_BTCUSDT_D1.csv"
    output_file = "outputs/BINANCE_BTCUSDT_D1_preprocessed.csv"

    preprocessor = DataPreprocessor()
    preprocessor.process_data(input_file, output_file)


if __name__ == "__main__":
    main()
