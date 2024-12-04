import pandas as pd
from pymongo import MongoClient
import os

def connect_to_mongodb():
    """
    Establish connection to MongoDB
    """
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['bungbing-data-binance-spot']
        collection = db['d1']
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def fetch_data(collection):
    """
    Fetch data from MongoDB and convert to DataFrame
    """
    try:
        # Fetch all BTCUSDT records, sorted by date
        cursor = collection.find(
            {"s": "BTCUSDT"}
        ).sort("t", 1)
        
        # Convert to list of dictionaries
        data = list(cursor)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Rename columns for clarity
        df = df.rename(columns={
            "t": "date",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "q": "quote_volume",
            "n": "number_of_trades",
            "tv": "taker_volume",
            "tq": "taker_quote_volume",
            "s": "symbol"
        })
        
        # Convert date field
        df['date'] = pd.to_datetime(df['date'])
        
        # Drop MongoDB _id column
        df = df.drop('_id', axis=1)
        
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def export_to_csv(df):
    """
    Export DataFrame to CSV file
    """
    try:
        # Create datasets directory if it doesn't exist
        os.makedirs('datasets', exist_ok=True)
        
        output_file = 'datasets/BINANCE_BTCUSDT_D1.csv'
        df.to_csv(output_file, index=False)
        print(f"Data successfully exported to {output_file}")
        print(f"Total records exported: {len(df)}")
    except Exception as e:
        print(f"Error exporting data: {e}")

def main():
    # Connect to MongoDB
    collection = connect_to_mongodb()
    if collection is None:
        return
    
    # Fetch data
    df = fetch_data(collection)
    if df is None:
        return
    
    # Export to CSV
    export_to_csv(df)

if __name__ == "__main__":
    main()