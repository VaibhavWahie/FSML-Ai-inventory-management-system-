import pandas as pd
import logging

def preprocess_data(df):
    try:
        logging.info("Starting preprocessing...")

        # Convert date
        df["date"] = pd.to_datetime(df["date"])

        # Sort for time series
        df = df.sort_values(by=["store_nbr", "family", "date"])

        # Fill missing sales
        df["sales"] = df["sales"].fillna(0)

        # Fill promotion
        df["onpromotion"] = df["onpromotion"].fillna(0)

        logging.info("Preprocessing completed")
        return df

    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        raise e