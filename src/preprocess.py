import pandas as pd
import logging

def preprocess_data(df):
    try:
        logging.info("Starting preprocessing...")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["store_nbr", "family", "date"])

        df["sales"] = df["sales"].fillna(0)
        df["onpromotion"] = df["onpromotion"].fillna(0)

        # Encode categorical
        df["family"] = df["family"].astype("category").cat.codes

        return df

    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        raise e