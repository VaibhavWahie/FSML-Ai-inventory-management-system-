import logging

def create_features(df):
    try:
        logging.info("Creating features...")

        # Time features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

        # Lag features
        df["lag_1"] = df.groupby(["store_nbr", "family"])["sales"].shift(1)
        df["lag_7"] = df.groupby(["store_nbr", "family"])["sales"].shift(7)

        # Rolling mean
        df["rolling_mean_7"] = df.groupby(["store_nbr", "family"])["sales"].transform(
            lambda x: x.rolling(7).mean()
        )

        # Drop NA from lagging
        df = df.dropna()

        logging.info("Feature engineering done")
        return df

    except Exception as e:
        logging.error(f"Feature error: {e}")
        raise e