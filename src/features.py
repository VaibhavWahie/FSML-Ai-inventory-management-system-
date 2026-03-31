import logging

def create_features(df):
    try:
        logging.info("Creating features...")

        # Sort first (VERY IMPORTANT)
        df = df.sort_values(["store_nbr", "family", "date"])

        # Time features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # 🔥 Lag features (MORE POWERFUL)
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            df[f"lag_{lag}"] = df.groupby(["store_nbr", "family"])["sales"].shift(lag)

        # 🔥 Rolling features (FIXED - NO LEAKAGE)
        df["rolling_mean_7"] = df.groupby(["store_nbr", "family"])["sales"].transform(
            lambda x: x.shift(1).rolling(7).mean()
        )

        df["rolling_mean_14"] = df.groupby(["store_nbr", "family"])["sales"].transform(
            lambda x: x.shift(1).rolling(14).mean()
        )

        df["rolling_std_7"] = df.groupby(["store_nbr", "family"])["sales"].transform(
            lambda x: x.shift(1).rolling(7).std()
        )

        # 🔥 Target (7-day ahead)
        df["target"] = df.groupby(["store_nbr", "family"])["sales"].shift(-7)

        # Drop NA
        df = df.dropna()

        return df

    except Exception as e:
        logging.error(f"Feature error: {e}")
        raise e