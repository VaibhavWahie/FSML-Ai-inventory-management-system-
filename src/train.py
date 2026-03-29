import pickle
import logging
import yaml

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def train_model(df):
    try:
        logging.info("Training started...")

        # Load params
        params = load_params()

        rf_params_list = params["random_forest"]
        xgb_params_list = params["xgboost"]

        # Prepare data
        X = df.drop(columns=["sales", "date"])
        y = df["sales"]

        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        best_model = None
        best_rmse = float("inf")
        best_model_name = ""

        # 🔥 RANDOM FOREST LOOP
        for i, rf_params in enumerate(rf_params_list):
            logging.info(f"Training RandomForest config {i}: {rf_params}")

            model = RandomForestRegressor(**rf_params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds, squared=False)

            logging.info(f"RF Config {i} RMSE: {rmse}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_name = f"RandomForest_{i}"

        # 🔥 XGBOOST LOOP
        for i, xgb_params in enumerate(xgb_params_list):
            logging.info(f"Training XGBoost config {i}: {xgb_params}")

            model = XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds, squared=False)

            logging.info(f"XGB Config {i} RMSE: {rmse}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_name = f"XGBoost_{i}"

        # Save best model
        with open("models/model_v1.pkl", "wb") as f:
            pickle.dump(best_model, f)

        logging.info(f"Best model: {best_model_name} with RMSE: {best_rmse}")
        logging.info("Model saved successfully")

        return best_model

    except Exception as e:
        logging.error(f"Training error: {e}")
        raise e