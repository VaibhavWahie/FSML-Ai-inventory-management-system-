import pickle
import logging
import yaml
import os

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train_model(df):
    try:
        mlflow.set_experiment("inventory-model")

        with mlflow.start_run():

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

            # 🔥 RANDOM FOREST (CPU)
            for i, rf_params in enumerate(rf_params_list):
                print(f"Training RF {i}: {rf_params}")

                model = RandomForestRegressor(**rf_params)
                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                mse = mean_squared_error(y_test, preds)
                rmse = mse ** 0.5

                print(f"RF {i} RMSE: {rmse}")

                mlflow.log_param(f"rf_params_{i}", rf_params)
                mlflow.log_metric(f"rf_rmse_{i}", rmse)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_model_name = f"RF_{i}"

            # 🔥 XGBOOST (GPU ENABLED)
            for i, xgb_params in enumerate(xgb_params_list):
                print(f"Training XGB {i}: {xgb_params}")

                model = XGBRegressor(
                    **xgb_params,
                    tree_method="gpu_hist",
                    predictor="gpu_predictor"
                )

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                mse = mean_squared_error(y_test, preds)
                rmse = mse ** 0.5

                print(f"XGB {i} RMSE: {rmse}")

                mlflow.log_param(f"xgb_params_{i}", xgb_params)
                mlflow.log_metric(f"xgb_rmse_{i}", rmse)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_model_name = f"XGB_{i}"

            # Save model
            os.makedirs("models", exist_ok=True)

            with open("models/model_v1.pkl", "wb") as f:
                pickle.dump(best_model, f)

            mlflow.sklearn.log_model(best_model, "model")

            print("\n✅ TRAINING COMPLETE")
            print(f"✅ Best Model: {best_model_name}")
            print(f"✅ Best RMSE: {best_rmse}")

            return best_model

    except Exception as e:
        logging.error(f"Training error: {e}")
        raise e