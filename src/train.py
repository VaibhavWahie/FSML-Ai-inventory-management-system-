import pickle
import logging
import yaml
import os

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn

# ✅ LOCAL MLFLOW
mlflow.set_tracking_uri("file:./mlruns")

# ✅ EXPERIMENT
mlflow.set_experiment("inventory")

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
        with mlflow.start_run(run_name="inventory_run"):

            params = load_params()

            rf_params_list = params["random_forest"]
            xgb_params_list = params["xgboost"]

            # Prepare data
            X = df.drop(columns=["sales", "date", "target"])
            y = df["target"]

            split = int(len(df) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            best_model = None
            best_rmse = float("inf")
            best_model_name = ""

            # 🔥 RANDOM FOREST
            for i, rf_params in enumerate(rf_params_list):
                print(f"Training RF {i}: {rf_params}")

                model = RandomForestRegressor(**rf_params)
                model.fit(X_train, y_train)

                preds = model.predict(X_test)
                rmse = mean_squared_error(y_test, preds) ** 0.5

                print(f"RF {i} RMSE: {rmse}")

                mlflow.log_params({f"rf_{i}": str(rf_params)})
                mlflow.log_metric(f"rf_rmse_{i}", rmse)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_model_name = f"RF_{i}"

            # 🔥 IMPROVED XGBOOST
            for i, xgb_params in enumerate(xgb_params_list):
                print(f"Training XGB {i}: {xgb_params}")

                model = XGBRegressor(
                    **xgb_params,
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)

                preds = model.predict(X_test)
                rmse = mean_squared_error(y_test, preds) ** 0.5

                print(f"XGB {i} RMSE: {rmse}")

                mlflow.log_params({f"xgb_{i}": str(xgb_params)})
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