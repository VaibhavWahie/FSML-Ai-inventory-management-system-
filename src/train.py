from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle
import logging

def train_model(df):
    try:
        logging.info("Training started...")

        # Features
        X = df.drop(columns=["sales", "date"])
        y = df["sales"]

        # Simple split
        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Model 1
        rf = RandomForestRegressor(n_estimators=50)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)

        # Model 2
        xgb = XGBRegressor(n_estimators=50)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)

        xgb_rmse = mean_squared_error(y_test, xgb_pred, squared=False)

        logging.info(f"RF RMSE: {rf_rmse}")
        logging.info(f"XGB RMSE: {xgb_rmse}")

        # Choose best
        best_model = xgb if xgb_rmse < rf_rmse else rf

        # Save model
        with open("models/model_v1.pkl", "wb") as f:
            pickle.dump(best_model, f)

        logging.info("Model saved successfully")

        return best_model

    except Exception as e:
        logging.error(f"Training error: {e}")
        raise e