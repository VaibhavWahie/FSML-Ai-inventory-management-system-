from sklearn.metrics import mean_squared_error
import logging

def evaluate_model(model, X_test, y_test):
    try:
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)

        logging.info(f"Evaluation RMSE: {rmse}")
        print(f"RMSE: {rmse}")

        return rmse

    except Exception as e:
        logging.error(f"Evaluation error: {e}")
        raise e