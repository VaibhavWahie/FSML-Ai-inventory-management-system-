import pickle
from src.preprocess import preprocess_data
from src.features import create_features

def predict(input_df):
    # Load model
    with open("models/model_v1.pkl", "rb") as f:
        model = pickle.load(f)

    # Apply same pipeline
    df = preprocess_data(input_df)
    df = create_features(df)

    X = df.drop(columns=["sales", "date"], errors="ignore")

    preds = model.predict(X)

    return preds.tolist()