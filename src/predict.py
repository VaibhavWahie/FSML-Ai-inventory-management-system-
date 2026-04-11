import pickle
import pandas as pd

with open("models/model_v1.pkl", "rb") as f:
    model = pickle.load(f)

def predict(data: dict):
    df = pd.DataFrame([data])

    # match training feature order
    df = df[model.feature_names_in_]

    return model.predict(df)[0]