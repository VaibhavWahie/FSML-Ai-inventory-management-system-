from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    id: int
    store_nbr: int
    family: int
    onpromotion: int
    day_of_week: int
    month: int
    week: int
    is_weekend: int
    lag_1: float
    lag_2: float
    lag_3: float
    lag_7: float
    lag_14: float
    lag_21: float
    lag_28: float
    rolling_mean_7: float
    rolling_mean_14: float
    rolling_std_7: float

# Load model
with open("models/model_v1.pkl", "rb") as f:
    model = pickle.load(f)

print("MODEL FEATURES:", model.feature_names_in_)

@app.get("/")
def home():
    return {"message": "API running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])

        # 🔥 CRITICAL FIX: match exact feature order
        df = df[model.feature_names_in_]

        prediction = model.predict(df)[0]
        return {"prediction": float(prediction)}

    except Exception as e:
        return {"error": str(e)}