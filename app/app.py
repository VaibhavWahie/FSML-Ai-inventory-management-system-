from fastapi import FastAPI
from app.schema import InputData
from src.predict import predict
import logging

app = FastAPI()

# ---------------- LOGGING ----------------
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO
)

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"message": "API running 🚀"}

@app.post("/predict")
def get_prediction(data: InputData):
    try:
        result = predict(data.dict())

        logging.info(f"Prediction requested: {data.dict()}")

        return {"prediction": float(result)}

    except Exception as e:
        logging.error(str(e))
        return {"error": str(e)}