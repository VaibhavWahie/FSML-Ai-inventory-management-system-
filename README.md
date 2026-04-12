# FSML-Ai-inventory-management-system-

📦 AI Smart Inventory Management System
🚀 Overview
This project is an AI-powered inventory demand forecasting system that predicts future product demand using historical retail sales data.

The system helps businesses:

Reduce overstocking costs

Avoid stockouts

Improve inventory planning and decision-making

It is designed as a complete end-to-end machine learning pipeline with deployment and MLOps components.

🎯 Problem Statement
Demand forecasting is a critical challenge in retail.

Inaccurate predictions can lead to:

Overstocking → increased storage costs

Understocking → lost revenue

Inefficient supply chain operations

This project aims to build a machine learning-based solution to accurately predict future demand.

📊 Dataset
The dataset consists of time-series retail data, where each row represents daily sales of a product in a store.

Features:
Store ID

Product Family

Date

Sales

Promotion

📌 Dataset is not included due to size
👉 Link available in: data/dataset_link.txt

⚙️ Project Structure
project/
│
├── README.md
├── requirements.txt
├── Dockerfile
│
├── data/
│   ├── README.md
│   └── dataset_link.txt
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── pipeline/
│   └── pipeline.py
│
├── models/
│   └── model_v1.pkl
│
├── app/
│   ├── app.py
│   └── schema.py
│
├── logs/
│   └── app.log
│
└── notebooks/
🔄 Machine Learning Pipeline
1. Data Preprocessing (preprocess.py)
Convert date column to datetime

Sort data chronologically

Handle missing values:

sales → 0

onpromotion → 0

Encode categorical variables

✅ Same preprocessing is used during training and inference

2. Feature Engineering (features.py)
Time-Based Features
Day of week

Month

Week number

Weekend indicator

Lag Features
lag_1, lag_2, lag_3

lag_7, lag_14, lag_21, lag_28

➡ Capture past sales patterns

Rolling Features
Rolling mean (7, 14 days)

Rolling standard deviation

➡ Capture trends and volatility

Additional Feature
lag difference (lag_1 - lag_7)

3. Model Training (train.py)
Models used:

Random Forest Regressor

XGBoost Regressor

Train-test split: 80/20

Best model selected using RMSE

✅ Final model: XGBoost

4. Model Evaluation (evaluate.py)
Metric: RMSE (Root Mean Squared Error)

XGBoost achieved better performance compared to Random Forest

5. Inference Pipeline
Input → Preprocess → Feature Engineering → Prediction
✅ Same logic reused → ensures consistency
✅ Prevents data leakage

🚀 Deployment (FastAPI)
Run API
uvicorn app.app:app --reload
Endpoint
POST /predict
Sample Input
{
  "id": 1,
  "store_nbr": 1,
  "family": 1,
  "onpromotion": 0,
  "day_of_week": 2,
  "month": 3,
  "week": 10,
  "is_weekend": 0,
  "lag_1": 100,
  "lag_2": 110,
  "lag_3": 120,
  "lag_7": 130,
  "lag_14": 140,
  "lag_21": 150,
  "lag_28": 160,
  "rolling_mean_7": 125,
  "rolling_mean_14": 135,
  "rolling_std_7": 10
}
Output
{
  "prediction": 132.45
}
📊 Dashboard (Streamlit)
Run
streamlit run streamlit_app.py
Features
Store & product selection

Auto-generated lag features

Demand prediction

Visualization of trends

🐳 Docker
Build
docker build -t inventory-app .
Run
docker run -p 8000:8000 inventory-app
Access:

http://localhost:8000/docs
🔁 CI/CD (GitHub Actions)
Pipeline runs on every push

Installs dependencies

Ensures project builds successfully

Deployment is supported via Docker containers.

📝 Logging
Logs stored in:

logs/app.log
Captures:

Prediction requests

Errors

🧠 Key Highlights
End-to-end ML pipeline

Strong feature engineering

No data leakage

Model comparison approach

Deployment-ready system

MLOps integration

🔮 Future Improvements
Add external features (weather, holidays)

Use deep learning models (LSTM, GRU)

Real-time data integration

👥 Team Members

Priyamvada (Data Engineering)

Vanessa (ML Engineering)

Vaibhav (MLOps)

Aditya (Backend & Deployment)

🎯 Conclusion
This project demonstrates a scalable, production-ready machine learning system for inventory demand forecasting, combining ML, deployment, and MLOps practices to solve real-world problems.
