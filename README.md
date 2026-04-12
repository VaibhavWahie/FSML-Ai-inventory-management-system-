# FSML-Ai-inventory-management-system-

📦 AI Smart Inventory Management System
🚀 Overview
This project is an AI-based inventory forecasting system that predicts future product demand using historical sales data.

It helps:

Reduce overstocking

Avoid stockouts

Improve inventory planning

📊 Features
Data preprocessing & feature engineering

Machine learning models (Random Forest, XGBoost)

Demand prediction API using FastAPI

Interactive dashboard using Streamlit

Docker support

Basic CI/CD pipeline

⚙️ Project Structure
src/        → ML logic  
pipeline/   → training pipeline  
models/     → saved model  
app/        → FastAPI API  
data/       → dataset link  
logs/       → logs  
▶️ How to Run
1. Train Model
python pipeline/pipeline.py
2. Run API
uvicorn app.app:app --reload
Open:

http://localhost:8000/docs
3. Run Dashboard
streamlit run streamlit_app.py
🐳 Docker
docker build -t inventory-app .
docker run -p 8000:8000 inventory-app
📌 Model
Final model: XGBoost

Metric used: RMSE

📝 Notes
Same preprocessing is used in training and prediction

Model is saved as models/model_v1.pkl

Logs stored in logs/app.log