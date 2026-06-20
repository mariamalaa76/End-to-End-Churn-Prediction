import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

app = FastAPI(
    title="Telecom Churn Prediction Service",
    description="Production-ready API to predict customer churn in real-time.",
    version="1.0"
)

PIPELINE_PATH = "data/processed/full_pipeline.joblib"
MODEL_PATH = "models/xgboost_model.joblib"

if os.path.exists(PIPELINE_PATH) and os.path.exists(MODEL_PATH):
    pipeline = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    print(" Pipeline and Model loaded successfully into memory!")
else:
    print(" Warning: Model or Pipeline files missing! Please run the pipeline first.")

class CustomerData(BaseModel):
    signup_date: str
    age: int
    gender: str
    annual_income: float
    education: str
    marital_status: str
    dependents: int
    tenure: int
    contract: str
    payment_method: str
    paperless_billing: str
    senior_citizen: int
    monthlycharges: float
    totalcharges: float
    num_services: int
    has_phone_service: int
    has_internet_service: int
    has_online_security: int
    has_online_backup: int
    has_device_protection: int
    has_tech_support: int
    has_streaming_tv: int
    has_streaming_movies: int
    customer_satisfaction: int
    num_complaints: int
    num_service_calls: int
    late_payments: int
    avg_monthly_gb: float
    days_since_last_interaction: int
    credit_score: float

@app.get("/")
def home():
    return {"message": "Welcome to the Churn Prediction API. Go to /docs for the Swagger UI."}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        input_data = pd.DataFrame([customer.model_dump()])
        processed_data = pipeline.transform(input_data)
        
        prediction = int(model.predict(processed_data)[0])
        probability = float(model.predict_proba(processed_data)[0][1])

        status = "Churn (Will Leave)" if prediction == 1 else "No Churn (Will Stay)"
        
        return {
            "prediction": prediction,
            "churn_status": status,
            "churn_probability": round(probability, 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")