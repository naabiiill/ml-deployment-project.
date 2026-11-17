from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schemas import HeartInput

MODEL_PATH = Path("model/heart_model.joblib")
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

app = FastAPI(title="Heart Disease Prediction API",
              description="Educational ML API for heart disease classification",
              version="1.0.0")

model = None

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/info")
def info():
    return {"model_type": "RandomForestClassifier",
            "features": FEATURES}

@app.post("/predict")
def predict(data: HeartInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    row = pd.DataFrame([data.model_dump()], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    return {"heart_disease": bool(prediction)}
