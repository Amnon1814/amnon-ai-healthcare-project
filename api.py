"""FastAPI endpoint for the same UCI-trained educational pipeline."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from train_model import main as train_model, FEATURES

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "heart_disease_pipeline.joblib"

app = FastAPI(
    title="Healthcare Intelligence Research API",
    description="Educational demonstration only. Not for clinical use.",
    version="1.0.0"
)

class HeartFeatures(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)

def get_model():
    if not MODEL_PATH.exists():
        train_model()
    return joblib.load(MODEL_PATH)["model"]

@app.get("/health")
def health():
    return {"status": "ok", "purpose": "educational demonstration only"}

@app.post("/predict")
def predict(data: HeartFeatures):
    row = pd.DataFrame([data.model_dump()], columns=FEATURES)
    model = get_model()
    return {
        "dataset_class_prediction": int(model.predict(row)[0]),
        "model_probability_class_1": round(float(model.predict_proba(row)[0, 1]), 4),
        "warning": "Not a medical risk score, diagnosis, or treatment recommendation."
    }
