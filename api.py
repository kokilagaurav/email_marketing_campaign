from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Load model and metrics
model = joblib.load('email_engagement_model.joblib')
metrics = joblib.load('email_engagement_metrics.joblib')

class PredictionInput(BaseModel):
    hour: int
    weekday: str
    user_country: str
    email_text: str
    email_version: str

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        input_df = pd.DataFrame([{
            'hour': data.hour,
            'weekday': data.weekday,
            'user_country': data.user_country,
            'email_text': data.email_text,
            'email_version': data.email_version
        }])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(max(probabilities))
        
        # Map status
        status_map = {
            0: "Not Opened",
            1: "Opened but Not Clicked",
            2: "Clicked and Opened"
        }
        
        return {
            "status": "success",
            "engagement_status": status_map[prediction],
            "confidence": confidence,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
