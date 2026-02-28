"""
FastAPI Inference API
Serves the flight delay prediction model
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.engineer import encode_single_record, get_airline_list, get_airport_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="✈️ Flight Delay Prediction API",
    description="""
    ## End-to-End ML System for Flight Delay Prediction
    
    Predicts whether a flight will be delayed more than 15 minutes.
    
    ### Models Trained:
    - Logistic Regression
    - Random Forest  
    - XGBoost
    - LightGBM
    
    ### Data Source: BTS On-Time Performance Dataset
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ───────────────────────────────────────────────────────────

MODEL_PATH = "models/pipeline.pkl"
ENCODER_PATH = "models/encoders.pkl"
METRICS_PATH = "models/metrics.json"
LOG_PATH = "data/inference_logs.csv"

model = None
encoders = None
model_metrics = {}
inference_count = 0


def load_artifacts():
    """Load model and encoders on startup."""
    global model, encoders, model_metrics
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"⚠️  No model found at {MODEL_PATH}. Run training first.")
    
    if os.path.exists(ENCODER_PATH):
        encoders = joblib.load(ENCODER_PATH)
        logger.info(f"✅ Encoders loaded from {ENCODER_PATH}")
    
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            model_metrics = json.load(f)


@app.on_event("startup")
async def startup_event():
    load_artifacts()
    Path("data").mkdir(exist_ok=True)
    logger.info("🚀 Flight Delay API started")


# ── Request/Response Schemas ───────────────────────────────────────────────

class FlightInput(BaseModel):
    airline: str = Field(..., example="AA", description="IATA airline code")
    origin: str = Field(..., example="JFK", description="Origin airport code")
    destination: str = Field(..., example="LAX", description="Destination airport code")
    departure_delay: float = Field(0.0, example=15.0,
                                    description="Departure delay in minutes (0 if unknown)")
    distance: float = Field(..., example=2475.0, description="Flight distance in miles")
    day_of_week: int = Field(..., ge=1, le=7, example=3,
                              description="Day of week (1=Mon, 7=Sun)")
    month: int = Field(..., ge=1, le=12, example=6, description="Month (1-12)")
    hour: int = Field(12, ge=0, le=23, example=14,
                       description="Departure hour (0-23)")

    @validator('airline')
    def airline_upper(cls, v):
        return v.upper().strip()

    @validator('origin', 'destination')
    def airport_upper(cls, v):
        return v.upper().strip()


class PredictionResponse(BaseModel):
    prediction: str
    delay_probability: float
    confidence: str
    risk_level: str
    model_name: str
    inference_id: str
    timestamp: str


class BatchFlightInput(BaseModel):
    flights: List[FlightInput]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    total_predictions: int
    uptime: str


class ModelInfoResponse(BaseModel):
    model_name: str
    metrics: dict
    features: list
    airlines: list
    airports: list


# ── Utility Functions ──────────────────────────────────────────────────────

def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.5:
        return "MODERATE"
    elif prob < 0.7:
        return "HIGH"
    else:
        return "VERY HIGH"


def get_confidence(prob: float) -> str:
    margin = abs(prob - 0.5)
    if margin > 0.35:
        return "HIGH"
    elif margin > 0.2:
        return "MEDIUM"
    else:
        return "LOW"


def log_inference(flight_input: dict, prediction: dict):
    """Async log inference to CSV."""
    try:
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "airline": flight_input.get("airline"),
            "origin": flight_input.get("origin"),
            "destination": flight_input.get("destination"),
            "distance": flight_input.get("distance"),
            "prediction": prediction.get("prediction"),
            "probability": prediction.get("delay_probability")
        }
        df = pd.DataFrame([record])
        df.to_csv(LOG_PATH, mode='a',
                  header=not os.path.exists(LOG_PATH), index=False)
    except Exception as e:
        logger.warning(f"Failed to log inference: {e}")


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "✈️ Flight Delay Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "model_info": "/model/info"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_name": model_metrics.get("model_name", "unknown"),
        "total_predictions": inference_count,
        "uptime": "running"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(flight: FlightInput, background_tasks: BackgroundTasks):
    """
    Predict whether a flight will be delayed more than 15 minutes.
    
    Returns probability, prediction, risk level, and confidence.
    """
    global inference_count
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run the training pipeline first."
        )
    
    try:
        # Build record dict
        record = {
            "OP_CARRIER": flight.airline,
            "ORIGIN": flight.origin,
            "DEST": flight.destination,
            "DEP_DELAY": flight.departure_delay,
            "DISTANCE": flight.distance,
            "DAY_OF_WEEK": flight.day_of_week,
            "MONTH": flight.month,
            "HOUR": flight.hour,
            "DEP_TIME": flight.hour * 100
        }
        
        # Encode features
        X = encode_single_record(record, encoders)
        
        # Predict
        prob = float(model.predict_proba(X)[0][1])
        prediction = "Delayed" if prob >= 0.5 else "On Time"
        
        inference_count += 1
        inference_id = f"INF-{inference_count:06d}"
        
        response = {
            "prediction": prediction,
            "delay_probability": round(prob, 4),
            "confidence": get_confidence(prob),
            "risk_level": get_risk_level(prob),
            "model_name": model_metrics.get("model_name", "Unknown"),
            "inference_id": inference_id,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        # Log async
        background_tasks.add_task(log_inference, record, response)
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(batch: BatchFlightInput):
    """Predict delays for multiple flights at once (max 100)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(batch.flights) > 100:
        raise HTTPException(status_code=400, detail="Max 100 flights per batch")
    
    results = []
    for flight in batch.flights:
        record = {
            "OP_CARRIER": flight.airline,
            "ORIGIN": flight.origin,
            "DEST": flight.destination,
            "DEP_DELAY": flight.departure_delay,
            "DISTANCE": flight.distance,
            "DAY_OF_WEEK": flight.day_of_week,
            "MONTH": flight.month,
            "HOUR": flight.hour,
            "DEP_TIME": flight.hour * 100
        }
        X = encode_single_record(record, encoders)
        prob = float(model.predict_proba(X)[0][1])
        results.append({
            "airline": flight.airline,
            "route": f"{flight.origin}→{flight.destination}",
            "prediction": "Delayed" if prob >= 0.5 else "On Time",
            "delay_probability": round(prob, 4),
            "risk_level": get_risk_level(prob)
        })
    
    return {"predictions": results, "count": len(results)}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model information, metrics, and supported inputs."""
    return {
        "model_name": model_metrics.get("model_name", "Unknown"),
        "metrics": model_metrics.get("metrics", {}),
        "features": [
            "DEP_DELAY", "DISTANCE", "HOUR", "DAY_OF_WEEK",
            "MONTH", "DIST_BUCKET", "IS_WEEKEND", "IS_PEAK_HOUR",
            "SEASON", "AIRLINE_ENC", "ORIGIN_ENC", "DEST_ENC"
        ],
        "airlines": get_airline_list(),
        "airports": get_airport_list()
    }


@app.get("/model/metrics", tags=["Model"])
async def get_metrics():
    """Return training metrics for the deployed model."""
    if not model_metrics:
        raise HTTPException(status_code=404, detail="No metrics found")
    return model_metrics


@app.get("/inference/logs", tags=["Monitoring"])
async def get_inference_logs(limit: int = 100):
    """Return recent inference logs."""
    if not os.path.exists(LOG_PATH):
        return {"logs": [], "total": 0}
    
    df = pd.read_csv(LOG_PATH).tail(limit)
    return {
        "logs": df.to_dict(orient='records'),
        "total": len(df),
        "delay_rate": float(
            (df['prediction'] == 'Delayed').mean()
        ) if len(df) > 0 else 0
    }


if __name__ == "__main__":
    import uvicorn
    load_artifacts()
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
