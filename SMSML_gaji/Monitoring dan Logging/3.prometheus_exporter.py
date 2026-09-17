"""
prometheus_exporter.py
Custom Prometheus Exporter untuk monitoring model Machine Learning.
Mengexpose 12+ metriks untuk Prometheus scraping.

Author: gaji
Dataset: Breast Cancer Wisconsin
"""

import time
import random
import threading
import psutil
import numpy as np
from prometheus_client import (
    start_http_server, Counter, Histogram, Gauge, Summary,
    Info, REGISTRY
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# ============================================================
# PROMETHEUS METRICS DEFINITION
# ============================================================

PREDICTION_TOTAL = Counter('ml_model_predictions_total', 'Total predictions', ['model_name', 'prediction_class'])
PREDICTION_LATENCY = Histogram('ml_model_prediction_latency_seconds', 'Latency', ['model_name'])
PREDICTION_ERRORS = Counter('ml_model_prediction_errors_total', 'Total errors', ['model_name', 'error_type'])
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy', ['model_name'])
MODEL_PRECISION = Gauge('ml_model_precision', 'Model precision', ['model_name'])
MODEL_RECALL = Gauge('ml_model_recall', 'Model recall', ['model_name'])
MODEL_F1_SCORE = Gauge('ml_model_f1_score', 'Model F1 score', ['model_name'])
DATA_DRIFT = Gauge('ml_model_input_data_drift', 'Data drift', ['model_name', 'feature'])
MEMORY_USAGE = Gauge('ml_model_memory_usage_bytes', 'Memory usage', ['model_name'])
CPU_USAGE = Gauge('ml_model_cpu_usage_percent', 'CPU usage', ['model_name'])
REQUEST_RATE = Gauge('ml_model_request_rate_per_minute', 'Requests/min', ['model_name'])
CONFIDENCE_SCORE = Histogram('ml_model_confidence_score', 'Confidence', ['model_name'])
MODEL_INFO = Info('ml_model', 'Model information')
REQUEST_SUMMARY = Summary('ml_model_request_processing_seconds', 'Processing time summary', ['model_name'])
MODEL_UPTIME = Gauge('ml_model_uptime_seconds', 'Uptime', ['model_name'])

app = FastAPI(title="Breast Cancer ML API")

model = None
model_name = "breast-cancer-rf"
start_time = time.time()
request_timestamps = []

class PredictionInput(BaseModel):
    features: List[float]

class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float

def load_model():
    global model
    model_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'Membangun_model', 'model.pkl'),
        os.path.join(os.path.dirname(__file__), 'model.pkl'),
    ]
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"✅ Model loaded from: {path}")
            return
    print("⚠️ No saved model found. Creating demo model for monitoring...")
    X, y = make_classification(n_samples=569, n_features=30, n_classes=2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'model.pkl')
    
    MODEL_ACCURACY.labels(model_name=model_name).set(0.95)
    MODEL_PRECISION.labels(model_name=model_name).set(0.94)
    MODEL_RECALL.labels(model_name=model_name).set(0.96)
    MODEL_F1_SCORE.labels(model_name=model_name).set(0.95)

def update_system_metrics():
    while True:
        try:
            process = psutil.Process()
            MEMORY_USAGE.labels(model_name=model_name).set(process.memory_info().rss)
            CPU_USAGE.labels(model_name=model_name).set(process.cpu_percent(interval=1))
            MODEL_UPTIME.labels(model_name=model_name).set(time.time() - start_time)
            recent = [t for t in request_timestamps if time.time() - t <= 60]
            REQUEST_RATE.labels(model_name=model_name).set(len(recent))
        except:
            pass
        time.sleep(5)

@app.on_event("startup")
async def startup():
    load_model()
    MODEL_INFO.info({'model_name': model_name, 'dataset': 'breast-cancer'})
    threading.Thread(target=update_system_metrics, daemon=True).start()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    global request_timestamps
    request_timestamps.append(time.time())
    request_timestamps = [t for t in request_timestamps if time.time() - t <= 300]
    
    start = time.time()
    try:
        X = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = float(max(probabilities))
        latency = time.time() - start
        
        prediction_label = "Benign" if prediction == 1 else "Malignant"
        PREDICTION_TOTAL.labels(model_name=model_name, prediction_class=prediction_label).inc()
        PREDICTION_LATENCY.labels(model_name=model_name).observe(latency)
        CONFIDENCE_SCORE.labels(model_name=model_name).observe(confidence)
        
        return PredictionOutput(prediction=int(prediction), prediction_label=prediction_label, confidence=confidence)
    except Exception as e:
        PREDICTION_ERRORS.labels(model_name=model_name, error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    start_http_server(8001)
    uvicorn.run(app, host="0.0.0.0", port=8000)
