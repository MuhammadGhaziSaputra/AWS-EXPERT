"""
3.prometheus_exporter.py
Custom Prometheus Exporter untuk monitoring model Machine Learning.
Mengexpose 12+ metrics untuk Prometheus scraping & Grafana visualization.

Author: Muhammad Ghazi Saputra (muhammad_ghazi_sptr1)
Dataset: Breast Cancer Wisconsin
Model: Random Forest Classifier
"""

import time
import threading
import psutil
import numpy as np
import pandas as pd
import joblib
import os
import sys
import warnings
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prometheus_client import (
    start_http_server, Counter, Histogram, Gauge, Summary,
    Info
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# PROMETHEUS METRICS DEFINITION
# ============================================================

PREDICTION_TOTAL = Counter('ml_model_predictions_total', 'Total predictions', ['model_name', 'prediction_class'])
PREDICTION_LATENCY = Histogram('ml_model_prediction_latency_seconds', 'Prediction latency in seconds', ['model_name'])
PREDICTION_ERRORS = Counter('ml_model_prediction_errors_total', 'Total prediction errors', ['model_name', 'error_type'])

MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy score from evaluation', ['model_name'])
MODEL_PRECISION = Gauge('ml_model_precision', 'Model precision score from evaluation', ['model_name'])
MODEL_RECALL = Gauge('ml_model_recall', 'Model recall score from evaluation', ['model_name'])
MODEL_F1_SCORE = Gauge('ml_model_f1_score', 'Model F1 score from evaluation', ['model_name'])

DATA_DRIFT = Gauge('ml_model_input_data_drift', 'Absolute data drift per feature against baseline', ['model_name', 'feature'])
MEMORY_USAGE = Gauge('ml_model_memory_usage_bytes', 'Process memory usage in bytes', ['model_name'])
CPU_USAGE = Gauge('ml_model_cpu_usage_percent', 'Process CPU usage percentage', ['model_name'])
REQUEST_RATE = Gauge('ml_model_request_rate_per_minute', 'Inference request rate per minute', ['model_name'])
CONFIDENCE_SCORE = Histogram('ml_model_confidence_score', 'Prediction confidence score distribution', ['model_name'])
MODEL_INFO = Info('ml_model', 'Model metadata information')
REQUEST_SUMMARY = Summary('ml_model_request_processing_seconds', 'Request processing duration summary', ['model_name'])
MODEL_UPTIME = Gauge('ml_model_uptime_seconds', 'Model serving uptime in seconds', ['model_name'])

app = FastAPI(
    title="Breast Cancer ML API",
    description="Model Serving and Prometheus Monitoring API for Breast Cancer Detection",
    version="1.0.0"
)

model = None
feature_names = None
baseline_means = None
model_name = "breast-cancer-rf"
start_time = time.time()
request_timestamps = []

class PredictionInput(BaseModel):
    features: List[float]

class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    latency_ms: float

def load_model():
    """Memuat model Machine Learning yang telah dilatih secara riil."""
    global model, feature_names, baseline_means
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths = [
        os.path.join(script_dir, 'model.pkl'),
        os.path.join(script_dir, '..', 'Membangun_model', 'model.pkl'),
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"[OK] Model loaded from: {path}")
            break
    else:
        raise FileNotFoundError(
            "Model file 'model.pkl' tidak ditemukan. "
            "Pastikan model telah dilatih dan disimpan sebelum menjalankan exporter."
        )

    # Dapatkan feature names dari model
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = [f"feature_{i}" for i in range(30)]

    # Evaluasi model menggunakan data pengujian riil (X_test.csv & y_test.csv)
    test_data_dirs = [
        os.path.join(script_dir, 'breast_cancer_preprocessing'),
        os.path.join(script_dir, '..', 'Membangun_model', 'breast_cancer_preprocessing'),
    ]
    
    evaluated = False
    for data_dir in test_data_dirs:
        X_test_path = os.path.join(data_dir, 'X_test.csv')
        y_test_path = os.path.join(data_dir, 'y_test.csv')
        X_train_path = os.path.join(data_dir, 'X_train.csv')
        
        if os.path.exists(X_test_path) and os.path.exists(y_test_path):
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path).values.ravel()
            
            y_pred = model.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            
            MODEL_ACCURACY.labels(model_name=model_name).set(acc)
            MODEL_PRECISION.labels(model_name=model_name).set(prec)
            MODEL_RECALL.labels(model_name=model_name).set(rec)
            MODEL_F1_SCORE.labels(model_name=model_name).set(f1)
            
            print(f"[OK] Real Model Evaluation Metrics Set:")
            print(f"     Accuracy : {acc:.4f}")
            print(f"     Precision: {prec:.4f}")
            print(f"     Recall   : {rec:.4f}")
            print(f"     F1 Score : {f1:.4f}")
            
            # Hitung baseline feature means untuk deteksi data drift riil
            if os.path.exists(X_train_path):
                X_train = pd.read_csv(X_train_path)
                baseline_means = X_train.mean().to_dict()
            else:
                baseline_means = X_test.mean().to_dict()
                
            evaluated = True
            break
            
    if not evaluated:
        raise FileNotFoundError(
            "Berkas pengujian X_test.csv dan y_test.csv tidak ditemukan di direktori breast_cancer_preprocessing. "
            "Evaluasi metrik memerlukan dataset riil."
        )

def update_system_metrics():
    """Memperbarui metrik sistem (Memory, CPU, Uptime, Request Rate)."""
    while True:
        try:
            process = psutil.Process()
            MEMORY_USAGE.labels(model_name=model_name).set(process.memory_info().rss)
            CPU_USAGE.labels(model_name=model_name).set(process.cpu_percent(interval=1))
            MODEL_UPTIME.labels(model_name=model_name).set(time.time() - start_time)
            
            recent = [t for t in request_timestamps if time.time() - t <= 60]
            REQUEST_RATE.labels(model_name=model_name).set(len(recent))
        except Exception:
            pass
        time.sleep(5)

@app.on_event("startup")
async def startup_event():
    load_model()
    MODEL_INFO.info({
        'model_name': model_name,
        'model_type': 'RandomForestClassifier',
        'dataset': 'breast_cancer_wisconsin',
        'author': 'muhammad_ghazi_sptr1'
    })
    threading.Thread(target=update_system_metrics, daemon=True).start()

@app.get("/")
async def root():
    return {
        "message": "Breast Cancer ML Model Serving API",
        "model_name": model_name,
        "author": "muhammad_ghazi_sptr1",
        "status": "active"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "uptime_seconds": time.time() - start_time}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    global request_timestamps
    request_timestamps.append(time.time())
    request_timestamps = [t for t in request_timestamps if time.time() - t <= 300]
    
    start_req = time.time()
    try:
        if len(input_data.features) != 30:
            raise ValueError(f"Diharapkan 30 fitur masukan, tetapi diterima {len(input_data.features)}.")
            
        X = np.array(input_data.features).reshape(1, -1)
        
        # Inference
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        confidence = float(max(probabilities))
        latency = time.time() - start_req
        
        # Label: 0 = Malignant, 1 = Benign
        prediction_label = "Benign" if prediction == 1 else "Malignant"
        
        # Update Prometheus metrics
        PREDICTION_TOTAL.labels(model_name=model_name, prediction_class=prediction_label).inc()
        PREDICTION_LATENCY.labels(model_name=model_name).observe(latency)
        CONFIDENCE_SCORE.labels(model_name=model_name).observe(confidence)
        REQUEST_SUMMARY.labels(model_name=model_name).observe(latency)
        
        # Update feature data drift metrics jika baseline tersedia
        if baseline_means is not None and feature_names is not None:
            for idx, val in enumerate(input_data.features[:5]):  # monitor 5 key features
                feat_name = feature_names[idx]
                ref_mean = baseline_means.get(feat_name, 0.0)
                drift_val = abs(val - ref_mean)
                DATA_DRIFT.labels(model_name=model_name, feature=feat_name).set(drift_val)
        
        return PredictionOutput(
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=round(confidence, 4),
            latency_ms=round(latency * 1000, 2)
        )
    except Exception as e:
        PREDICTION_ERRORS.labels(model_name=model_name, error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start Prometheus metrics HTTP server pada port 8001
    start_http_server(8001)
    print("Prometheus metrics server started at http://localhost:8001/metrics")
    
    # Start FastAPI server pada port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
