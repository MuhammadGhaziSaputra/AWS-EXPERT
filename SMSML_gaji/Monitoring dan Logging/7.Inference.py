"""
7.Inference.py
Mengirimkan request inferensi riil ke model serving API menggunakan dataset pengujian Breast Cancer.
Menghasilkan metrik prediksi (Counter, Latency, Confidence, Request Rate, dll) untuk monitoring Prometheus.

Author: Muhammad Ghazi Saputra (muhammad_ghazi_sptr1)
"""

import requests
import time
import os
import sys
import pandas as pd
import numpy as np

API_URL = "http://localhost:8000"

def check_health():
    """Memeriksa kesehatan endpoint API."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def load_test_samples():
    """Memuat data uji riil untuk inferensi."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'breast_cancer_preprocessing', 'X_test.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df.values.tolist()
    else:
        raise FileNotFoundError(
            f"Berkas dataset pengujian '{data_path}' tidak ditemukan. "
            "Inferensi memerlukan data pengujian riil."
        )

def run_inference(samples, n_samples=40):
    """Menjalankan batch request inferensi."""
    print(f"\n🚀 Memulai inferensi {n_samples} sampel ke {API_URL}/predict...")
    results = {"Benign": 0, "Malignant": 0, "errors": 0}
    
    total_samples = len(samples)
    for i in range(n_samples):
        idx = i % total_samples
        features = samples[idx]
        
        payload = {"features": features}
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                label = data.get("prediction_label", "Unknown")
                confidence = data.get("confidence", 0.0)
                latency = data.get("latency_ms", 0.0)
                results[label] = results.get(label, 0) + 1
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"[{i+1}/{n_samples}] Prediksi: {label} (confidence: {confidence:.4f}, latency: {latency:.2f}ms)")
            else:
                results["errors"] += 1
                print(f"[{i+1}/{n_samples}] Error HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            results["errors"] += 1
            print(f"[{i+1}/{n_samples}] Exception: {e}")
            
        time.sleep(0.05)
        
    print(f"\n✅ Selesai! Ringkasan Hasil Inferensi: {results}")

def main():
    print("=" * 60)
    print("  BREAST CANCER MODEL INFERENCE SCRIPT")
    print("  Author: Muhammad Ghazi Saputra (muhammad_ghazi_sptr1)")
    print("=" * 60)
    
    if not check_health():
        print(f"❌ API tidak dapat dijangkau di {API_URL}. Pastikan model serving telah berjalan.")
        sys.exit(1)
        
    print(f"✅ Koneksi ke {API_URL} berhasil!")
    samples = load_test_samples()
    print(f"Memuat {len(samples)} sampel data pengujian riil.")
    
    run_inference(samples, n_samples=40)

if __name__ == "__main__":
    main()
