"""
7.Inference.py
Script untuk melakukan inference/prediksi menggunakan model yang sudah di-serve.
Mengirim request ke API dan menampilkan hasil prediksi.

Author: gaji
"""

import requests
import json
import time
import random
import sys

# Fix Windows console encoding for emoji/unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


API_URL = "http://localhost:8000"


def check_health():
    """Cek status API."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Uptime: {data['uptime_seconds']:.1f}s")
            return True
        else:
            print(f"❌ API returned status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {API_URL}")
        print(f"   Pastikan prometheus_exporter.py sudah berjalan!")
        return False


def single_prediction():
    """Melakukan satu prediksi."""
    print("\n" + "="*50)
    print("SINGLE PREDICTION")
    print("="*50)
    
    # Sample wine data
    sample = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    
    print(f"\nInput Features:")
    for k, v in sample.items():
        print(f"  {k}: {v}")
    
    response = requests.post(f"{API_URL}/predict", json=sample)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 Prediction Result:")
        print(f"   Class: {result['prediction']} ({result['prediction_label']})")
        print(f"   Confidence: {result['confidence']:.4f}")
    else:
        print(f"❌ Error: {response.text}")
    
    return response.json() if response.status_code == 200 else None


def batch_predictions(n_samples=10):
    """Melakukan beberapa prediksi untuk menghasilkan metriks monitoring."""
    print(f"\n{'='*50}")
    print(f"BATCH PREDICTIONS ({n_samples} samples)")
    print(f"{'='*50}")
    
    results = {"Good": 0, "Bad": 0, "errors": 0}
    latencies = []
    
    for i in range(n_samples):
        # Generate random wine features (realistic ranges)
        sample = {
            "fixed_acidity": round(random.uniform(4.6, 15.9), 2),
            "volatile_acidity": round(random.uniform(0.12, 1.58), 3),
            "citric_acid": round(random.uniform(0.0, 1.0), 2),
            "residual_sugar": round(random.uniform(0.9, 15.5), 2),
            "chlorides": round(random.uniform(0.012, 0.611), 4),
            "free_sulfur_dioxide": round(random.uniform(1.0, 72.0), 1),
            "total_sulfur_dioxide": round(random.uniform(6.0, 289.0), 1),
            "density": round(random.uniform(0.9901, 1.0037), 4),
            "pH": round(random.uniform(2.74, 4.01), 2),
            "sulphates": round(random.uniform(0.33, 2.0), 2),
            "alcohol": round(random.uniform(8.4, 14.9), 1)
        }
        
        start = time.time()
        try:
            response = requests.post(f"{API_URL}/predict", json=sample)
            latency = time.time() - start
            latencies.append(latency)
            
            if response.status_code == 200:
                result = response.json()
                results[result['prediction_label']] += 1
                status = "✅"
            else:
                results["errors"] += 1
                status = "❌"
        except Exception as e:
            results["errors"] += 1
            latency = time.time() - start
            status = "❌"
        
        print(f"  [{i+1}/{n_samples}] {status} Latency: {latency*1000:.1f}ms")
        time.sleep(0.1)  # Small delay between requests
    
    # Summary
    print(f"\n📊 Batch Results:")
    print(f"   Good Wine: {results['Good']}")
    print(f"   Bad Wine:  {results['Bad']}")
    print(f"   Errors:    {results['errors']}")
    if latencies:
        print(f"\n⏱️  Latency Statistics:")
        print(f"   Mean:   {sum(latencies)/len(latencies)*1000:.1f}ms")
        print(f"   Min:    {min(latencies)*1000:.1f}ms")
        print(f"   Max:    {max(latencies)*1000:.1f}ms")


def stress_test(duration=30, rps=5):
    """Stress test untuk generate lebih banyak metriks."""
    print(f"\n{'='*50}")
    print(f"STRESS TEST (duration={duration}s, target_rps={rps})")
    print(f"{'='*50}")
    
    end_time = time.time() + duration
    total_requests = 0
    errors = 0
    
    while time.time() < end_time:
        sample = {
            "fixed_acidity": round(random.uniform(4.6, 15.9), 2),
            "volatile_acidity": round(random.uniform(0.12, 1.58), 3),
            "citric_acid": round(random.uniform(0.0, 1.0), 2),
            "residual_sugar": round(random.uniform(0.9, 15.5), 2),
            "chlorides": round(random.uniform(0.012, 0.611), 4),
            "free_sulfur_dioxide": round(random.uniform(1.0, 72.0), 1),
            "total_sulfur_dioxide": round(random.uniform(6.0, 289.0), 1),
            "density": round(random.uniform(0.9901, 1.0037), 4),
            "pH": round(random.uniform(2.74, 4.01), 2),
            "sulphates": round(random.uniform(0.33, 2.0), 2),
            "alcohol": round(random.uniform(8.4, 14.9), 1)
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=sample, timeout=5)
            total_requests += 1
            if response.status_code != 200:
                errors += 1
        except:
            errors += 1
            total_requests += 1
        
        time.sleep(1.0 / rps)
    
    print(f"\n📊 Stress Test Results:")
    print(f"   Total Requests: {total_requests}")
    print(f"   Errors:         {errors}")
    print(f"   Success Rate:   {(total_requests-errors)/total_requests*100:.1f}%")
    print(f"   Actual RPS:     {total_requests/duration:.1f}")


def main():
    """Main inference function."""
    print("="*60)
    print("  WINE QUALITY ML MODEL - INFERENCE SCRIPT")
    print("  Author: gaji")
    print("="*60)
    
    # Check health
    if not check_health():
        print("\n⚠️  Jalankan server terlebih dahulu:")
        print("   python 3.prometheus_exporter.py")
        sys.exit(1)
    
    # Single prediction
    single_prediction()
    
    # Batch predictions
    batch_predictions(n_samples=20)
    
    # Stress test for monitoring
    stress_test(duration=30, rps=3)
    
    print(f"\n{'='*60}")
    print("✅ INFERENCE SELESAI")
    print(f"{'='*60}")
    print(f"\n📊 Cek metriks di:")
    print(f"   Prometheus: http://localhost:9090")
    print(f"   Grafana:    http://localhost:3000")
    print(f"   Raw Metrics: http://localhost:8001/metrics")


if __name__ == "__main__":
    main()
