import requests
import time
import random
import sys

API_URL = "http://localhost:8000"

def check_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def generate_random_features():
    # 30 random features for breast cancer
    return [random.uniform(0.1, 2.0) for _ in range(30)]

def batch_predictions(n_samples=20):
    print(f"\nBATCH PREDICTIONS ({n_samples} samples)")
    results = {"Benign": 0, "Malignant": 0, "errors": 0}
    for i in range(n_samples):
        sample = {"features": generate_random_features()}
        try:
            response = requests.post(f"{API_URL}/predict", json=sample)
            if response.status_code == 200:
                results[response.json()['prediction_label']] += 1
            else:
                results["errors"] += 1
        except:
            results["errors"] += 1
        time.sleep(0.1)
    print(results)

def main():
    if not check_health():
        print("API not running")
        sys.exit(1)
    batch_predictions(n_samples=20)
    print("Inference completed")

if __name__ == "__main__":
    main()
