"""
1.bukti_serving - Model Serving Script
Menjalankan model sebagai API menggunakan FastAPI + Prometheus metrics.

Author: gaji

Cara menjalankan:
    python "Monitoring dan Logging/1.bukti_serving.py"
    
API Endpoints:
    - http://localhost:8000/        -> Info
    - http://localhost:8000/health   -> Health check
    - http://localhost:8000/predict  -> Prediksi
    - http://localhost:8000/docs     -> Swagger UI
    - http://localhost:8001/metrics  -> Prometheus metrics

Catatan:
    Script ini menjalankan serving model yang sama dengan prometheus_exporter.py.
    Bukti serving berupa screenshot API yang berjalan pada browser.
"""

import subprocess
import sys
import os


if __name__ == "__main__":
    print("=" * 60)
    print("  BREAST CANCER MODEL SERVING")
    print("  Author: gaji")
    print("=" * 60)
    print()
    print("Menjalankan model serving...")
    print("API akan tersedia di: http://localhost:8000")
    print("Prometheus metrics di: http://localhost:8001/metrics")
    print("Swagger docs di: http://localhost:8000/docs")
    print()

    # Jalankan prometheus_exporter.py dari direktori yang sama
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exporter_path = os.path.join(script_dir, "3.prometheus_exporter.py")

    if not os.path.exists(exporter_path):
        print(f"❌ File tidak ditemukan: {exporter_path}")
        sys.exit(1)

    try:
        subprocess.run([sys.executable, exporter_path], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server dihentikan.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error menjalankan server: {e}")
        sys.exit(1)
