# 📊 Panduan Monitoring & Logging - Wine Quality ML Model

**Author:** gaji  
**Proyek:** Membangun Sistem Machine Learning (Dicoding)

---

## 📋 Prasyarat

1. **Docker Desktop** terinstall dan berjalan
2. **Python 3.12+** (untuk menjalankan inference script)
3. **pip packages:** `requests` (`pip install requests`)

---

## 🚀 Langkah 1: Jalankan Stack Monitoring

Buka terminal di folder `Monitoring dan Logging`, lalu jalankan:

```bash
docker-compose up --build -d
```

Tunggu hingga semua container berjalan (~1-2 menit untuk build pertama kali).

Cek status:
```bash
docker-compose ps
```

**Pastikan 3 container berjalan:**
- `wine-quality-model` (API port 8000, Metrics port 8001)
- `prometheus` (port 9090)
- `grafana` (port 3000)

---

## 🔗 Langkah 2: Verifikasi Semua Service

| Service | URL | Keterangan |
|---------|-----|------------|
| Model API | http://localhost:8000 | Info endpoint |
| Swagger Docs | http://localhost:8000/docs | API documentation |
| Health Check | http://localhost:8000/health | Status API |
| Prometheus Metrics | http://localhost:8001/metrics | Raw metrics |
| Prometheus UI | http://localhost:9090 | Prometheus dashboard |
| Grafana | http://localhost:3000 | Login: admin/admin |

---

## 📡 Langkah 3: Generate Metrics (Inference)

Jalankan script inference untuk menghasilkan data metrics:

```bash
python 7.Inference.py
```

Script ini akan:
- Melakukan health check
- Mengirim 1 prediksi tunggal
- Mengirim 20 prediksi batch
- Menjalankan stress test selama 30 detik

---

## 📸 Langkah 4: Ambil Screenshot Bukti

### 4a. Bukti Serving (folder: sudah ada bukti di `1.bukti_serving.py`)
1. Buka http://localhost:8000/docs di browser
2. **Screenshot** halaman Swagger UI
3. Klik endpoint `/predict` → Try it out → Execute
4. **Screenshot** hasil prediksi
5. Simpan ke: *sudah tercakup dalam bukti serving*

### 4b. Bukti Monitoring Prometheus
1. Buka http://localhost:9090
2. Di search bar, ketik: `ml_model_predictions_total`
3. Klik **Execute** → Tab **Graph**
4. **Screenshot 1:** Graph prediction total
5. Ketik: `ml_model_prediction_latency_seconds_bucket` → Execute → Graph
6. **Screenshot 2:** Latency distribution
7. Ketik: `ml_model_accuracy` → Execute → Graph
8. **Screenshot 3:** Model accuracy
9. Buka http://localhost:9090/targets
10. **Screenshot 4:** Targets page (pastikan status UP)
11. **Simpan semua screenshot ke folder:** `4.bukti monitoring Prometheus/`

### 4c. Bukti Monitoring Grafana
1. Buka http://localhost:3000
2. Login: **admin** / **admin** (skip ganti password)
3. Navigasi ke: **Dashboards** → **ML Monitoring** → **Wine Quality ML Model - Monitoring**
4. **Screenshot 1:** Dashboard overview (tampilan penuh)
5. Scroll ke bawah
6. **Screenshot 2:** Panel latency & confidence
7. Scroll ke bawah lagi
8. **Screenshot 3:** Panel data drift & request processing
9. **Simpan semua screenshot ke folder:** `5.bukti monitoring Grafana/`

### 4d. Bukti Alerting Grafana
1. Di Grafana, navigasi ke: **Alerting** → **Alert rules**
2. **Screenshot 1:** Daftar alert rules (4 alert rules)
3. Klik salah satu alert (misalnya "High Prediction Error Rate")
4. **Screenshot 2:** Detail alert rule
5. Navigasi ke: **Alerting** → **Contact points**
6. **Screenshot 3:** Contact points page
7. **Simpan semua screenshot ke folder:** `6.bukti alerting Grafana/`

---

## 🛑 Langkah 5: Hentikan Stack

```bash
docker-compose down
```

Untuk menghapus volumes juga:
```bash
docker-compose down -v
```

---

## ❓ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Port sudah dipakai | Ubah port di `docker-compose.yml` |
| Container gagal start | Cek log: `docker-compose logs model-serving` |
| Prometheus target DOWN | Pastikan `model-serving` container sudah sehat |
| Grafana tidak ada dashboard | Cek: `docker-compose logs grafana` |
| Metrics kosong di Grafana | Jalankan `7.Inference.py` terlebih dahulu |

---

## 📁 Struktur Folder Bukti

```
Monitoring dan Logging/
├── 1.bukti_serving.py          # Script serving
├── 2.prometheus.yml            # Prometheus config
├── 3.prometheus_exporter.py    # FastAPI + Prometheus exporter
├── 4.bukti monitoring Prometheus/
│   ├── prometheus_predictions_graph.png
│   ├── prometheus_latency_graph.png
│   ├── prometheus_accuracy_graph.png
│   └── prometheus_targets_up.png
├── 5.bukti monitoring Grafana/
│   ├── grafana_dashboard_overview.png
│   ├── grafana_latency_confidence.png
│   └── grafana_drift_processing.png
├── 6.bukti alerting Grafana/
│   ├── grafana_alert_rules_list.png
│   ├── grafana_alert_rule_detail.png
│   └── grafana_contact_points.png
├── 7.Inference.py              # Inference script
├── Dockerfile                  # Docker image for serving
├── docker-compose.yml          # Full monitoring stack
├── requirements_serving.txt    # Dependencies for Docker
└── grafana/
    ├── dashboards/
    │   └── wine-quality-monitoring.json
    └── provisioning/
        ├── alerting/
        │   └── alerts.yml
        ├── dashboards/
        │   └── dashboard.yml
        └── datasources/
            └── datasource.yml
```
