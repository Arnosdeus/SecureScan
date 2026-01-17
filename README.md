# NLP Phissing detection

"""
Features:
 - structured logging (console + rotating file)
 - health/metrics/admin endpoints
 - thread-safe VirusTotal caching + throttling
 - model/vectorizer loading with metadata-driven thresholds
 - heuristic + ML + VirusTotal layered decision logic
 - file-hash handling and explainability (top n-grams)
 - optional Redis caching (set REDIS_URL)
 - ready to run under gunicorn (recommended) or python app.py for dev

Environment variables:
  VIRUSTOTAL_API_KEY  - your VirusTotal API key (optional but recommended)
  ADMIN_TOKEN         - bearer token for admin endpoints (recommended)
  MODEL_DIR           - path to models directory (default ./models)
  VT_TTL_HOURS        - VT cache TTL in hours (default 6)
  VT_MIN_SECONDS      - VT per-URL throttle seconds (default 3)
  LOG_PATH            - rotating log file path (default ./logs/app.log)
  REDIS_URL           - optional redis url for shared cache (e.g. redis://localhost:6379/0)
  FLASK_DEBUG         - "True" to run dev server in debug mode
"""
Backend training and serving for a URL/file phishing detection demo.

Backend
- `backend/app.py` - Flask server with /api/detect.
- `backend/train_model.py` - Training script (TF-IDF char n-grams + LogisticRegression).
- `backend/test_model.py` - Small helper to load the saved model and run example predictions.

How to train

1. Install dependencies (preferably inside a virtualenv):

```powershell
pip install -r backend/requirements.txt
```

2. Run a quick dry-run to validate CSV detection:

```powershell
cd backend
python train_model.py --dry-run
```

3. Quick sample train (fast):

```powershell
python train_model.py --sample-size 5000
```

4. Full training (may take several minutes):

```powershell
python train_model.py
```

Test saved model

```powershell
python backend/test_model.py
```

Server

```powershell
python backend/app.py
```
