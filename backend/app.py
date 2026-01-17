import os
import re
import json
import time
import math
import joblib
import logging
import threading
import traceback
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from dotenv import load_dotenv
from scipy import sparse
from urllib.parse import urlparse

# Optional Redis — used only if REDIS_URL provided
try:
    import redis
except Exception:
    redis = None

# ---------------------------
# Load environment
# ---------------------------
load_dotenv()

# Paths & config
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models"
MODEL_DIR = Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
MODEL_PATH = MODEL_DIR / "url_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
FRONTEND_DIR = (BASE_DIR.parent / "frontend") if (BASE_DIR.parent / "frontend").exists() else (BASE_DIR / "frontend")

VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
REDIS_URL = os.getenv("REDIS_URL", "")

DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", 0.6))
DEFAULT_GRAY_LOW = float(os.getenv("GRAY_LOW", 0.45))
DEFAULT_GRAY_HIGH = float(os.getenv("GRAY_HIGH", 0.55))

VT_TTL_HOURS = float(os.getenv("VT_TTL_HOURS", 6))
VT_MIN_SECONDS_BETWEEN_SUBMISSIONS = int(os.getenv("VT_MIN_SECONDS", 3))

LOG_PATH = os.getenv("LOG_PATH", str(BASE_DIR / "logs" / "app.log"))
os.makedirs(Path(LOG_PATH).parent, exist_ok=True)

# ---------------------------
# Logging: console + rotating file
# ---------------------------
logger = logging.getLogger("securescan.backend")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)

fh = RotatingFileHandler(LOG_PATH, maxBytes=10_000_000, backupCount=5)
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/")
CORS(app)

# ---------------------------
# Globals and locks
# ---------------------------
_model_lock = threading.RLock()
_vt_lock = threading.RLock()
_cache_lock = threading.RLock()

# In-memory caches (fallback, use Redis if REDIS_URL is set)
VT_CACHE = {}           # key -> (timestamp, data)
VT_LAST_SUBMIT = {}     # key -> datetime of last submit

# Metrics
METRICS = {
    "requests_total": 0,
    "heuristic_rejects": 0,
    "ml_malicious": 0,
    "vt_verified_malicious": 0,
    "errors": 0
}
_METRICS_LOCK = threading.Lock()

# Redis client (optional)
_redis_client = None
if REDIS_URL and redis is not None:
    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=False)
        logger.info("Redis caching enabled (REDIS_URL provided)")
    except Exception as e:
        logger.warning("Failed to create Redis client: %s", e)
        _redis_client = None

# ---------------------------
# Utility helpers
# ---------------------------
def incr_metric(name, n=1):
    with _METRICS_LOCK:
        METRICS[name] = METRICS.get(name, 0) + n

def now():
    return datetime.utcnow()

def to_iso(dt):
    if not dt:
        return None
    return dt.isoformat() + "Z"

# ---------------------------
# Model & vectorizer loading
# ---------------------------
url_model = None
vectorizer = None
model_metadata = None

def load_artifacts():
    global url_model, vectorizer, model_metadata
    with _model_lock:
        logger.info("Loading artifacts from %s", MODEL_DIR)
        try:
            url_model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
        except Exception:
            logger.exception("Failed to load model; leaving url_model=None")
            url_model = None
        try:
            vectorizer = joblib.load(VECTORIZER_PATH) if VECTORIZER_PATH.exists() else None
        except Exception:
            logger.exception("Failed to load vectorizer; leaving vectorizer=None")
            vectorizer = None
        model_metadata = {}
        try:
            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r", encoding="utf-8") as fh:
                    model_metadata = json.load(fh)
        except Exception:
            logger.exception("Failed to load metadata; using defaults")
            model_metadata = {}

        # adopt thresholds from metadata when available
        global DEFAULT_THRESHOLD, DEFAULT_GRAY_LOW, DEFAULT_GRAY_HIGH
        DEFAULT_THRESHOLD = float(model_metadata.get("threshold", DEFAULT_THRESHOLD))
        DEFAULT_GRAY_LOW = float(model_metadata.get("gray_zone", {}).get("low", DEFAULT_GRAY_LOW))
        DEFAULT_GRAY_HIGH = float(model_metadata.get("gray_zone", {}).get("high", DEFAULT_GRAY_HIGH))

        logger.info("Artifacts loaded. model=%s vectorizer=%s thresholds=(%s,%s,%s)",
                    bool(url_model), bool(vectorizer),
                    DEFAULT_THRESHOLD, DEFAULT_GRAY_LOW, DEFAULT_GRAY_HIGH)

# initial load
load_artifacts()

# ---------------------------
# Heuristic & preprocessing
# ---------------------------
SUSPICIOUS_TLDS = set([
    'zip','review','country','kim','work','xyz','top','tk','ml','ga','cf','gq',
    'cn','ru','biz','info','click','link','site','online','shop','loan','win'
])
SHORTENERS = set(['bit.ly','goo.gl','t.co','tinyurl','ow.ly','is.gd','buff.ly','cutt.ly','rebrand.ly','adf.ly'])
KNOWN_BRANDS = set(['paypal','amazon','apple','microsoft','google','facebook','instagram','netflix','bankofamerica','icici','hdfc','sbi','axisbank'])
PHISH_KEYWORDS = set(['verify','update','secure','account','confirm','login','signin','bank','free','prize','reward','gift','claim','win','money','bonus'])

def ensure_parseable(url: str) -> str:
    #Return URL with scheme ensured (add http:// if missing).
    if not url:
        return url
    if "://" not in url:
        return "http://" + url
    return url

def preprocess_url(url: str) -> str:
    
  #Normalizing input to match training (strip scheme and www, lower-case).
  
    if not url:
        return ""
    u = url.lower().strip()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.rstrip("/")
    u = re.sub(r"[^a-z0-9./:_-]", "", u)
    return u

def local_url_heuristic(raw_input: str):
    if not raw_input or not isinstance(raw_input, str):
        return {"ok": False, "reason": "empty_url", "parsed": None}

    parseable = ensure_parseable(raw_input)
    try:
        u = urlparse(parseable)
    except Exception:
        return {"ok": False, "reason": "invalid_url_parse", "parsed": parseable}

    hostname = (u.hostname or "").lower()
    path = (u.path or "").lower()
    query = (u.query or "").lower()
    full = hostname + path + query

    # Basic quick rejects
    if not hostname:
        return {"ok": False, "reason": "no_hostname", "parsed": parseable}
    if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', hostname):
        return {"ok": False, "reason": "ip_in_hostname", "parsed": parseable}
    if len(hostname) > 80:
        return {"ok": False, "reason": "long_hostname", "parsed": parseable}
    tld = hostname.split('.')[-1]
    if tld in SUSPICIOUS_TLDS:
        return {"ok": False, "reason": "suspicious_tld", "parsed": parseable}
    if any(short in hostname for short in SHORTENERS):
        return {"ok": False, "reason": "url_shortener", "parsed": parseable}
    if any(k in full for k in PHISH_KEYWORDS):
        return {"ok": False, "reason": "suspicious_keyword", "parsed": parseable}
    if '@' in raw_input:
        return {"ok": False, "reason": "contains_at_symbol", "parsed": parseable}
    if len(re.findall(r'%[0-9a-fA-F]{2}', raw_input)) > 3:
        return {"ok": False, "reason": "encoded_chars", "parsed": parseable}
    if re.search(r'\.(php|asp|exe|js|sh|scr|phtml)(?:$|[?/])', path):
        return {"ok": False, "reason": "suspicious_extension", "parsed": parseable}

    scheme = (u.scheme or "").lower()
    if scheme not in ("http", "https", ""):
        return {"ok": False, "reason": f"unsupported_scheme_{scheme}", "parsed": parseable}
    if scheme == "http":
        return {"ok": True, "reason": "insecure_http", "parsed": parseable}

    # brand impersonation — mark suspicious but allow to continue
    for brand in KNOWN_BRANDS:
        if brand in hostname and not hostname.endswith(f"{brand}.com"):
            return {"ok": True, "reason": f"suspected_impersonation_{brand}", "parsed": parseable}

    return {"ok": True, "reason": "looks_ok", "parsed": parseable}

# ---------------------------
# VirusTotal integration (thread-safe + cache + throttle)
# ---------------------------
VT_CACHE_TTL = timedelta(hours=VT_TTL_HOURS)
def _cache_get(key: str):
    # Try Redis first if configured
    if _redis_client:
        try:
            val = _redis_client.get(key)
            if val:
                return json.loads(val)
        except Exception:
            logger = getattr(logger, "exception")
            logger("Redis get failed for %s", key)
    # Fallback in-memory
    with _vt_lock:
        entry = VT_CACHE.get(key)
        if not entry:
            return None
        ts, data = entry
        if datetime.utcnow() - ts < VT_CACHE_TTL:
            return data
        # expired
        VT_CACHE.pop(key, None)
        return None

def _cache_set(key: str, data):
    if _redis_client:
        try:
            _redis_client.set(key, json.dumps(data), ex=int(VT_CACHE_TTL.total_seconds()))
        except Exception:
            logger.exception("Redis set failed for %s", key)
    with _vt_lock:
        VT_CACHE[key] = (datetime.utcnow(), data)

def _can_submit(url_key: str):
    if _redis_client:
        try:
            last = _redis_client.get(f"_last_submit_{url_key}")
            if not last:
                return True
            last_ts = float(last.decode() if isinstance(last, bytes) else last)
            return (time.time() - last_ts) >= VT_MIN_SECONDS_BETWEEN_SUBMISSIONS
        except Exception:
            logger.exception("Redis VT last submit check failed")
            return True
    with _vt_lock:
        last = VT_LAST_SUBMIT.get(url_key)
        if not last:
            return True
        return (datetime.utcnow() - last).total_seconds() >= VT_MIN_SECONDS_BETWEEN_SUBMISSIONS

def _set_last_submit(url_key: str):
    if _redis_client:
        try:
            _redis_client.set(f"_last_submit_{url_key}", str(time.time()), ex=int(VT_CACHE_TTL.total_seconds()))
        except Exception:
            logger.exception("Redis set last submit failed")
    with _vt_lock:
        VT_LAST_SUBMIT[url_key] = datetime.utcnow()

def virustotal_url_report(url: str):
    """
    Submit a URL to VirusTotal and attempt to get analysis.
    Returns parsed JSON or {"error": "..."}.
    Uses _cache_get/_cache_set and throttles frequent submits per-URL.
    """
    if not VIRUSTOTAL_API_KEY:
        return {"error": "no_api_key_provided"}
    key = f"vt_url:{url}"
    cached = _cache_get(key)
    if cached:
        return cached

    if not _can_submit(url):
        return {"error": "vt_rate_limited"}

    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    try:
        submit = requests.post("https://www.virustotal.com/api/v3/urls", data={"url": url}, headers=headers, timeout=15)
        if submit.status_code not in (200, 201):
            logger.warning("VT submit returned %s for %s: %s", submit.status_code, url, submit.text[:300])
            return {"error": f"vt_submit_status_{submit.status_code}", "raw": submit.text}
        submit_json = submit.json()
        url_id = submit_json.get("data", {}).get("id")
        if not url_id:
            logger.warning("VT submit returned no id for %s: %s", url, submit_json)
            return {"error": "no_vt_id", "raw": submit_json}

        _set_last_submit(url)

        # Poll analysis endpoint a few times (small backoff)
        analysis_url = f"https://www.virustotal.com/api/v3/analyses/{url_id}"
        for attempt in range(5):
            if attempt > 0:
                time.sleep(min(2.0 * attempt, 6.0))
            r = requests.get(analysis_url, headers=headers, timeout=15)
            if r.status_code == 200:
                j = r.json()
                _cache_set(key, j)
                return j
            else:
                logger.debug("VT analysis attempt %s returned %s for %s", attempt, r.status_code, url)
        return {"error": "vt_analysis_unavailable", "raw": submit_json}
    except Exception as e:
        logger.exception("VirusTotal URL error for %s: %s", url, e)
        return {"error": str(e)}

def virustotal_file_report(file_hash: str):
    #Lookup file hash on VirusTotal using files endpoint. Uses cache.
    if not VIRUSTOTAL_API_KEY:
        return {"error": "no_api_key_provided"}
    key = f"vt_file:{file_hash}"
    cached = _cache_get(key)
    if cached:
        return cached
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    try:
        r = requests.get(f"https://www.virustotal.com/api/v3/files/{file_hash}", headers=headers, timeout=15)
        if r.status_code == 200:
            j = r.json()
            _cache_set(key, j)
            return j
        else:
            logger.warning("VT file lookup returned %s for %s", r.status_code, file_hash)
            return {"error": f"vt_file_status_{r.status_code}", "raw": r.text}
    except Exception as e:
        logger.exception("VirusTotal file error: %s", e)
        return {"error": str(e)}

# ---------------------------
# Featurization & explainability
# ---------------------------

def featurize_for_model(raw_url: str):
    if vectorizer:
        try:
            return vectorizer.transform([raw_url])
        except Exception:
            logger.exception("Vectorizer transform failed for %s", raw_url)
    # fallback numeric vector consistent with training fallback
    length = len(raw_url)
    num_digits = sum(c.isdigit() for c in raw_url)
    num_special = len(re.findall(r"[^\w]", raw_url))
    has_https = 1 if raw_url.startswith("https") else 0
    return np.array([[length, num_digits, num_special, has_https]], dtype=float)

def extract_top_features(X, top_k=8):
    try:
        if url_model is None or vectorizer is None:
            return None
        if not hasattr(url_model, "coef_"):
            return None
        coef = url_model.coef_[0]
        # gather non-zero indices in X
        indices = []
        vals = []
        if sparse.issparse(X):
            row = X.tocsr()
            start, end = row.indptr[0], row.indptr[1]
            indices = row.indices[start:end].tolist()
            vals = row.data[start:end].tolist()
        else:
            arr = np.asarray(X)
            nz = np.nonzero(arr[0])[0]
            indices = nz.tolist()
            vals = [float(arr[0, i]) for i in nz]
        feat_names = None
        try:
            feat_names = vectorizer.get_feature_names_out()
        except Exception:
            feat_names = None
        contribs = []
        for idx, v in zip(indices, vals):
            w = float(coef[idx]) if idx < len(coef) else 0.0
            contrib = float(v * w)
            name = feat_names[idx] if feat_names is not None and idx < len(feat_names) else str(idx)
            contribs.append((contrib, name, float(v)))
        contribs.sort(key=lambda x: x[0], reverse=True)
        top_pos = [{'ngram': n, 'contrib': c, 'tfidf': t} for c, n, t in contribs[:top_k]]
        top_neg = [{'ngram': n, 'contrib': c, 'tfidf': t} for c, n, t in contribs[-top_k:]][::-1]
        return {'positive': top_pos, 'negative': top_neg}
    except Exception:
        logger.exception("Failed to extract top features")
        return None

# ---------------------------
# Routes: health/info/admin
# ---------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": bool(url_model and vectorizer), "time": to_iso(now())})

@app.route("/api/model_info")
def api_model_info():
    with _model_lock:
        return jsonify({
            "model_loaded": bool(url_model and vectorizer),
            "metadata": model_metadata or {},
            "thresholds": {
                "threshold": DEFAULT_THRESHOLD,
                "gray_low": DEFAULT_GRAY_LOW,
                "gray_high": DEFAULT_GRAY_HIGH
            },
            "vt_cache_ttl_hours": VT_TTL_HOURS
        })

@app.route("/metrics")
def metrics():
    with _METRICS_LOCK:
        return jsonify(METRICS)

@app.route("/admin/reload_model", methods=["POST"])
def admin_reload_model():
    auth = request.headers.get("Authorization", "")
    if not ADMIN_TOKEN:
        return jsonify({"error": "admin_token_not_configured"}), 403
    if auth != f"Bearer {ADMIN_TOKEN}":
        return jsonify({"error": "unauthorized"}), 401
    try:
        load_artifacts()
        return jsonify({"status": "reloaded", "model_loaded": bool(url_model and vectorizer)})
    except Exception as e:
        logger.exception("Reload failed: %s", e)
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Main detection endpoint
# ---------------------------

# Seed local bad-hash db (extend as needed)
known_bad_hashes = {
    "44d88612fea8a8f36de82e1278abb02f": "EICAR-Test-String"
}

@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            incr_metric("errors", 1)
            return jsonify({"error": "invalid_json"}), 400
        incr_metric("requests_total", 1)

        # ---- URL scanning ----
        if "url" in payload:
            raw = payload.get("url", "")
            if not isinstance(raw, str) or raw.strip() == "":
                return jsonify({"error": "empty_url"}), 400

            # 1) Heuristic layer (fast)
            h = local_url_heuristic(raw)
            if not h.get("ok", False):
                incr_metric("heuristic_rejects", 1)
                logger.info("[HEURISTIC_REJECT] %s -> %s", raw, h.get("reason"))
                return jsonify({
                    "type": "url",
                    "input": raw,
                    "verdict": "malicious",
                    "layer": "heuristic",
                    "reason": h.get("reason")
                }), 200

            insecure_flag = (h.get("reason") == "insecure_http")

            # 2) ML layer
            if url_model is None:
                # fallback: no model loaded
                reason = "heuristic_ok"
                if insecure_flag:
                    reason = "insecure_http"
                return jsonify({
                    "type": "url",
                    "input": raw,
                    "verdict": "unknown",
                    "layer": "fallback",
                    "reason": reason
                }), 200

            processed = preprocess_url(h.get("parsed") or raw)  # ensure preprocessing matches training
            X = featurize_for_model(processed)
            proba = None
            try:
                if hasattr(url_model, "predict_proba"):
                    proba = float(url_model.predict_proba(X)[0][1])
                else:
                    # Model does not expose predict_proba — use predict as fallback
                    p = int(url_model.predict(X)[0])
                    proba = float(p)
            except Exception:
                logger.exception("Model prediction failed for %s", processed)
                incr_metric("errors", 1)
                proba = None

            # Decision with thresholds & gray-zone
            verdict = "unknown"
            vt_info = None
            vt_malicious = None
            vt_total = None

            if proba is None:
                verdict = "unknown"
            elif proba >= DEFAULT_THRESHOLD:
                verdict = "malicious"
                incr_metric("ml_malicious", 1)
            elif proba <= DEFAULT_GRAY_LOW:
                verdict = "safe"
            elif DEFAULT_GRAY_LOW < proba < DEFAULT_GRAY_HIGH:
                # Gray zone -> consult VirusTotal
                logger.info("[GRAY] %s prob=%0.4f -> consult VT", raw, proba)
                vt_info = virustotal_url_report(h.get("parsed") or raw)
                if isinstance(vt_info, dict) and "error" not in vt_info:
                    # try to read last_analysis_stats or attributes.stats
                    stats = {}
                    try:
                        stats = (vt_info.get("data", {}) or {}).get("attributes", {}) or {}
                        analysis_stats = stats.get("last_analysis_stats") or stats.get("stats") or {}
                    except Exception:
                        analysis_stats = {}
                    if isinstance(analysis_stats, dict):
                        vt_malicious = int(analysis_stats.get("malicious", 0) or 0)
                        vt_total = sum(int(v) for v in analysis_stats.values() if isinstance(v, int))
                        if vt_malicious > 0:
                            verdict = "malicious (verified via VirusTotal)"
                            incr_metric("vt_verified_malicious", 1)
                        else:
                            verdict = "safe (verified via VirusTotal)"
                    else:
                        verdict = "suspicious (vt_inconclusive)"
                else:
                    verdict = "suspicious (vt_unavailable)"
            else:
                verdict = "suspicious"

            reason = "vectorizer+classifier"
            if insecure_flag:
                reason = "insecure_http; " + reason

            # Explainability (top features) best-effort
            top_features = extract_top_features(X, top_k=8)

            resp = {
                "type": "url",
                "input": raw,
                "layer": "ml_model" + ("+vt" if vt_info else ""),
                "verdict": verdict,
                "probability": round(proba, 6) if proba is not None else None,
                "thresholds": {"threshold": DEFAULT_THRESHOLD, "gray_low": DEFAULT_GRAY_LOW, "gray_high": DEFAULT_GRAY_HIGH},
                "reason": reason,
                "trained_at": model_metadata.get("trained_at") if model_metadata else None
            }
            if vt_malicious is not None:
                resp["vt_malicious"] = int(vt_malicious)
            if vt_total is not None:
                resp["vt_total"] = int(vt_total)
            if top_features is not None:
                resp["top_features"] = top_features

            logger.info("[RESULT] %s -> %s (p=%s) [%s]", raw, verdict, resp.get("probability"), resp.get("layer"))
            return jsonify(resp), 200

        # ---- File-hash scan ----
        if "fileHash" in payload:
            file_hash = (payload.get("fileHash") or "").strip().lower()
            file_name = payload.get("fileName", "unknown")
            if not file_hash:
                return jsonify({"error": "empty_fileHash"}), 400
            # local DB
            info = known_bad_hashes.get(file_hash)
            if info:
                return jsonify({
                    "type": "file", "input": file_name, "hash": file_hash,
                    "verdict": "malicious", "source": "local_hash_db", "reason": info
                }), 200
            # VirusTotal lookup
            vtfile = virustotal_file_report(file_hash)
            if isinstance(vtfile, dict) and "error" not in vtfile:
                stats = (vtfile.get("data", {}) or {}).get("attributes", {}) or {}
                last_stats = stats.get("last_analysis_stats") or {}
                malicious_count = int(last_stats.get("malicious", 0) or 0) if isinstance(last_stats, dict) else 0
                if malicious_count > 0:
                    return jsonify({
                        "type": "file", "input": file_name, "hash": file_hash,
                        "verdict": "malicious", "source": "virustotal", "vt_malicious": malicious_count
                    }), 200
                else:
                    return jsonify({
                        "type": "file", "input": file_name, "hash": file_hash,
                        "verdict": "safe", "source": "virustotal", "vt_malicious": 0
                    }), 200
            return jsonify({
                "type": "file", "input": file_name, "hash": file_hash,
                "verdict": "unknown", "source": "local", "reason": "hash not found"
            }), 200

        return jsonify({"error": "no valid input (send {\"url\":...} or {\"fileName\":..., \"fileHash\":...})"}), 400

    except Exception as e:
        logger.exception("Unhandled detect error: %s", e)
        incr_metric("errors", 1)
        return jsonify({"error": "internal_error"}), 500

# ---------------------------
# Serve frontend root 
# ---------------------------
@app.route("/")
def index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return send_from_directory(str(FRONTEND_DIR), "index.html")
    return jsonify({"status": "ok", "message": "SecureScan API running (frontend not found)"}), 200


if __name__ == "__main__":
    debug_flag = os.getenv("FLASK_DEBUG", "False").lower() in ("1", "true", "yes")
    # Dev server only. For production run gunicorn: `gunicorn backend.app:app -w 4`
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=debug_flag)
