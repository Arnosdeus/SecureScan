import joblib, os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
model_path = os.path.join(MODEL_DIR, "url_model.pkl")
vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

if not os.path.exists(model_path) or not os.path.exists(vec_path):
    print("Model or vectorizer not found in", MODEL_DIR)
    raise SystemExit(1)

model = joblib.load(model_path)
vectorizer = joblib.load(vec_path)

samples = [
    "http://example.com/login",
    "http://secure-bank.example.verify-login.com/",
    "https://github.com/",
    "http://free-money.example.co/login.php?user=abc",
]
X = vectorizer.transform(samples)
probs = None
try:
    probs = model.predict_proba(X)[:,1]
except Exception:
    probs = None
preds = model.predict(X)

for s, p, pr in zip(samples, preds, (probs if probs is not None else [None]*len(samples))):
    print(f"{s} -> pred={int(p)} score={float(pr) if pr is not None else 'n/a'}")
