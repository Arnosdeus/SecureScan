import argparse
import time
import sys
import threading
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, confusion_matrix, classification_report
)
import joblib
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Utility Functions ---
def find_default_csv():
    candidates = [
        Path(__file__).parent / "training_dataset.csv",
        Path(__file__).parent / "venv" / "training_dataset.csv",
        Path(__file__).parent.parent / "training_dataset.csv",
        Path(__file__).parent.parent / "data" / "training_dataset.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def read_csv_robust(path):
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded CSV with encoding {enc}")
            return df
        except Exception:
            pass
    return pd.read_csv(path)


class TrainingTimer:
    def __init__(self):
        self._start = None
        self._running = False
    def start(self):
        self._start = time.time()
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()
    def stop(self):
        self._running = False
    def _run(self):
        while self._running:
            elapsed = int(time.time() - self._start)
            print(f"Training... {elapsed} seconds elapsed.")
            time.sleep(30)


def map_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        vals = set(series.dropna().unique())
        if vals <= {0, 1}:
            return series.fillna(0).astype(int)
        return (series.fillna(0) != 0).astype(int)
    s = series.astype(str).str.lower().str.strip()
    mapping = {
        'benign': 0, 'legitimate': 0, 'normal': 0, 'safe': 0,
        'phishing': 1, 'defacement': 1, 'malware': 1, 'malicious': 1
    }
    mapped = s.map(mapping)
    mapped = mapped.fillna(0)
    return mapped.astype(int)


def preprocess_url(url: str) -> str:
    url = str(url).lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    url = url.rstrip("/")
    url = re.sub(r"[^a-z0-9./:_-]", "", url)
    return url


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Train enhanced URL classifier with threshold and gray zone logic")
    parser.add_argument("--data", help="Path to CSV dataset (optional)")
    parser.add_argument("--sample-size", type=int, default=0)
    args = parser.parse_args()

    data_path = args.data or find_default_csv()
    if not data_path:
        print("❌ Could not find dataset CSV", file=sys.stderr)
        sys.exit(2)
    print(f"Using data: {data_path}")

    df = read_csv_robust(data_path)
    print(f"CSV columns: {df.columns.tolist()}")

    cols_lower = {c.lower(): c for c in df.columns}
    url_col = next((cols_lower[c] for c in ["url", "link", "uri"] if c in cols_lower), None)
    label_col = next((cols_lower[c] for c in ["type", "label", "class", "category"] if c in cols_lower), None)

    if not url_col or not label_col:
        print("❌ Could not detect URL or label column.")
        sys.exit(2)
    print(f"Using URL column '{url_col}' and label column '{label_col}'")

    df["label_int"] = map_labels(df[label_col])
    df = df.dropna(subset=["label_int"])
    df["label_int"] = df["label_int"].astype(int)
    print(f"Total rows after cleaning: {len(df)}")
    print(df["label_int"].value_counts())

    # Optional sampling
    if args.sample_size > 0 and len(df) > args.sample_size:
        df = df.sample(args.sample_size, random_state=42)
        print(f"Sampled {len(df)} rows")

    print("\n🔧 Preprocessing URLs...")
    df[url_col] = df[url_col].astype(str).apply(preprocess_url)

    # --- Augmentation ---
    safe_df = df[df["label_int"] == 0]
    mal_df = df[df["label_int"] == 1]
    if len(safe_df) > 100:
        extra_safe = safe_df.sample(frac=0.3, random_state=42)[url_col].apply(lambda u: u + "/index.html")
        extra_mal = mal_df.sample(frac=0.2, random_state=42)[url_col].apply(lambda u: u + "/verify/login")
        df_aug = pd.concat([
            pd.DataFrame({url_col: extra_safe, "label_int": 0}),
            pd.DataFrame({url_col: extra_mal, "label_int": 1})
        ], ignore_index=True)
        df = pd.concat([df, df_aug], ignore_index=True)
        print(f"Augmented {len(df_aug)} samples (total: {len(df)})")

    X_raw = df[url_col].values
    y = df["label_int"].values

    print("\n⚙️ Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 7),
        max_features=50000,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print("\n🚀 Training Logistic Regression...")
    timer = TrainingTimer()
    timer.start()

    model = LogisticRegression(
        max_iter=2500,
        class_weight="balanced",
        solver="saga",
        C=2.0,
        penalty="l2",
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)

    timer.stop()
    print("✅ Training complete.\n")

    # ---------------- Threshold and Evaluation ---------------- #
    print("📊 Evaluating model with custom threshold...")

    y_proba = model.predict_proba(X_test)[:, 1]

    # 🔧 Custom threshold + optional gray zone
    THRESHOLD = 0.6
    LOW_ZONE, HIGH_ZONE = 0.45, 0.55  # for potential VirusTotal escalation

    y_pred = (y_proba >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.close()

    print("💾 Saving model artifacts...")
    joblib.dump(model, os.path.join(MODEL_DIR, "url_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "threshold": THRESHOLD,
        "gray_zone": {"low": LOW_ZONE, "high": HIGH_ZONE},
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1},
        "vectorizer": {"ngram_range": [3, 7], "max_features": 50000},
        "model": {"C": 2.0, "solver": "saga", "penalty": "l2"}
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ---------------- Sanity Check ---------------- #
    print("\n🔍 Sanity check (with threshold & gray zone):")

    def classify_with_gray_zone(prob):
        if prob < LOW_ZONE:
            return "benign"
        elif prob > HIGH_ZONE:
            return "malicious"
        else:
            return "suspicious (verify with VirusTotal)"

    samples = [
        "https://google.com",
        "http://malicious-site.work/login",
        "paypal.secure-update.com"
    ]
    Xs = vectorizer.transform([preprocess_url(s) for s in samples])
    probs = model.predict_proba(Xs)[:, 1]
    for s, p in zip(samples, probs):
        label = classify_with_gray_zone(p)
        print(f"  {s:<40} -> probability: {p:.4f} | classified as: {label}")

    print("\n🎉 Training and evaluation complete.\n")


if __name__ == "__main__":
    main()

