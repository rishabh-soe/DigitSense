"""
DigitSense — Production Flask Web Server
========================================
Serves the DigitSense web application (and optionally provides a /predict REST API
backed by the trained scikit-learn RBF SVM pipeline).

Deployment structure:
  project/
  ├── app.py
  ├── requirements.txt
  └── templates/
      └── index.html   <-- Rename "DigitSense — Handwritten Digit Recognition.html" to index.html
"""

import os
from functools import lru_cache
from flask import Flask, jsonify, render_template, request, send_from_directory
import numpy as np

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

app = Flask(__name__, template_folder="templates", static_folder="static")

HTML_FILENAME = "DigitSense — Handwritten Digit Recognition.html"


def engineer_features(X: np.ndarray) -> np.ndarray:
    """Expand 64 pixel intensities into 82 engineered features."""
    X = np.asarray(X, dtype=np.float64)
    imgs = X.reshape(-1, 8, 8)
    row_sums = imgs.sum(axis=2)
    col_sums = imgs.sum(axis=1)
    total_ink = X.sum(axis=1, keepdims=True)
    active_pixels = (X > 0).sum(axis=1, keepdims=True)
    return np.hstack([X, row_sums, col_sums, total_ink, active_pixels])


@lru_cache(maxsize=1)
def get_trained_pipeline() -> Pipeline:
    """Train and cache the DigitSense RBF SVM pipeline for optional API inference."""
    digits = load_digits()
    X_train, _, y_train, _ = train_test_split(
        digits.data, digits.target, test_size=0.20, stratify=digits.target, random_state=42
    )
    pipeline = Pipeline([
        ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("classifier", SVC(C=7.0, kernel="rbf", probability=True, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


@app.route("/")
def index():
    """Serve the DigitSense web app from templates/index.html or root directory."""
    template_path = os.path.join(app.template_folder or "templates", "index.html")
    if os.path.exists(template_path):
        return render_template("index.html")
    if os.path.exists(HTML_FILENAME):
        return send_from_directory(".", HTML_FILENAME)
    if os.path.exists("index.html"):
        return send_from_directory(".", "index.html")
    return (
        "<h1>DigitSense HTML file not found</h1>"
        "<p>Place your HTML file at <code>templates/index.html</code> or "
        "<code>index.html</code> in the project root.</p>",
        404,
    )


@app.route("/health")
def health():
    """Health check endpoint for Render, Railway, Fly.io, and Docker."""
    return jsonify({"status": "ok", "service": "DigitSense"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Optional server-side inference endpoint.
    Expects JSON: {"pixels": [64 float values in 0..16]}
    Returns JSON: {"prediction": int, "confidence": float, "probabilities": [10 floats]}
    """
    payload = request.get_json(silent=True) or {}
    pixels = payload.get("pixels")
    if not isinstance(pixels, list) or len(pixels) != 64:
        return jsonify({"error": "Expected JSON body with 'pixels' list of length 64."}), 400

    sample = np.array(pixels, dtype=np.float64).reshape(1, 64)
    model = get_trained_pipeline()
    probs = model.predict_proba(sample)[0]
    pred_class = int(np.argmax(probs))

    return jsonify({
        "prediction": pred_class,
        "confidence": round(float(probs[pred_class]), 4),
        "probabilities": [round(float(p), 4) for p in probs],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
