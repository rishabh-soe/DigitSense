"""
DigitSense — Streamlit Cloud Web Application
============================================
Interactive Streamlit app for handwritten digit recognition (0-9).
Ready for one-click deployment on Streamlit Community Cloud (share.streamlit.io).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

st.set_page_config(
    page_title="DigitSense — Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide",
)

RANDOM_STATE = 42
ACCENT_ORANGE = "#d4571f"
ACCENT_TEAL = "#1f6f6b"
INK_DARK = "#1d1b18"
MUTED_GRAY = "#6b665e"
SOFT_SAND = "#f1ece3"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d8d2c7",
    "axes.labelcolor": INK_DARK,
    "text.color": INK_DARK,
    "xtick.color": MUTED_GRAY,
    "ytick.color": MUTED_GRAY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def engineer_features(X: np.ndarray) -> np.ndarray:
    """Expand 64 pixel intensities into 82 engineered features."""
    X = np.asarray(X, dtype=np.float64)
    imgs = X.reshape(-1, 8, 8)
    row_sums = imgs.sum(axis=2)
    col_sums = imgs.sum(axis=1)
    total_ink = X.sum(axis=1, keepdims=True)
    active_pixels = (X > 0).sum(axis=1, keepdims=True)
    return np.hstack([X, row_sums, col_sums, total_ink, active_pixels])


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ("classifier", classifier),
    ])


@st.cache_resource
def train_digitsense_models():
    """Load data, run 5-fold CV comparison, and evaluate on held-out test set."""
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    classifiers = {
        "RBF SVM": SVC(C=7.0, kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, weights="distance", metric="euclidean"
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted = {}

    for name, clf in classifiers.items():
        pipe = build_pipeline(clf)
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        results.append({
            "Model": name,
            "CV Macro F1": cv_scores["test_score"].mean(),
            "CV Std": cv_scores["test_score"].std(),
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Test Balanced Acc": balanced_accuracy_score(y_test, y_pred),
            "Test Macro Precision": precision_score(y_test, y_pred, average="macro"),
            "Test Macro Recall": recall_score(y_test, y_pred, average="macro"),
            "Test Macro F1": f1_score(y_test, y_pred, average="macro"),
            "Test ROC-AUC (OvR)": roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="macro"
            ),
        })

    cmp_df = pd.DataFrame(results).sort_values("CV Macro F1", ascending=False).reset_index(drop=True)
    best_name = cmp_df.loc[0, "Model"]
    return X, y, X_test, y_test, cmp_df, fitted[best_name], best_name


def preprocess_uploaded_image(uploaded_file) -> np.ndarray:
    """Crop, center, downsample to 8x8, and normalize pixel intensities to [0, 16]."""
    img = Image.open(uploaded_file).convert("L")
    arr = np.asarray(img, dtype=np.float64)
    if arr.mean() > 127:
        arr = 255.0 - arr

    ys, xs = np.where(arr > 40)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        size = int(max(x1 - x0, y1 - y0) * 1.25 + 4)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = size // 2
        padded = np.zeros((size, size), dtype=np.float64)
        src_x0, src_y0 = max(0, int(cx - half)), max(0, int(cy - half))
        src_x1, src_y1 = min(arr.shape[1], int(cx - half) + size), min(arr.shape[0], int(cy - half) + size)
        dst_x0, dst_y0 = src_x0 - int(cx - half), src_y0 - int(cy - half)
        padded[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = arr[src_y0:src_y1, src_x0:src_x1]
        arr = padded

    resized = Image.fromarray(arr.astype(np.uint8)).resize((8, 8), Image.Resampling.BILINEAR)
    px = np.asarray(resized, dtype=np.float64).flatten()
    max_px = px.max() if px.max() > 0 else 1.0
    return np.round(np.clip((px / max_px) * 16.0, 0, 16)).reshape(1, 64)


X, y, X_test, y_test, cmp_df, best_model, best_model_name = train_digitsense_models()

st.title("✍️ DigitSense — Handwritten Digit Recognition (0–9)")
st.markdown(
    "Recognizing handwritten digits from **8×8 grayscale images** using a leak-free "
    "`scikit-learn` pipeline (**82 engineered features → StandardScaler → PCA (38 dims) → RBF SVM**)."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test Accuracy", f"{cmp_df.loc[0, 'Test Accuracy']*100:.2f}%")
c2.metric("Test Macro F1", f"{cmp_df.loc[0, 'Test Macro F1']:.4f}")
c3.metric("Dataset Samples", f"{len(X):,}")
c4.metric("Features → PCA", "82 → 38 dims")

tab_demo, tab_results, tab_eda, tab_html = st.tabs([
    "🎯 Live Digit Predictor",
    "📊 Model Comparison & Confusion Matrix",
    "🔍 Dataset EDA",
    "🌐 Original Interactive HTML App",
])

with tab_demo:
    col_left, col_right = st.columns([1, 1.3])
    with col_left:
        mode = st.radio(
            "Choose input source:",
            ["Pick from Held-Out Test Set", "Upload Handwritten Digit Image"],
        )
        if mode == "Pick from Held-Out Test Set":
            idx = st.slider("Held-out test sample index", 0, len(X_test) - 1, 0)
            sample_64 = X_test[idx : idx + 1]
            true_label = int(y_test[idx])
            st.info(f"Ground-truth label: **{true_label}**")
        else:
            uploaded = st.file_uploader("Upload a PNG or JPG image of a single digit", type=["png", "jpg", "jpeg"])
            if uploaded is not None:
                sample_64 = preprocess_uploaded_image(uploaded)
                true_label = None
            else:
                sample_64 = X_test[0:1]
                true_label = int(y_test[0])
                st.caption("Showing sample #0 until you upload an image.")

    with col_right:
        probs = best_model.predict_proba(sample_64)[0]
        pred_class = int(np.argmax(probs))
        confidence = probs[pred_class] * 100.0

        st.subheader(f"Predicted Digit: `{pred_class}` (Confidence: {confidence:.1f}%)")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0), gridspec_kw={"width_ratios": [1, 2.2]})
        ax1.imshow(sample_64.reshape(8, 8), cmap="gray", vmin=0, vmax=16)
        ax1.set_title("Model Input (8×8)")
        ax1.set_xticks([])
        ax1.set_yticks([])

        colors = [ACCENT_ORANGE if i == pred_class else ACCENT_TEAL for i in range(10)]
        ax2.barh(range(10), probs * 100, color=colors)
        ax2.set_yticks(range(10))
        ax2.invert_yaxis()
        ax2.set_xlim(0, 108)
        ax2.set_xlabel("Probability (%)")
        ax2.set_title("Class Probabilities")
        for i, p in enumerate(probs):
            if p >= 0.01:
                ax2.text(p * 100 + 1.5, i, f"{p*100:.1f}%", va="center", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab_results:
    st.subheader("5-Fold Cross-Validation & Held-Out Test Comparison")
    st.dataframe(cmp_df.style.format({
        "CV Macro F1": "{:.4f}",
        "CV Std": "{:.4f}",
        "Test Accuracy": "{:.4f}",
        "Test Balanced Acc": "{:.4f}",
        "Test Macro Precision": "{:.4f}",
        "Test Macro Recall": "{:.4f}",
        "Test Macro F1": "{:.4f}",
        "Test ROC-AUC (OvR)": "{:.4f}",
    }), use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        x = np.arange(len(cmp_df))
        w = 0.26
        ax.bar(x - w, cmp_df["CV Macro F1"], w, yerr=cmp_df["CV Std"], capsize=3, label="CV Macro F1", color=ACCENT_TEAL)
        ax.bar(x, cmp_df["Test Accuracy"], w, label="Test Accuracy", color=ACCENT_ORANGE)
        ax.bar(x + w, cmp_df["Test Macro F1"], w, label="Test Macro F1", color="#4a5568")
        ax.set_xticks(x)
        ax.set_xticklabels(cmp_df["Model"], rotation=12, ha="right")
        ax.set_ylim(0.90, 1.01)
        ax.set_ylabel("Score")
        ax.set_title("Classifier Comparison")
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with r2:
        y_test_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_test_pred)
        fig, ax = plt.subplots(figsize=(6.0, 4.6))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = cm.max() / 2.0
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else INK_DARK, fontsize=9)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Predicted Digit")
        ax.set_ylabel("True Digit")
        ax.set_title(f"Confusion Matrix ({best_model_name})")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader(f"Per-Class Classification Report ({best_model_name})")
    report_dict = classification_report(y_test, best_model.predict(X_test), output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().iloc[:10]
    st.dataframe(report_df.round(4), use_container_width=True)

with tab_eda:
    imgs = X.reshape(-1, 8, 8)
    e1, e2 = st.columns(2)
    with e1:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        classes, counts = np.unique(y, return_counts=True)
        ax.bar(classes, counts, color=ACCENT_TEAL, width=0.65)
        ax.axhline(counts.mean(), color=ACCENT_ORANGE, linestyle="--", label=f"Mean ({counts.mean():.1f})")
        ax.set_xticks(classes)
        ax.set_xlabel("Digit Class")
        ax.set_ylabel("Samples")
        ax.set_title("Class Distribution (n = 1,797)")
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with e2:
        fig, axes = plt.subplots(2, 5, figsize=(6.5, 4.0))
        for digit in range(10):
            ax = axes[digit // 5, digit % 5]
            ax.imshow(imgs[y == digit].mean(axis=0), cmap="YlOrRd", vmin=0, vmax=16)
            ax.set_title(f"Digit {digit}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Average Pixel Intensity per Class", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab_html:
    html_candidates = [
        "DigitSense — Handwritten Digit Recognition.html",
        "index.html",
    ]
    found_html = next((p for p in html_candidates if os.path.exists(p)), None)
    if found_html:
        with open(found_html, "r", encoding="utf-8") as f:
            components.html(f.read(), height=1400, scrolling=True)
    else:
        st.info(
            "Place `DigitSense — Handwritten Digit Recognition.html` or `index.html` in the "
            "repository root to also embed the standalone HTML drawing canvas inside this tab."
        )
