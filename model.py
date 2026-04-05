"""
model.py
Trains Logistic Regression (baseline) and Random Forest (main) models
on the synthetic soil dataset. Exposes a predict() function used by app.py.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# ── Feature column order (must stay consistent everywhere) ──────────────────
FEATURES = ["pH", "Nitrogen", "Phosphorus", "Potassium", "Moisture", "OrganicCarbon"]
TARGET   = "FertilityClass"
MODEL_PATH  = "rf_model.joblib"
SCALER_PATH = "scaler.joblib"
ENCODER_PATH = "label_encoder.joblib"


def train_and_save(csv_path: str = "dataset.csv") -> dict:
    """
    Train Logistic Regression + Random Forest, save the RF model artifacts,
    and return a dict of evaluation metrics.
    """
    df = pd.read_csv(csv_path)
    X  = df[FEATURES].values
    y  = df[TARGET].values

    # Encode labels  Low=0  Medium=1  High=2
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # ── Scaler (required for Logistic Regression; stored for app use) ───────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Baseline: Logistic Regression ───────────────────────────────────────
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_sc))

    # ── Main model: Random Forest ────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc   = accuracy_score(y_test, rf_preds)
    report   = classification_report(
        y_test, rf_preds,
        target_names=le.classes_,
        output_dict=True
    )

    # Persist artifacts
    joblib.dump(rf,     MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le,     ENCODER_PATH)

    return {
        "lr_accuracy": round(lr_acc * 100, 2),
        "rf_accuracy": round(rf_acc * 100, 2),
        "report": report,
        "feature_importances": dict(zip(FEATURES, rf.feature_importances_))
    }


def load_artifacts():
    """Load saved model, scaler and label encoder. Train if not found."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
        train_and_save()
    rf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return rf, le


def predict(inputs: dict) -> dict:
    """
    inputs: dict with keys matching FEATURES
    Returns: { class, confidence, probabilities, feature_importances }
    """
    rf, le = load_artifacts()
    X = np.array([[inputs[f] for f in FEATURES]])

    proba  = rf.predict_proba(X)[0]           # shape: (n_classes,)
    idx    = int(np.argmax(proba))
    label  = le.inverse_transform([idx])[0]
    conf   = float(proba[idx])

    # Map class probabilities to human-readable labels
    prob_map = {le.inverse_transform([i])[0]: round(float(p) * 100, 1)
                for i, p in enumerate(proba)}

    # Feature importances from the trained RF
    feat_imp = dict(zip(FEATURES, rf.feature_importances_))
    feat_imp_pct = {k: round(v * 100, 1) for k, v in feat_imp.items()}

    return {
        "class":               label,
        "confidence":          round(conf * 100, 1),
        "probabilities":       prob_map,
        "feature_importances": feat_imp_pct
    }


# ── Run standalone to train & print metrics ─────────────────────────────────
if __name__ == "__main__":
    print("Training models …")
    metrics = train_and_save()
    print(f"\nLogistic Regression accuracy : {metrics['lr_accuracy']}%")
    print(f"Random Forest accuracy       : {metrics['rf_accuracy']}%")
    print("\nRandom Forest feature importances:")
    for feat, imp in sorted(metrics["feature_importances"].items(),
                             key=lambda x: -x[1]):
        print(f"  {feat:<18} {imp:.4f}")
    print("\nClassification Report (Random Forest):")
    for cls, vals in metrics["report"].items():
        if isinstance(vals, dict):
            print(f"  {cls:<10}  precision={vals['precision']:.2f}  "
                  f"recall={vals['recall']:.2f}  f1={vals['f1-score']:.2f}")
