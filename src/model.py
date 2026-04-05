"""
model.py
SoilHealth Predictor — Model Training, Evaluation & Prediction
SRM Institute of Science and Technology | Data Science Mini Project 2026

Models trained:
    1. Random Forest Classifier  (primary model)
    2. Decision Tree Classifier
    3. K-Nearest Neighbours (k=5)
    4. Gaussian Naive Bayes

Outputs:
    outputs/results/classification_report.txt
    outputs/results/cv_scores.txt
    outputs/results/model_comparison.csv
    outputs/graphs/confusion_matrix_rf.png
    outputs/graphs/model_comparison_bar.png
    outputs/graphs/feature_importance_rf.png
    outputs/graphs/cv_scores_lineplot.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH    = os.path.join(BASE_DIR, "dataset", "raw_data",  "soil_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
GRAPHS_DIR  = os.path.join(BASE_DIR, "outputs", "graphs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
FEATURES     = ["N", "P", "K", "pH", "Moisture", "Temperature", "EC"]
CLASS_ORDER  = ["High Fertility", "Medium Fertility", "Low Fertility", "Poor Fertility"]
SHORT_LABELS = ["High", "Medium", "Low", "Poor"]
PALETTE      = {
    "High Fertility":   "#3B6D11",
    "Medium Fertility": "#97C459",
    "Low Fertility":    "#EF9F27",
    "Poor Fertility":   "#E24B4A",
}

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 150


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREP
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare():
    df = pd.read_csv(RAW_PATH)

    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y = le.transform(df["Fertility_Class"])
    X = df[FEATURES].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, le, scaler, df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10,
            min_samples_split=5, random_state=RANDOM_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=5,
            random_state=RANDOM_SEED
        ),
        "KNN (k=5)":     KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "Naive Bayes":   GaussianNB(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(models, X_train, X_test, y_train, y_test, le):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    print("\n── Model Training & Evaluation ─────────────────")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

        results[name] = {
            "model":     model,
            "y_pred":    y_pred,
            "accuracy":  acc,
            "f1":        f1,
            "cv_scores": cv_scores,
            "cv_mean":   cv_scores.mean(),
            "cv_std":    cv_scores.std(),
        }

        print(f"\n  [{name}]")
        print(f"    Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"    Weighted F1   : {f1:.4f}")
        print(f"    CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_reports(results, X_test, y_test, le):
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    cv_path     = os.path.join(RESULTS_DIR, "cv_scores.txt")
    comp_path   = os.path.join(RESULTS_DIR, "model_comparison.csv")

    with open(report_path, "w") as rf, open(cv_path, "w") as cf:
        rf.write("=== SoilHealth Predictor — Classification Reports ===\n\n")
        cf.write("=== SoilHealth Predictor — Cross-Validation Scores ===\n\n")

        for name, res in results.items():
            # Classification report
            rf.write(f"── {name} ──────────────────────────────\n")
            rf.write(f"Test Accuracy : {res['accuracy']*100:.2f}%\n")
            rf.write(f"Weighted F1   : {res['f1']:.4f}\n\n")
            rf.write(classification_report(
                y_test, res["y_pred"],
                target_names=le.classes_
            ))
            rf.write("\n")

            # CV scores
            cf.write(f"── {name} ──────────────────────────────\n")
            for fold, s in enumerate(res["cv_scores"], 1):
                cf.write(f"  Fold {fold}: {s:.4f}\n")
            cf.write(f"  Mean : {res['cv_mean']:.4f}\n")
            cf.write(f"  Std  : {res['cv_std']:.4f}\n\n")

    print(f"\n  Reports saved → {report_path}")
    print(f"  CV scores  saved → {cv_path}")

    # Model comparison CSV
    comp = pd.DataFrame([
        {
            "Model":       name,
            "Accuracy (%)": round(res["accuracy"]*100, 2),
            "F1 Score":     round(res["f1"], 4),
            "CV Mean":      round(res["cv_mean"], 4),
            "CV Std":       round(res["cv_std"], 4),
        }
        for name, res in results.items()
    ]).sort_values("Accuracy (%)", ascending=False)
    comp.to_csv(comp_path, index=False)
    print(f"  Comparison CSV saved → {comp_path}")
    return comp


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(results, y_test, le):
    rf_preds = results["Random Forest"]["y_pred"]
    cm = confusion_matrix(y_test, rf_preds)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGn",
        xticklabels=SHORT_LABELS,
        yticklabels=SHORT_LABELS,
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title("Confusion Matrix — Random Forest", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(GRAPHS_DIR, "confusion_matrix_rf.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_model_comparison(comp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#3B6D11", "#639922", "#97C459", "#C0DD97"]

    bars = ax.barh(
        comp["Model"], comp["Accuracy (%)"],
        color=colors[:len(comp)], edgecolor="none", height=0.5
    )
    for bar, val in zip(bars, comp["Accuracy (%)"]):
        ax.text(bar.get_width() - 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", ha="right", fontsize=10,
                color="white", fontweight="bold")
    ax.set_xlim(70, 100)
    ax.set_xlabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Model Comparison — Test Accuracy", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linewidth=0.4)
    fig.tight_layout()
    path = os.path.join(GRAPHS_DIR, "model_comparison_bar.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_feature_importance(results, df):
    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    idx  = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.YlGn(np.linspace(0.4, 0.9, len(FEATURES)))[::-1]
    ax.barh(
        [FEATURES[i] for i in idx],
        [importances[i] for i in idx],
        color=colors, edgecolor="none", height=0.55
    )
    ax.set_xlabel("Feature Importance Score", fontsize=11)
    ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.4)
    fig.tight_layout()
    path = os.path.join(GRAPHS_DIR, "feature_importance_rf.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    # Print
    print("\n  Feature Importances (Random Forest):")
    for i in idx:
        print(f"    {FEATURES[i]:<18} {importances[i]*100:.2f}%")


def plot_cv_scores(results):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_map = {
        "Random Forest": "#3B6D11",
        "Decision Tree": "#EF9F27",
        "KNN (k=5)":     "#1D9E75",
        "Naive Bayes":   "#E24B4A",
    }
    folds = [f"Fold {i}" for i in range(1, 6)]

    for name, res in results.items():
        ax.plot(folds, res["cv_scores"] * 100,
                marker="o", label=name,
                color=colors_map[name], linewidth=1.8, markersize=5)

    ax.set_ylim(70, 100)
    ax.set_ylabel("CV Accuracy (%)", fontsize=11)
    ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linewidth=0.4)
    fig.tight_layout()
    path = os.path.join(GRAPHS_DIR, "cv_scores_lineplot.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

RECOMMENDATIONS = {
    "High Fertility":   "Excellent soil condition. Maintain organic matter levels and practice crop rotation.",
    "Medium Fertility": "Adequate soil. Consider minor nitrogen top-dressing for optimal crop yield.",
    "Low Fertility":    "Nutrient deficient. Apply balanced NPK fertilizer; check and correct pH to 6.0-6.5.",
    "Poor Fertility":   "Severely degraded. Requires lime application, organic compost addition, and a rest period.",
}


def predict_fertility(rf_model, scaler, le, sample: dict) -> dict:
    """
    Predict fertility class for a single soil sample.

    Parameters:
        sample: dict with keys N, P, K, pH, Moisture, Temperature, EC

    Returns:
        dict with predicted_class, confidence, probabilities, recommendation
    """
    X = np.array([[sample[f] for f in FEATURES]])
    X_s = scaler.transform(X)

    pred_label = rf_model.predict(X_s)[0]
    proba      = rf_model.predict_proba(X_s)[0]
    pred_class = le.inverse_transform([pred_label])[0]

    return {
        "predicted_class": pred_class,
        "confidence":      round(proba[pred_label] * 100, 2),
        "probabilities":   {cls: round(p * 100, 2) for cls, p in zip(le.classes_, proba)},
        "recommendation":  RECOMMENDATIONS[pred_class],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SoilHealth Predictor — Model Training & Evaluation")
    print("=" * 55)

    if not os.path.exists(RAW_PATH):
        print("  Raw dataset not found. Run src/preprocessing.py first.")
        return

    # Load data
    X_train, X_test, y_train, y_test, le, scaler, df = load_and_prepare()
    print(f"\n  Loaded: {len(df)} samples | Train: {len(X_train)} | Test: {len(X_test)}")

    # Train
    models  = get_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test, le)

    # Reports
    print("\n── Saving Reports ──────────────────────────────")
    comp = save_reports(results, X_test, y_test, le)

    # Detailed RF report
    print(f"\n── Random Forest — Full Classification Report ──")
    print(classification_report(
        y_test, results["Random Forest"]["y_pred"],
        target_names=le.classes_
    ))

    # Plots
    print("── Generating Model Plots ──────────────────────")
    plot_confusion_matrix(results, y_test, le)
    plot_model_comparison(comp)
    plot_feature_importance(results, df)
    plot_cv_scores(results)

    # Demo prediction
    print("\n── Demo Prediction ─────────────────────────────")
    sample = {"N": 92, "P": 46, "K": 178, "pH": 6.5, "Moisture": 38, "Temperature": 26, "EC": 1.3}
    rf_model = results["Random Forest"]["model"]
    result   = predict_fertility(rf_model, scaler, le, sample)
    print(f"  Input: {sample}")
    print(f"  Predicted class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']}%")
    print(f"  Probabilities   : {result['probabilities']}")
    print(f"  Recommendation  : {result['recommendation']}")

    print("\n✓ Model pipeline complete.\n")


if __name__ == "__main__":
    main()
