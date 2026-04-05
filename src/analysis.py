"""
analysis.py
SoilHealth Predictor — Exploratory Data Analysis & Visualisation
SRM Institute of Science and Technology | Data Science Mini Project 2026

Generates and saves the following plots to outputs/graphs/:
    1. class_distribution.png        — Pie chart of target class counts
    2. feature_distributions.png     — Histograms for all 7 features
    3. boxplots_by_class.png          — Box plots per feature per class
    4. correlation_heatmap.png        — Pearson correlation heatmap
    5. pairplot_npk_ph.png            — Scatter matrix (N, P, K, pH)
    6. scatter_N_vs_P.png             — Scatter N vs P coloured by class
    7. violin_ph_by_class.png         — Violin plot of pH per class
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH    = os.path.join(BASE_DIR, "dataset", "raw_data",  "soil_data.csv")
GRAPHS_DIR  = os.path.join(BASE_DIR, "outputs", "graphs")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

os.makedirs(GRAPHS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ["N", "P", "K", "pH", "Moisture", "Temperature", "EC"]
CLASS_ORDER   = ["High Fertility", "Medium Fertility", "Low Fertility", "Poor Fertility"]
PALETTE = {
    "High Fertility":   "#3B6D11",
    "Medium Fertility": "#97C459",
    "Low Fertility":    "#EF9F27",
    "Poor Fertility":   "#E24B4A",
}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams["figure.dpi"] = 150


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def save(fig, name: str):
    path = os.path.join(GRAPHS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame):
    counts = df["Fertility_Class"].value_counts()[CLASS_ORDER]
    colors = [PALETTE[c] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts, labels=CLASS_ORDER, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Target Class Distribution", fontsize=13, fontweight="bold", pad=15)
    save(fig, "class_distribution.png")


def plot_feature_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        for cls in CLASS_ORDER:
            subset = df[df["Fertility_Class"] == cls][feat]
            ax.hist(subset, bins=25, alpha=0.55, color=PALETTE[cls], label=cls, edgecolor="none")
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linewidth=0.5)

    # Legend in last subplot
    handles = [mpatches.Patch(color=PALETTE[c], label=c) for c in CLASS_ORDER]
    axes[-1].legend(handles=handles, fontsize=9, loc="center")
    axes[-1].axis("off")

    fig.suptitle("Feature Distributions by Fertility Class", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "feature_distributions.png")


def plot_boxplots_by_class(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        data_by_class = [df[df["Fertility_Class"] == cls][feat].values for cls in CLASS_ORDER]
        bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                        medianprops={"color": "white", "linewidth": 2},
                        whiskerprops={"linewidth": 1.2},
                        capprops={"linewidth": 1.2},
                        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
        for patch, cls in zip(bp["boxes"], CLASS_ORDER):
            patch.set_facecolor(PALETTE[cls])
            patch.set_alpha(0.85)
        short_labels = ["High", "Medium", "Low", "Poor"]
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_title(feat, fontsize=11)
        ax.grid(axis="y", linewidth=0.5)

    axes[-1].axis("off")
    fig.suptitle("Box Plots — Feature Range by Fertility Class", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "boxplots_by_class.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    corr = df[FEATURES].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="YlGn",
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        ax=ax, vmin=0, vmax=1
    )
    ax.set_title("Feature Correlation Matrix (Pearson)", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, "correlation_heatmap.png")

    # Save correlation values to results
    corr.to_csv(os.path.join(RESULTS_DIR, "correlation_matrix.csv"))
    print(f"  Correlation matrix CSV saved → outputs/results/correlation_matrix.csv")


def plot_scatter_N_vs_P(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in CLASS_ORDER:
        subset = df[df["Fertility_Class"] == cls]
        ax.scatter(subset["N"], subset["P"], c=PALETTE[cls], label=cls,
                   alpha=0.45, s=18, edgecolors="none")
    ax.set_xlabel("Nitrogen — N (mg/kg)", fontsize=11)
    ax.set_ylabel("Phosphorus — P (mg/kg)", fontsize=11)
    ax.set_title("Nitrogen vs Phosphorus by Fertility Class", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linewidth=0.4)
    fig.tight_layout()
    save(fig, "scatter_N_vs_P.png")


def plot_violin_ph(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=df, x="Fertility_Class", y="pH",
        order=CLASS_ORDER, palette=PALETTE,
        inner="quartile", linewidth=1.2, ax=ax
    )
    ax.set_xlabel("Fertility Class", fontsize=11)
    ax.set_ylabel("pH Level", fontsize=11)
    ax.set_title("pH Distribution by Fertility Class (Violin)", fontsize=13, fontweight="bold")
    ax.set_xticklabels(["High", "Medium", "Low", "Poor"])
    ax.axhline(7.0, color="gray", linestyle="--", linewidth=0.8, label="Neutral pH (7.0)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linewidth=0.4)
    fig.tight_layout()
    save(fig, "violin_ph_by_class.png")


def plot_pairplot(df: pd.DataFrame):
    subset = df[["N", "P", "K", "pH", "Fertility_Class"]].copy()
    g = sns.pairplot(
        subset, hue="Fertility_Class", hue_order=CLASS_ORDER,
        palette=PALETTE, plot_kws={"alpha": 0.35, "s": 15},
        diag_kind="kde", corner=True
    )
    g.figure.suptitle("Pair Plot — N, P, K, pH", fontsize=13, fontweight="bold", y=1.01)
    path = os.path.join(GRAPHS_DIR, "pairplot_npk_ph.png")
    g.figure.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def eda_summary(df: pd.DataFrame):
    print("\n── EDA Summary ─────────────────────────────────")
    print(f"  Shape          : {df.shape}")
    print(f"  Features       : {FEATURES}")
    print(f"\n  Descriptive Statistics:\n{df[FEATURES].describe().round(2).to_string()}")
    print(f"\n  Class Counts:\n{df['Fertility_Class'].value_counts().to_string()}")

    # Skewness
    skew = df[FEATURES].skew().round(3)
    print(f"\n  Feature Skewness:\n{skew.to_string()}")

    # Save summary to results
    summary_path = os.path.join(RESULTS_DIR, "eda_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== EDA SUMMARY — SoilHealth Predictor ===\n\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Class Distribution:\n")
        f.write(df["Fertility_Class"].value_counts().to_string() + "\n\n")
        f.write("Descriptive Statistics:\n")
        f.write(df[FEATURES].describe().round(3).to_string() + "\n\n")
        f.write("Skewness:\n")
        f.write(skew.to_string() + "\n")
    print(f"\n  EDA summary saved → {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SoilHealth Predictor — EDA & Visualisation")
    print("=" * 55)

    if not os.path.exists(RAW_PATH):
        print("  Raw dataset not found. Run src/preprocessing.py first.")
        return

    df = pd.read_csv(RAW_PATH)
    print(f"\n  Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    eda_summary(df)

    print("\n── Generating Plots ────────────────────────────")
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_boxplots_by_class(df)
    plot_correlation_heatmap(df)
    plot_scatter_N_vs_P(df)
    plot_violin_ph(df)
    plot_pairplot(df)

    print("\n✓ All plots saved to outputs/graphs/\n")


if __name__ == "__main__":
    main()
