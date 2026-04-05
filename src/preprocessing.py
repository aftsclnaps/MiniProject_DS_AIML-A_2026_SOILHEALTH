"""
preprocessing.py
SoilHealth Predictor — Data Generation & Preprocessing
SRM Institute of Science and Technology | Data Science Mini Project 2026

Steps:
    1. Generate synthetic soil dataset (ICAR agronomic ranges)
    2. Add realistic noise per class
    3. Save raw dataset to dataset/raw_data/soil_data.csv
    4. Clean, encode, scale, and save to dataset/processed_data/soil_data_processed.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH   = os.path.join(BASE_DIR, "dataset", "raw_data",       "soil_data.csv")
PROC_PATH  = os.path.join(BASE_DIR, "dataset", "processed_data", "soil_data_processed.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "processed_data", "train.csv")
TEST_PATH  = os.path.join(BASE_DIR, "dataset", "processed_data", "test.csv")

os.makedirs(os.path.dirname(RAW_PATH),  exist_ok=True)
os.makedirs(os.path.dirname(PROC_PATH), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

# Class definitions based on ICAR soil nutrient guidelines
# Each class: {feature: (mean, std)}
CLASS_PARAMS = {
    "High Fertility": {
        "N":           (148, 18),
        "P":           (76,  12),
        "K":           (224, 22),
        "pH":          (6.9,  0.25),
        "Moisture":    (46,   6),
        "Temperature": (24,   3),
        "EC":          (0.9,  0.2),
    },
    "Medium Fertility": {
        "N":           (92,  16),
        "P":           (46,  10),
        "K":           (178, 20),
        "pH":          (6.5,  0.3),
        "Moisture":    (38,   7),
        "Temperature": (26,   3),
        "EC":          (1.3,  0.25),
    },
    "Low Fertility": {
        "N":           (56,  14),
        "P":           (28,   8),
        "K":           (140, 18),
        "pH":          (6.0,  0.35),
        "Moisture":    (28,   6),
        "Temperature": (29,   4),
        "EC":          (1.9,  0.3),
    },
    "Poor Fertility": {
        "N":           (24,  10),
        "P":           (12,   5),
        "K":           (88,  15),
        "pH":          (5.1,  0.4),
        "Moisture":    (18,   5),
        "Temperature": (33,   4),
        "EC":          (2.9,  0.5),
    },
}

# Samples per class
CLASS_COUNTS = {
    "High Fertility":   500,
    "Medium Fertility": 700,
    "Low Fertility":    600,
    "Poor Fertility":   400,
}

FEATURES = ["N", "P", "K", "pH", "Moisture", "Temperature", "EC"]


def generate_dataset() -> pd.DataFrame:
    """Generate synthetic soil samples with Gaussian noise per class."""
    records = []
    for cls, count in CLASS_COUNTS.items():
        params = CLASS_PARAMS[cls]
        for _ in range(count):
            row = {feat: max(0.01, np.random.normal(params[feat][0], params[feat][1]))
                   for feat in FEATURES}
            # Clip pH to valid agronomic range [3.5, 9.5]
            row["pH"] = np.clip(row["pH"], 3.5, 9.5)
            row["Fertility_Class"] = cls
            records.append(row)

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning: type rounding, outlier check, null check."""
    print("\n── Data Cleaning ──────────────────────────────")

    # Round continuous features for readability
    for col in ["N", "P", "K", "Moisture", "Temperature"]:
        df[col] = df[col].round(1)
    df["pH"] = df["pH"].round(2)
    df["EC"]  = df["EC"].round(2)

    # Null check
    nulls = df.isnull().sum()
    print(f"  Null values:\n{nulls[nulls > 0] if nulls.sum() > 0 else '  None found ✓'}")

    # Duplicate check
    dupes = df.duplicated().sum()
    print(f"  Duplicate rows: {dupes}")
    if dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  Dropped {dupes} duplicates.")

    # IQR-based outlier report (do not remove — they may be genuine edge cases)
    print("\n  Outlier counts (IQR method, not removed):")
    for col in FEATURES:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        outs = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].shape[0]
        if outs > 0:
            print(f"    {col}: {outs} outliers")

    print(f"\n  Final shape after cleaning: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Encode labels, scale features, train-test split.
    Returns X_train_s, X_test_s, y_train, y_test, scaler, label_encoder, df_processed
    """
    print("\n── Preprocessing ──────────────────────────────")

    # Label encoding
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Fertility_Class"])
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Encoded: {list(range(len(le.classes_)))}")

    X = df[FEATURES].values
    y = df["Label"].values

    # Train-test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # Feature scaling (StandardScaler fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"  Scaling: mean={scaler.mean_.round(2)}")
    print(f"           std ={scaler.scale_.round(2)}")

    # Build processed DataFrame
    train_df = pd.DataFrame(X_train_s, columns=FEATURES)
    train_df["Label"] = y_train
    train_df["Split"] = "train"

    test_df = pd.DataFrame(X_test_s, columns=FEATURES)
    test_df["Label"] = y_test
    test_df["Split"] = "test"

    df_processed = pd.concat([train_df, test_df], ignore_index=True)

    return X_train_s, X_test_s, y_train, y_test, scaler, le, df_processed


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SoilHealth Predictor — Preprocessing Pipeline")
    print("=" * 55)

    # Generate
    print("\n── Dataset Generation ─────────────────────────")
    df_raw = generate_dataset()
    print(f"  Generated {len(df_raw)} samples across {df_raw['Fertility_Class'].nunique()} classes")
    print(f"  Class distribution:\n{df_raw['Fertility_Class'].value_counts().to_string()}")
    print(f"\n  Feature statistics:\n{df_raw[FEATURES].describe().round(2).to_string()}")

    # Save raw
    df_raw.to_csv(RAW_PATH, index=False)
    print(f"\n  Raw dataset saved → {RAW_PATH}")

    # Clean
    df_clean = clean_data(df_raw.copy())

    # Preprocess
    X_train_s, X_test_s, y_train, y_test, scaler, le, df_proc = preprocess(df_clean)

    # Save processed
    df_proc.to_csv(PROC_PATH, index=False)
    print(f"\n  Processed dataset saved → {PROC_PATH}")

    # Save train/test splits
    train_split = df_proc[df_proc["Split"] == "train"].drop(columns=["Split"])
    test_split  = df_proc[df_proc["Split"] == "test"].drop(columns=["Split"])
    train_split.to_csv(TRAIN_PATH, index=False)
    test_split.to_csv(TEST_PATH,  index=False)
    print(f"  Train split saved → {TRAIN_PATH}")
    print(f"  Test  split saved → {TEST_PATH}")

    print("\n✓ Preprocessing complete.\n")


if __name__ == "__main__":
    main()
