"""
generate_dataset.py
Generates a synthetic soil dataset with realistic feature ranges and labels.
Run this once to produce dataset.csv before training.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N_SAMPLES = 2000

# --- Feature generation with realistic distributions ---
pH       = np.random.uniform(4.0, 9.0, N_SAMPLES)
nitrogen = np.random.uniform(0, 150, N_SAMPLES)
phosphorus = np.random.uniform(0, 150, N_SAMPLES)
potassium  = np.random.uniform(0, 150, N_SAMPLES)
moisture   = np.random.uniform(0, 100, N_SAMPLES)
organic_c  = np.random.uniform(0, 10, N_SAMPLES)

# --- Label creation based on agronomic thresholds ---
def assign_label(n, p, k, ph, m, oc):
    score = 0
    score += 2 if n > 80 else (1 if n > 40 else 0)
    score += 2 if p > 60 else (1 if p > 30 else 0)
    score += 2 if k > 80 else (1 if k > 40 else 0)
    score += 2 if 6.0 <= ph <= 7.0 else (1 if 5.5 <= ph <= 7.5 else 0)
    score += 1 if m > 30 else 0
    score += 1 if oc > 2.0 else 0
    if score >= 7:
        return "High"
    elif score >= 4:
        return "Medium"
    else:
        return "Low"

labels = [
    assign_label(nitrogen[i], phosphorus[i], potassium[i],
                 pH[i], moisture[i], organic_c[i])
    for i in range(N_SAMPLES)
]

df = pd.DataFrame({
    "pH": np.round(pH, 2),
    "Nitrogen": np.round(nitrogen, 1),
    "Phosphorus": np.round(phosphorus, 1),
    "Potassium": np.round(potassium, 1),
    "Moisture": np.round(moisture, 1),
    "OrganicCarbon": np.round(organic_c, 2),
    "FertilityClass": labels
})

df.to_csv("dataset.csv", index=False)
print(f"Dataset saved: {len(df)} rows")
print(df["FertilityClass"].value_counts())
