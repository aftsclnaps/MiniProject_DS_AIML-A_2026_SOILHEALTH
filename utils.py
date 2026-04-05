"""
utils.py
Pure-Python utility functions consumed by app.py:
  - Soil Health Score (0-100)
  - Crop Recommendation System
  - Soil Improvement Suggestions
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ────────────────────────────────────────────────────────────────────────────
# 1.  Soil Health Score
# ────────────────────────────────────────────────────────────────────────────

# Ideal agronomic reference values
IDEAL = {
    "pH":           6.5,
    "Nitrogen":     100.0,   # kg/ha
    "Phosphorus":    70.0,
    "Potassium":    100.0,
    "Moisture":      50.0,   # %
    "OrganicCarbon":  3.5,   # %
}

# Max realistic values for normalisation
MAX_VAL = {
    "pH":           9.0,
    "Nitrogen":   150.0,
    "Phosphorus": 150.0,
    "Potassium":  150.0,
    "Moisture":   100.0,
    "OrganicCarbon": 10.0,
}

# Contribution weights (must sum to 1.0)
WEIGHTS = {
    "Nitrogen":     0.22,
    "Phosphorus":   0.20,
    "Potassium":    0.20,
    "pH":           0.18,
    "Moisture":     0.10,
    "OrganicCarbon":0.10,
}


def compute_health_score(inputs: dict) -> dict:
    """
    Returns:
        score      – integer 0-100
        grade      – Excellent / Good / Fair / Poor
        sub_scores – per-feature contribution (%)
    """
    sub_scores = {}
    weighted_sum = 0.0

    for feat, weight in WEIGHTS.items():
        val = inputs[feat]

        if feat == "pH":
            # Bell-curve: perfect at 6.5, degrades symmetrically
            deviation = abs(val - IDEAL["pH"])
            norm = max(0.0, 1.0 - deviation / 2.5)
        else:
            norm = min(val / IDEAL[feat], 1.0)   # capped at ideal

        contribution  = norm * weight
        weighted_sum += contribution
        sub_scores[feat] = round(norm * 100, 1)

    score = int(round(weighted_sum * 100))
    score = max(0, min(100, score))

    if score >= 75:
        grade = "Excellent"
    elif score >= 55:
        grade = "Good"
    elif score >= 35:
        grade = "Fair"
    else:
        grade = "Poor"

    return {"score": score, "grade": grade, "sub_scores": sub_scores}


# ────────────────────────────────────────────────────────────────────────────
# 2.  Crop Recommendation System
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CropRec:
    name:    str
    emoji:   str
    reason:  str
    season:  str = ""


CROP_DB: dict[str, List[CropRec]] = {
    "High": [
        CropRec("Rice",      "🌾", "Thrives in high-fertility, moist soils",         "Kharif"),
        CropRec("Wheat",     "🌾", "Benefits from nutrient-rich loamy soil",          "Rabi"),
        CropRec("Sugarcane", "🌿", "Heavy feeder; responds well to high NPK",        "Annual"),
        CropRec("Banana",    "🍌", "Demands high potassium and rich organic matter",  "Perennial"),
        CropRec("Maize",     "🌽", "High nitrogen utilisation for grain yield",       "Kharif"),
    ],
    "Medium": [
        CropRec("Maize",     "🌽", "Adapts well to moderate fertility",               "Kharif"),
        CropRec("Cotton",    "🌿", "Moderate NPK demand, prefers well-drained soil",  "Kharif"),
        CropRec("Soybean",   "🌱", "Fixes atmospheric N; improves soil fertility",   "Kharif"),
        CropRec("Groundnut", "🥜", "Tolerates moderate P and K levels",              "Kharif"),
        CropRec("Sunflower", "🌻", "Adaptable; moderate nutrient requirements",       "Rabi"),
    ],
    "Low": [
        CropRec("Pearl Millet", "🌾", "Drought-tolerant; performs on poor soils",    "Kharif"),
        CropRec("Sorghum",      "🌾", "Resilient to low-fertility conditions",        "Kharif"),
        CropRec("Cowpea",       "🌱", "Nitrogen-fixing legume; improves soil health", "Kharif"),
        CropRec("Lentils",      "🌿", "Low-input legume; fixes N in degraded soils",  "Rabi"),
        CropRec("Cassava",      "🌿", "Tolerates infertile, acidic soils well",       "Annual"),
    ],
}

# pH-override rules (applied before class-based lookup)
PH_OVERRIDES: List[dict] = [
    {"condition": lambda ph: ph < 5.0,
     "crops": [
         CropRec("Tea",        "🍵", "Thrives in strongly acidic soils (pH 4–5)",   "Perennial"),
         CropRec("Blueberry",  "🫐", "Prefers acidic, well-drained soils",          "Perennial"),
         CropRec("Potato",     "🥔", "Tolerates mild-to-moderate acidity well",      "Rabi"),
     ]},
    {"condition": lambda ph: ph > 8.0,
     "crops": [
         CropRec("Barley",     "🌾", "Tolerates alkaline, calcareous soils",         "Rabi"),
         CropRec("Sugarbeet",  "🌿", "Well-suited to neutral-alkaline conditions",   "Annual"),
         CropRec("Asparagus",  "🌱", "Prefers slightly alkaline, sandy loam",        "Perennial"),
     ]},
]


def get_crop_recommendations(fertility_class: str, inputs: dict) -> List[CropRec]:
    """Return 3-5 crop recommendations based on fertility class and pH."""
    ph = inputs["pH"]

    for rule in PH_OVERRIDES:
        if rule["condition"](ph):
            return rule["crops"]

    return CROP_DB.get(fertility_class, CROP_DB["Medium"])


# ────────────────────────────────────────────────────────────────────────────
# 3.  Soil Improvement Suggestions
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Suggestion:
    icon:    str
    title:   str
    detail:  str
    urgency: str   # High / Medium / Low


def get_suggestions(inputs: dict) -> List[Suggestion]:
    """Generate human-readable, farmer-friendly improvement advice."""
    suggestions: List[Suggestion] = []

    n, p, k    = inputs["Nitrogen"], inputs["Phosphorus"], inputs["Potassium"]
    ph, m, oc  = inputs["pH"],       inputs["Moisture"],  inputs["OrganicCarbon"]

    # ── Nitrogen ────────────────────────────────────────────────────────────
    if n < 40:
        suggestions.append(Suggestion(
            icon="🌿", title="Boost Nitrogen",
            detail="Apply compost (5–10 t/ha) or urea fertilizer (100–200 kg/ha). "
                   "Consider intercropping with legumes for natural N-fixation.",
            urgency="High"
        ))
    elif n < 70:
        suggestions.append(Suggestion(
            icon="🌿", title="Supplement Nitrogen",
            detail="Top-dress with ammonium sulfate (60–80 kg/ha) mid-season "
                   "to sustain crop growth.",
            urgency="Medium"
        ))

    # ── Phosphorus ──────────────────────────────────────────────────────────
    if p < 30:
        suggestions.append(Suggestion(
            icon="🌱", title="Increase Phosphorus",
            detail="Apply Single Super Phosphate (SSP) at 100–150 kg/ha or "
                   "Di-Ammonium Phosphate (DAP) at 50–75 kg/ha at sowing.",
            urgency="High"
        ))
    elif p < 55:
        suggestions.append(Suggestion(
            icon="🌱", title="Maintain Phosphorus",
            detail="Incorporate rock phosphate (2–3 t/ha) as a slow-release "
                   "amendment to sustain levels.",
            urgency="Low"
        ))

    # ── Potassium ───────────────────────────────────────────────────────────
    if k < 40:
        suggestions.append(Suggestion(
            icon="🍂", title="Replenish Potassium",
            detail="Apply Muriate of Potash (MOP) at 80–100 kg/ha. Potassium "
                   "improves drought tolerance and disease resistance.",
            urgency="High"
        ))
    elif k < 70:
        suggestions.append(Suggestion(
            icon="🍂", title="Top-up Potassium",
            detail="Use Sulphate of Potash (SOP) at 40–60 kg/ha, especially "
                   "for quality-sensitive crops like fruits and vegetables.",
            urgency="Medium"
        ))

    # ── pH ──────────────────────────────────────────────────────────────────
    if ph < 5.5:
        suggestions.append(Suggestion(
            icon="⚗️", title="Correct Soil Acidity",
            detail="Apply agricultural lime (dolomite) at 2–4 t/ha. Retest pH "
                   "after 6 weeks. Avoid applying lime with ammonium fertilizers.",
            urgency="High"
        ))
    elif ph < 6.0:
        suggestions.append(Suggestion(
            icon="⚗️", title="Raise pH Slightly",
            detail="Light lime application (0.5–1 t/ha) will shift pH toward "
                   "the ideal 6.0–7.0 window for most crops.",
            urgency="Medium"
        ))
    elif ph > 8.0:
        suggestions.append(Suggestion(
            icon="⚗️", title="Reduce Alkalinity",
            detail="Apply elemental sulfur (200–500 kg/ha) or gypsum (1–2 t/ha) "
                   "to acidify the soil gradually over one season.",
            urgency="High"
        ))
    elif ph > 7.5:
        suggestions.append(Suggestion(
            icon="⚗️", title="Monitor Alkalinity",
            detail="Incorporate organic matter to buffer alkalinity. Use "
                   "acidifying fertilizers (ammonium sulfate) as needed.",
            urgency="Low"
        ))

    # ── Moisture ────────────────────────────────────────────────────────────
    if m < 20:
        suggestions.append(Suggestion(
            icon="💧", title="Urgent: Improve Moisture",
            detail="Install drip or sprinkler irrigation. Add mulch (3–5 cm) "
                   "to retain soil moisture. Consider drought-tolerant varieties.",
            urgency="High"
        ))
    elif m < 35:
        suggestions.append(Suggestion(
            icon="💧", title="Increase Soil Moisture",
            detail="Schedule regular irrigation to maintain 40–60% field capacity. "
                   "Add organic mulch to reduce evaporation losses.",
            urgency="Medium"
        ))

    # ── Organic Carbon ──────────────────────────────────────────────────────
    if oc < 1.0:
        suggestions.append(Suggestion(
            icon="🌾", title="Restore Organic Matter",
            detail="Incorporate green manure crops (dhaincha, sunhemp) or "
                   "vermicompost (3–5 t/ha). Avoid burning crop residues.",
            urgency="High"
        ))
    elif oc < 2.0:
        suggestions.append(Suggestion(
            icon="🌾", title="Build Organic Carbon",
            detail="Add farmyard manure (FYM) at 10–15 t/ha annually and "
                   "practice minimal tillage to preserve soil structure.",
            urgency="Medium"
        ))

    # ── All-clear message ───────────────────────────────────────────────────
    if not suggestions:
        suggestions.append(Suggestion(
            icon="✅", title="Soil Health Excellent",
            detail="All parameters are within optimal ranges. Continue current "
                   "practices and conduct annual soil testing to maintain health.",
            urgency="Low"
        ))

    return suggestions
