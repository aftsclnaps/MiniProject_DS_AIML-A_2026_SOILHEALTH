SoilHealth Predictor: Machine Learning for Soil Fertility Classification

Repository: MiniProject_DS_AIML-A_SoilHealthPredictor
Institution: SRM Institute of Science and Technology
Department: B.Tech AI & ML — Section A


Project Title

SoilHealth Predictor: Machine Learning for Soil Fertility Classification

Abstract

Soil fertility is a critical factor in determining agricultural productivity,
yet traditional soil testing methods are expensive, slow, and inaccessible
to small-scale farmers across rural India. This project proposes a machine
learning-based approach to classify soil fertility into four categories —
High, Medium, Low, and Poor — using seven measurable physicochemical
parameters: Nitrogen (N), Phosphorus (P), Potassium (K), pH, moisture
content, temperature, and electrical conductivity (EC).

A synthetic dataset of 2,200 samples was generated based on ICAR (Indian
Council of Agricultural Research) agronomic standards to simulate realistic
field conditions across different soil types. Four classification algorithms
were implemented and compared: Random Forest, Decision Tree, K-Nearest
Neighbours, and Gaussian Naive Bayes. Random Forest achieved the highest
test accuracy of 94.1% with a weighted F1-score of 0.938. The project also
includes an interactive prediction module that takes soil readings as input
and returns a fertility class along with a fertilization recommendation,
making it a practical tool for agricultural decision support.


Problem Statement

Farmers and agricultural planners in India frequently lack access to timely
and affordable soil health assessments. Manual soil testing through labs
takes 7–14 days and costs ₹500–₹2000 per test, making it impractical for
smallholder farmers who may need to test multiple plots seasonally.

This delay in soil fertility diagnosis leads to:
- Over-application of fertilizers, increasing cost and causing soil degradation
- Under-nourishment of crops, reducing yield by up to 30–40%
- Uninformed crop selection that does not match soil conditions

This project addresses the need for an automated, data-driven system that
can classify soil fertility rapidly using measurable on-site sensor readings,
and provide actionable fertilization recommendations without the need for
laboratory testing.

Dataset Source

- Type:     Synthetic dataset (NumPy random generation, class-conditional Gaussian)
- Standard: ICAR (Indian Council of Agricultural Research) nutrient range guidelines
- Size:     2,200 samples × 7 features + 1 target class
- Features: N (mg/kg), P (mg/kg), K (mg/kg), pH, Moisture (%), Temperature (°C), EC (dS/m)
- Classes:  High Fertility | Medium Fertility | Low Fertility | Poor Fertility
- Files:    dataset/raw_data/soil_data.csv
            dataset/processed_data/train.csv
            dataset/processed_data/test.csv

Class distribution:
  High Fertility    — 500 samples (22.7%)
  Medium Fertility  — 700 samples (31.8%)
  Low Fertility     — 600 samples (27.3%)
  Poor Fertility    — 400 samples (18.2%)

Methodology / Workflow

Stage 1 — Problem Identification
  Identified the gap in affordable real-time soil fertility testing for
  Indian smallholder farmers. Defined classification targets based on ICAR
  nutrient classification standards.

Stage 2 — Dataset Collection
  Generated synthetic dataset using NumPy with class-conditional Gaussian
  distributions calibrated to ICAR standard nutrient ranges for each
  fertility class. Added realistic noise (std = 10–20% of mean per feature).

Stage 3 — Data Cleaning / Preprocessing
  - Null value check: no missing values found
  - Duplicate check: no duplicates found
  - IQR-based outlier detection: outliers retained (genuine edge cases)
  - Label encoding: LabelEncoder (4 classes → 0, 1, 2, 3)
  - Feature scaling: StandardScaler (fit on train set only)
  - Train-test split: 80% train / 20% test, stratified by class

Stage 4 — Exploratory Data Analysis
  - Descriptive statistics (mean, std, skewness, kurtosis)
  - Pearson correlation matrix across all 7 features
  - Class distribution analysis
  - IQR outlier counts per feature

Stage 5 — Data Visualization
  - Class distribution pie and bar charts
  - Histograms of all 7 features by class
  - Box plots per feature per fertility class
  - Correlation heatmap (Pearson)
  - N vs P scatter plot coloured by class
  - Violin plot of pH by class
  - Pair plot (N, P, K, pH)

Stage 6 — Model Development
  Four classifiers trained and evaluated:
  1. Random Forest     (n_estimators=100, max_depth=10)
  2. Decision Tree     (max_depth=8)
  3. K-Nearest Neighbours (k=5, metric=euclidean)
  4. Gaussian Naive Bayes

Stage 7 — Result Interpretation
  Models evaluated on: accuracy, precision, recall, F1-score (weighted),
  confusion matrix, and 5-fold stratified cross-validation. Feature
  importances extracted from Random Forest to identify key predictors.

Tools Used

| Category         | Tool / Library            | Version  |
|------------------|---------------------------|----------|
| Language         | Python                    | 3.10     |
| ML Library       | scikit-learn              | 1.3.0    |
| Data Handling    | pandas                    | 2.0.3    |
| Numerical        | numpy                     | 1.24.3   |
| Visualisation    | matplotlib                | 3.7.2    |
| Visualisation    | seaborn                   | 0.12.2   |
| Notebooks        | Jupyter Notebook           | 7.0.6    |
| Version Control  | Git & GitHub              | —        |

Results / Findings

Model comparison (test set, 440 samples):

| Model            | Accuracy | Weighted F1 | CV Mean (5-fold) |
|------------------|----------|-------------|------------------|
| Random Forest    | 94.1%    | 0.938       | 93.5% ± 0.7%     |
| KNN (k=5)        | 90.2%    | 0.899       | 89.8% ± 1.1%     |
| Decision Tree    | 88.3%    | 0.879       | 87.6% ± 1.4%     |
| Naive Bayes      | 81.7%    | 0.812       | 81.2% ± 1.8%     |

Key findings:
- Random Forest outperforms all other models by a margin of 3.9% over KNN
- pH is the single most important feature (23.1% importance in RF)
- Nitrogen (N) is the second most important feature (19.8%)
- EC and Temperature contribute the least to classification (4.8%, 8.9%)
- High Fertility is classified with the highest precision (0.96)
- Poor Fertility is the most challenging class (precision 0.92)
- The model generalises well: test accuracy (94.1%) ≈ CV mean (93.5%)
- Ensemble methods significantly outperform single-tree and probabilistic
  classifiers for this multiclass soil fertility task

Team Members

| Name                  | Register Number    | Contribution                          |
|-----------------------|--------------------|---------------------------------------|
| RAMYA SRI U           | RA2311026050042    | Model development, GitHub setup       |
|                       |                    | EDA & visualisation (analysis.py)     |
| HARSITHA G P          | RA2311026050197    | Preprocessing & dataset generation    |
|                       |                    | Report, README, documentation         |

GitHub Repository: https://github.com/aftsclnaps/MiniProject_DS_AIML-A_2026_SOILHEALTH.git


How to Run

Step 1: Clone the repository
git clone https://github.com/[username]/MiniProject_DS_AIML-B_2026_SoilHealthPredictor.git
cd MiniProject_DS_AIML-B_2026_SoilHealthPredictor

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Generate dataset and preprocess
python src/preprocessing.py

Step 4: Run EDA and save plots
python src/analysis.py

Step 5: Train models and evaluate
python src/model.py

*Submitted for SRM Institute of Science and Technology*
*Data Science Mini Project | April 2026*
*Deadline: 06 April 2026*
