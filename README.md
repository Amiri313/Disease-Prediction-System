# 🏥 Healthcare Disease Prediction System (Symptom-Based)

## Project Description
This project builds a Machine Learning model that predicts a patient's disease based on which symptoms they are experiencing. The model is trained using the Random Forest Classifier algorithm on a real, clinically-grounded symptom-disease dataset containing 4,920 training records and a held-out test set of 42 records.

Given 132 binary symptom indicators (e.g. `itching`, `high_fever`, `joint_pain`, `yellowing_of_eyes`), the model classifies each patient into **one of 41 possible diagnoses**, including Diabetes, Hypertension, Malaria, Dengue, Typhoid, Migraine, Hepatitis (A–E), Pneumonia, Tuberculosis, Common Cold, Acne, and more.

---

## ML Algorithm Used
**Random Forest Classifier**

A Random Forest is an ensemble of many Decision Trees. Each tree independently predicts the disease class, and the final prediction is determined by majority voting across all trees. It is robust to overfitting, handles high-dimensional binary features well, and provides feature importance scores.

---

## Dataset

| | |
|---|---|
| Source | [Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) (Kaggle) |
| Features | 132 binary symptom columns (1 = present, 0 = absent) |
| Target | `prognosis` — 41 disease classes |
| Training samples | 4,920 (perfectly balanced: 120 per class) |
| Testing samples | 42 (one held-out example per class) |
| Missing values | None |

*A previous version of this project used a synthetically generated dataset with randomly assigned labels — see "Project History" below for why that dataset was replaced.*

---

## Preprocessing Steps

1. **Column cleanup** — Dropped a trailing empty/unnamed column present in the raw CSV export
2. **Whitespace stripping** — Removed extra spaces from the `prognosis` target column
3. **Label Encoding** — Target column encoded as integers (0–40) using `LabelEncoder`
4. No imputation needed (dataset has zero missing values) and no manual feature encoding needed (all symptom columns are already binary)

---

## Visualizations

### 1. Disease Distribution (Bar Chart)
Shows the count of patients per disease in the training set. The dataset is perfectly balanced — exactly 120 samples per disease across all 41 classes.

### 2. Top 20 Most Common Symptoms
Ranks symptoms by how frequently they appear across the training set, giving a sense of which symptoms are broadly shared across diseases versus which are rare/specific.

### 3. Confusion Matrix
Shows model prediction results per class on the test set. The matrix is almost purely diagonal, meaning the model correctly identifies the disease from its symptom pattern in nearly all cases.

### 4. Feature Importance Bar Chart
Ranks the 20 most influential symptoms the Random Forest relied on. General/systemic symptoms (fatigue, high fever, muscle pain) and disease-specific markers (yellowing of eyes, family history) rank highly — consistent with real clinical intuition.

---

## Accuracy & Results

| Metric | Value |
|---|---|
| Model | Random Forest Classifier |
| Train / Test Split | Provided (Training.csv / Testing.csv) |
| Training Samples | 4,920 |
| Test Samples | 42 |
| **Accuracy** | **~97.6%** |
| Random Baseline (41 classes) | ~2.4% |

> **Note:** Because each disease in this dataset has a distinctive, clinically-derived symptom signature, the Random Forest classifier separates the 41 classes almost perfectly. The test set is small (about one example per disease), so this figure has wide error bars — a larger held-out set would give a more statistically robust estimate. This is an educational project and is **not** a diagnostic tool; predictions should never replace professional medical evaluation.

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook Notebook.ipynb
```

Or run as a script:
```bash
python3 Notebook.py
```

### Files Required
```
project/
├── Notebook.ipynb              ← Main notebook
├── Training.csv                ← Training data (4,920 rows)
├── Testing.csv                 ← Held-out test data (42 rows)
├── README.md                   ← This file
├── requirements.txt            ← Python dependencies
```

---

## Project History
An earlier version of this project used a synthetic dataset (`dataset_health.csv`) with 5 disease classes (Asthma, Diabetes, Healthy, Heart Disease, Hypertension) where labels were randomly assigned and carried no true relationship to the input features. That version honestly reported ~21.67% accuracy (barely above the 20% random baseline for 5 balanced classes) — a technically correct but unimpressive result for a portfolio project. This version replaces that dataset with real symptom-disease data so the model's performance reflects genuine predictive signal.

---

## Author
**Jahangir Amiri**
Postgraduate Diploma in Data Science & AI
NED University of Engineering & Technology, Karachi
