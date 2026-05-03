# 🛡️ AssurIA — Insurance Subscription Prediction

> Predicting the probability that a customer will subscribe to a car insurance contract, using machine learning and an interactive Streamlit dashboard.

---

## 📋 Project Overview

This project was carried out as part of a Data Science consulting mission for an insurance company. The goal is to **anticipate which customers are likely to subscribe to a car insurance contract** during the year, in order to optimize marketing campaigns and focus commercial efforts on the most promising profiles.

The project follows the full Data Science lifecycle:

```
Data Exploration → Feature Engineering → Modeling → Evaluation → Production → Dashboard
```

---

## 🗂️ Project Structure

```
TP/
│
├── Data/
│   ├── train_info.csv                  # Labeled training dataset (381,109 clients)
│   ├── clients_a_contacter.csv         # Unlabeled production dataset
│   └── clients_cibles.csv              # Model output — targeted clients [0.3 – 0.7]
│
├── pages/
│   ├── Contexte.py                     # Page 1 — Project context & variable descriptions
│   ├── EDA.py                          # Page 2 — Interactive exploratory data analysis
│   ├── Modelisation.py                 # Page 3 — Model results, ROC curve, feature importance
│   ├── Prediction.py                   # Page 4 — Client targeting & filterable table
│   ├── Decision.py                     # Page 5 — Decision support & marketing recommendations
│   └── Prediction_individuelle.py      # Page 6 — Real-time individual prediction form
│
├── pkl_files/
│   ├── model.pkl                       # Trained XGBoost model
│   ├── onehot.pkl                      # OneHotEncoder (categorical variables)
│   ├── ordinal.pkl                     # OrdinalEncoder (age_vehicule)
│   ├── minmax.pkl                      # MinMaxScaler
│   ├── robust.pkl                      # RobustScaler
│   ├── standard.pkl                    # StandardScaler
│   ├── mean_canal.pkl                  # Business encoding — canal_communication
│   ├── mean_region.pkl                 # Business encoding — code_regional
│   ├── columns_order.pkl               # Training column order (alignment)
│   ├── minmax_cols.pkl                 # Column list for MinMaxScaler
│   ├── robust_cols.pkl                 # Column list for RobustScaler
│   └── standard_cols.pkl              # Column list for StandardScaler
│
├── app.py                              # Streamlit entry point
├── TP.ipynb                            # EDA + Feature Engineering + Modeling notebook
├── clinet_a_contacter.ipynb           # Production pipeline notebook
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 📊 Dataset

| File | Rows | Description |
|------|------|-------------|
| `train_info.csv` | 381,109 | Labeled clients used for training and evaluation |
| `clients_a_contacter.csv` | — | Unlabeled clients for production predictions |
| `clients_cibles.csv` | — | Output — clients in the uncertainty zone [0.3 – 0.7] |

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| `id_client` | ID | Unique client identifier |
| `genre` | Categorical | Client gender |
| `age` | Numerical | Client age |
| `permis_conduire` | Binary | 0 = no license, 1 = has license |
| `code_regional` | Categorical | Region code |
| `ancien_assure` | Binary | 0 = never insured, 1 = previously insured |
| `age_vehicule` | Ordinal | Vehicle age (< 1 yr, 1-2 yrs, > 2 yrs) |
| `vehicule_endommage` | Binary | 1 = vehicle was damaged, 0 = intact |
| `prime_annuelle` | Numerical | Annual premium amount (€) |
| `canal_communication` | Categorical | Anonymized communication channel |
| `anciennete` | Numerical | Client seniority in days |
| `reponse_client` | **Target** | 0 = not interested, 1 = interested |

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis
- No missing values or duplicates detected
- Strong class imbalance: **12.3%** positive class
- Key findings via Spearman correlation:
  - `ancien_assure` most correlated with target (−0.34)
  - `vehicule_endommage` strongly influences subscription
  - High correlation between `age` and `canal_communication` (−0.58)

### 2. Feature Engineering
- Age discretization into 7 bins → `tranche_age`
- 6 interaction variables combining age group, vehicle age, damage status and insurance history
- Encoding: `OneHotEncoder`, `OrdinalEncoder`, business-logic encoding for high-cardinality variables
- Scaling: `MinMaxScaler`, `RobustScaler`, `StandardScaler`

### 3. Modeling
- **Model:** XGBClassifier (XGBoost)
- **Class imbalance handling:** `scale_pos_weight = 7`
- **Hyperparameter tuning:** RandomizedSearchCV (20 iterations, 3-fold CV, F1 scoring)
- **Decision threshold:** 0.70 (optimized for F1-score)

### 4. Results

| Metric | Class 0 (No) | Class 1 (Yes) | Overall |
|--------|-------------|--------------|---------|
| Precision | 0.96 | 0.35 | — |
| Recall | 0.79 | **0.73** | — |
| F1-score | 0.87 | **0.47** | — |
| Accuracy | — | — | 0.79 |
| ROC-AUC | — | — | **0.84** |

### 5. Targeting Strategy

| Probability | Decision | Reason |
|-------------|----------|--------|
| `< 0.3` | ❌ Do not contact | Low ROI |
| `[0.3 – 0.7]` | ✅ **Contact — priority** | Uncertainty zone |
| `> 0.7` | ❌ Do not contact | Likely to subscribe without intervention |

---

## 🖥️ Streamlit Dashboard

| Page | File | Description |
|------|------|-------------|
| 📋 Context | `Contexte.py` | Project overview, KPIs, variable descriptions |
| 🔍 EDA | `EDA.py` | Interactive charts, filters, Spearman heatmap |
| 🤖 Modeling | `Modelisation.py` | Confusion matrix, ROC curve, feature importance |
| 🎯 Prediction & Targeting | `Prediction.py` | Filtered client table, probability distribution, CSV export |
| 💡 Decision Support | `Decision.py` | Client profiles, comparisons, marketing recommendations |
| 🔮 Individual Prediction | `Prediction_individuelle.py` | Real-time prediction form for a single client |

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run app.py
```

> **Note:** Make sure you run `TP.ipynb` first to generate all `.pkl` files inside the `pkl_files/` folder, then run `clinet_a_contacter.ipynb` to generate `clients_cibles.csv` inside `Data/`.

---

## 📁 Deliverables

- ✅ `TP.ipynb` — Full EDA and modeling notebook
- ✅ `clinet_a_contacter.ipynb` — Production pipeline notebook
- ✅ Streamlit dashboard (AssurIA) — 6 interactive pages
- ✅ Synthetic report (PDF + LaTeX source)

---

## 🏫 Academic Context

**School:** ENSIM — École d'ingénieurs, Le Mans Université  
**Course:** Data Visualization and Exploration  
**Academic Year:** 2024–2025  
**Supervisor:** Kais Hassan  
**Author:** El Moujahid Omar
