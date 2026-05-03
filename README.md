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

## 🗂️ Repository Structure

```
assuria-insurance-prediction/
│
├── Data/
│   ├── train_info.csv              # Training dataset (labeled)
│   └── clients_a_contacter.csv     # Production dataset (unlabeled)
│
├── pages/
│   ├── 1_Contexte.py               # Project context page
│   ├── 2_EDA.py                    # Exploratory Data Analysis page
│   ├── 3_Modelisation.py           # Modeling results page
│   ├── 4_Prediction.py             # Client targeting page
│   ├── 5_Decision.py               # Decision support page
│   └── 6_Prediction_individuelle.py # Individual prediction page
│
├── utils/
│   └── data_loader.py              # Shared data loading functions
│
├── app.py                          # Streamlit main entry point
├── TP.ipynb                        # EDA + Modeling notebook
├── clinet_a_contacter.ipynb        # Production pipeline notebook
├── clients_cibles.csv              # Targeted clients (model output)
│
├── model.pkl                       # Trained XGBoost model
├── onehot.pkl                      # OneHotEncoder
├── ordinal.pkl                     # OrdinalEncoder
├── minmax.pkl                      # MinMaxScaler
├── robust.pkl                      # RobustScaler
├── standard.pkl                    # StandardScaler
├── mean_canal.pkl                  # Canal communication encoding
├── mean_region.pkl                 # Regional code encoding
├── columns_order.pkl               # Training columns order
├── standard_cols.pkl               # Columns for StandardScaler
├── minmax_cols.pkl                 # Columns for MinMaxScaler
├── robust_cols.pkl                 # Columns for RobustScaler
│
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 📊 Dataset

| File | Rows | Description |
|------|------|-------------|
| `train_info.csv` | 381,109 | Labeled clients used for training and evaluation |
| `clients_a_contacter.csv` | — | Unlabeled clients for production predictions |

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

Clients are targeted based on their predicted probability:

| Probability | Decision | Reason |
|-------------|----------|--------|
| `< 0.3` | ❌ Do not contact | Low ROI |
| `[0.3 – 0.7]` | ✅ **Contact — priority** | Uncertainty zone, action can influence decision |
| `> 0.7` | ❌ Do not contact | Likely to subscribe without intervention |

---

## 🖥️ Streamlit Dashboard

The **AssurIA** dashboard covers the full Data Science pipeline across 6 interactive pages:

| Page | Description |
|------|-------------|
| 📋 Context | Project overview, variable descriptions, KPIs |
| 🔍 EDA | Interactive charts, filters, Spearman correlation heatmap |
| 🤖 Modeling | Confusion matrix, ROC curve, feature importance |
| 🎯 Prediction & Targeting | Targeted clients table, probability distribution, CSV export |
| 💡 Decision Support | Client profiles, comparisons, marketing recommendations |
| 🔮 Individual Prediction | Real-time prediction form for a single client |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run app.py
```

### Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.18.0
scikit-learn>=1.4.0
xgboost>=2.0.0
joblib>=1.3.0
imbalanced-learn>=0.12.0
```

> **Note:** The `.pkl` files (model and encoders) must be present in the root directory. Run `TP.ipynb` first to generate them.

---

## 📁 Deliverables

- ✅ `TP.ipynb` — Full EDA and modeling notebook
- ✅ `clinet_a_contacter.ipynb` — Production pipeline notebook
- ✅ Streamlit dashboard (AssurIA)
- ✅ Synthetic report (PDF)

---

## 🏫 Academic Context

**School:** ENSIM — École d'ingénieurs, Le Mans Université  
**Course:** Data Visualization and Exploration  
**Academic Year:** 2024–2025  
**Supervisor:** Kais Hassan  
**Author:** El Moujahid Omar
