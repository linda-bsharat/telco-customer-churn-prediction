# 📡 Telco Customer Churn Prediction
> Identifying At-Risk Telecom Customers Using Machine Learning
> Predicting which telecom customers are likely to leave — before they do.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=flat-square&logo=streamlit)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Neural%20Network-FF6F00?style=flat-square&logo=tensorflow)
![PowerBI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=flat-square&logo=powerbi)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

---

## 🌐 Live Demo

**▶️ [Launch Streamlit App](https://telco-customer-churn-prediction-ibtggateway.streamlit.app/)**

---

## 🎯 Problem Statement

Customer churn is one of the biggest challenges facing telecom companies.
Losing a customer costs far more than retaining one.

This project builds a machine learning pipeline to **identify at-risk customers early**,
enabling the business to take proactive retention actions before churn occurs.

**Who benefits?** Telecom retention teams, customer success managers, and business analysts.

---

## 📁 Project Structure
```
telco-customer-churn-prediction/
│
├── 📂 data/                  # Raw and cleaned datasets
├── 📂 figures/               # Key visualizations and charts
├── 📂 notebooks/             # Jupyter notebooks (cleaning, EDA, modeling)
├── 📂 slides/                # Final presentation deck
├── 📂 src/                   # Streamlit interactive demo app
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Telco Customer Churn (academic use) |
| Records | ~67,987 customers |
| Features | 24 (after feature engineering) |
| Target | `Churn` (1 = churned, 0 = retained) |

**Key features:** tenure · contract type · internet service · monthly charges · payment method

---

## 🔬 Project Workflow

### 01 — Data Cleaning `notebooks/01_data_cleaning_and_eda.ipynb`
- Handled missing values across all columns
- Standardized inconsistent labels (gender, contract, payment method)
- Fixed logical inconsistencies in service columns
- Converted all features to numeric format
- Clipped outliers using IQR method

### 02 — Feature Engineering
| Feature | Description |
|---------|-------------|
| `IsNewCustomer` | Tenure ≤ 6 months |
| `IsLongTermCustomer` | Tenure ≥ 48 months |
| `AvgMonthlyCharge` | TotalCharges / (tenure + 1) |
| `TotalServices` | Count of subscribed services |

### 03 — Exploratory Data Analysis `notebooks/02_eda_and_visualization.ipynb`
- Churn distribution and rate analysis
- Churn by contract type, internet service, and tenure groups
- Feature correlation heatmap
- Top features correlated with churn

### 04 — Modeling & Evaluation `notebooks/03_model_building.ipynb`

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| ✅ Logistic Regression | 77.0% | 70.1% | 89.6% | 78.7% |
| Random Forest | 76.4% | 70.2% | 86.8% | 77.6% |
| Neural Network | 76.8% | 69.7% | 89.5% | 78.4% |

> ✅ **Final model: Logistic Regression**
> Best balance of recall and interpretability.
> High recall (89.6%) means we correctly identify most customers who will churn.

---

## 💡 Key Findings

| # | Finding |
|---|---------|
| 📋 | **Contract type** is the strongest predictor — month-to-month customers churn at 3× the rate of two-year contract holders |
| ⏱️ | **New customers** (≤ 6 months) are at the highest churn risk |
| 🌐 | **Fiber Optic** subscribers show higher churn than DSL users |
| 💳 | **Electronic check** payment method correlates with higher churn |
| 📦 | Customers with **more services** tend to be more loyal |

---

## 🖥️ Streamlit Demo

**▶️ Live at: [telco-customer-churn-prediction-ibtggateway.streamlit.app](https://telco-customer-churn-prediction-ibtggateway.streamlit.app/)**

An interactive dashboard to explore the data and predict churn for any customer.

**To run locally:**
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
streamlit run src/app.py
```

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, churn distribution, key findings |
| 📊 Data Explorer | Interactive charts and feature analysis |
| 🤖 Model Performance | Metrics, confusion matrix, ROC curves |
| 🔮 Predict Customer | Live churn prediction with risk factor breakdown |

---

## 📊 Power BI Dashboard

An executive-level Power BI dashboard was built to complement the ML pipeline,
providing business-friendly visualizations of churn trends, customer segments,
and key risk indicators for non-technical stakeholders.

---

## 📂 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_cleaning_and_eda.ipynb` | Data cleaning, preprocessing, feature engineering |
| `02_eda_and_visualization.ipynb` | Exploratory analysis and visualizations |
| `03_model_building.ipynb` | Model training, evaluation, and comparison |
| `Models_Compare.ipynb` | Final model selection and decision |

---

## 👥 Team

| Name |
|------|
| Linda Bsharat |
| Ahmad Hlawa |
| Naser abdalsalam|
| Ibrahim Kilani |
| Ruba Halabi |


