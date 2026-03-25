# 📡 Telco Customer Churn Prediction

> Predicting which telecom customers are likely to leave — before they do.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

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
├── 📂 documents/             # Project report and presentation
├── 📂 figures/               # Key visualizations
├── 📂 notebooks/             # Jupyter notebooks (cleaning, EDA, modeling)
├── 📂 src/                   # Streamlit demo app
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Telco Customer Churn (academic) |
| Records | ~67,987 customers |
| Features | 24 (after feature engineering) |
| Target | `Churn` (1 = churned, 0 = retained) |

**Key features:** tenure, contract type, internet service, monthly charges, payment method

---

## 🔬 Project Workflow

### 1 — Data Cleaning
- Handled missing values across 21 columns
- Standardized inconsistent labels (gender, contract, payment method)
- Converted all categorical features to numeric format

### 2 — Feature Engineering
- `IsNewCustomer` — tenure ≤ 6 months
- `IsLongTermCustomer` — tenure ≥ 48 months
- `AvgMonthlyCharge` — TotalCharges / (tenure + 1)
- `TotalServices` — count of subscribed services

### 3 — Exploratory Data Analysis
- Churn rate by contract type, internet service, and tenure
- Feature correlation heatmap
- Distribution analysis for key variables

### 4 — Modeling & Evaluation

| Model | Accuracy | F1 Score | Recall |
|-------|----------|----------|--------|
| Logistic Regression ✅ | 77.0% | 0.787 | 89.6% |
| Random Forest | 76.4% | 0.776 | 86.8% |
| Neural Network | 76.8% | 0.784 | 89.5% |

> ✅ **Final model: Logistic Regression** — best balance of recall and interpretability.

---

## 💡 Key Findings

- 📋 **Contract type** is the strongest churn predictor — month-to-month customers churn at 3x the rate of two-year contract customers
- ⏱️ **New customers** (≤ 6 months tenure) are at the highest risk
- 🌐 **Fiber Optic** subscribers show slightly higher churn than DSL users
- 💳 **Electronic check** payment correlates with higher churn rates

---

## 🖥️ Streamlit Demo

An interactive dashboard was built to explore the data and predict churn for individual customers.

**To run locally:**
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
streamlit run src/app.py
```

**Features:**
- 🏠 Overview dashboard with KPIs
- 📊 Data explorer with interactive charts
- 🤖 Model performance comparison
- 🔮 Live churn prediction for any customer

---

## 👥 Team

| Name | Role |
|------|------|
| Linda Bsharat | EDA & Visualization Lead |
| Ahmad Hlawa | Data Lead |
| — | Modeling Lead |
| — | Documentation Lead |

---

## 🏫 About

**IBT × GGateway Data Science & Machine Learning Bootcamp**
Capstone Project — 2025
