# Data

This folder contains the datasets used in the Telco Customer Churn Prediction project.

## Files

| File | Description |
|------|-------------|
| `telco_customer_data_v2.csv` | Raw dataset before cleaning (~70,000 customer records, 21 features) |
| `telco_customer_data_cleaned.csv` | Final cleaned dataset used for modeling (24 features after feature engineering) |

## Dataset Overview

- **Source:** Telco Customer Churn dataset (academic use)
- **Rows:** ~67,987 customers
- **Target variable:** `Churn` (1 = churned, 0 = retained)

## Features

| Column | Type | Description |
|--------|------|-------------|
| `gender` | int | Customer gender (1 = Male, 0 = Female) |
| `SeniorCitizen` | int | Whether the customer is a senior citizen (1 = Yes, 0 = No) |
| `Partner` | int | Has a partner (1 = Yes, 0 = No) |
| `Dependents` | int | Has dependents (1 = Yes, 0 = No) |
| `tenure` | float | Number of months the customer has stayed |
| `PhoneService` | int | Has phone service (1 = Yes, 0 = No) |
| `MultipleLines` | int | Multiple lines (0 = No phone, 1 = No, 2 = Yes) |
| `InternetService` | int | Internet type (0 = No, 1 = DSL, 2 = Fiber optic) |
| `OnlineSecurity` | int | Has online security (1 = Yes, 0 = No) |
| `OnlineBackup` | int | Has online backup (1 = Yes, 0 = No) |
| `DeviceProtection` | int | Has device protection (1 = Yes, 0 = No) |
| `TechSupport` | int | Has tech support (1 = Yes, 0 = No) |
| `StreamingTV` | int | Has streaming TV (1 = Yes, 0 = No) |
| `StreamingMovies` | int | Has streaming movies (1 = Yes, 0 = No) |
| `Contract` | int | Contract type (0 = Month-to-month, 1 = One year, 2 = Two year) |
| `PaperlessBilling` | int | Uses paperless billing (1 = Yes, 0 = No) |
| `PaymentMethod_*` | int | One-hot encoded payment method columns |
| `IsNewCustomer` | int | Tenure ≤ 6 months (1 = Yes, 0 = No) |
| `IsLongTermCustomer` | int | Tenure ≥ 48 months (1 = Yes, 0 = No) |
| `AvgMonthlyCharge` | float | TotalCharges / (tenure + 1) |
| `TotalServices` | int | Total number of services subscribed |
| `Churn` | int | Target variable (1 = Churned, 0 = Retained) |

## Notes

- Dataset is uploaded for academic purposes only.
- All preprocessing steps are documented in `notebooks/01_data_cleaning_and_eda.ipynb`.
