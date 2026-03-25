# Notebooks

This folder contains all Jupyter notebooks for the Telco Customer Churn project,
organized in order of execution.

## Notebook Order

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_data_cleaning_and_eda.ipynb` | Data loading, cleaning, handling missing values, encoding, and feature engineering |
| 02 | `02_eda_and_visualization.ipynb` | Exploratory data analysis with charts and insights |
| 03 | `03_model_building.ipynb` | Model training and evaluation (Logistic Regression, Random Forest, Neural Network) |
| — | `Models_Compare.ipynb` | Final model comparison and selection |

## How to Run

All notebooks are designed to run on **Google Colab**.

Click the badge at the top of each notebook to open it directly in Colab,
or run them locally in order from 01 → 02 → 03.

## Key Outputs

- Cleaned dataset saved to `data/telco_customer_data_cleaned.csv`
- Final model: **Logistic Regression** (best balance of accuracy and recall)
- All visualizations saved to `figures/`
