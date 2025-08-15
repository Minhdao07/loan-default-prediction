#  Loan Default Prediction

##  Project Overview
This project predicts the probability of loan default using a dataset from Kaggle.  
We build an **end-to-end Machine Learning pipeline**: from Exploratory Data Analysis (EDA) and data preprocessing to model training and evaluation.

** Goal:** Assist financial institutions in identifying high-risk borrowers to improve decision-making and reduce credit risk.

---

##  Objectives
1. Understand the dataset and identify patterns related to loan defaults.
2. Clean and preprocess the data for modeling.
3. Train a predictive model to estimate the probability of default.
4. Visualize important relationships between variables.
5. Document the workflow for reproducibility.

---

##  Project Structure

```
loan-default-prediction/
│
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned dataset
│
├── notebooks/              # Jupyter notebooks for EDA & experiments
│
├── src/
│   ├── preprocessing.py    # Data cleaning & feature engineering
│   ├── visualization.py    # Plotting functions
│   ├── model_training.py   # Model training functions
│   ├── evaluation.py       # Evaluation metrics & feature importance
│   └── __init__.py
│
├── outputs/                # Saved images, reports, or results
│
├── main.py                 # Main script to run pipeline
├── requirements.txt        # Python dependencies
└── README.md               # Project description
```

---

##  Exploratory Data Analysis (EDA)

Performed steps:
- Inspect dataset structure (rows, columns, datatypes)
- Missing value analysis
- Target variable distribution
- Summary statistics for numerical features
- Frequency counts for categorical features
- Correlation heatmap to identify relationships between numerical variables

** Correlation Heatmap Notes:**  
- Values close to +1 → strong positive correlation  
- Values close to -1 → strong negative correlation  
- Values near 0 → weak or no linear relationship  

 Helps detect **multicollinearity** and find features strongly related to the target.

---

##  Data Preprocessing

Steps applied:
- Handle missing values:
  - Numerical → filled with median
  - Categorical → filled with `"Unknown"`
- Remove duplicate rows
- Standardize column names (lowercase & underscores)
- Save cleaned dataset to:  
  `data/processed/loan_default_cleaned.csv`

---

##  Machine Learning Pipeline

- **Train-Test Split** for model evaluation
- **Feature Scaling** with `StandardScaler`
- **Models used**:
  - Logistic Regression (baseline, interpretable)
  - Random Forest (non-linear, robust)
- **Hyperparameters**:
  - Logistic Regression: `max_iter=1000`
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

---