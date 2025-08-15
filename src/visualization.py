import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_auc_score

def plot_numeric_distributions(df):
    #Plot histograms for all numeric features to visualize their distributions, 
    # detect skewness, identify potential outliers, and understand the overall data spread before modeling.
    
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols].hist(bins=30, figsize=(15, 10), grid=False)
    plt.suptitle("Numeric Feature Distributions", fontsize=16)
    plt.show()

def plot_correlation_heatmap(df):
    # Select only numerical columns

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # Print description & analysis
    description = """
    Purpose of the Correlation Heatmap:
    The heatmap displays the strength of linear relationships between numerical features.
    High absolute correlation values (close to 1 or -1) indicate strong relationships,
    while values near 0 indicate weak relationships. This is useful for:
        - Detecting multicollinearity between features.
        - Identifying patterns or dependencies.
        - Guiding feature engineering and selection.

    Example Analysis:
    1. loan_amount and property_value: Strong positive correlation (approximately 0.69).
    2. rate_of_interest and interest_rate_spread: Strong positive correlation (approximately 0.62).
    3. interest_rate_spread and property_value: Moderate negative correlation (approximately -0.29).
    4. income and loan_amount: Moderate positive correlation (approximately 0.44).
    5. Most other features have weak correlations, suggesting low linear dependency.

    Modeling Implication:
    Highly correlated features may require regularization (L1/L2) or dimensionality reduction
    to avoid multicollinearity issues, especially in linear models.
    """
    return description 

def plot_roc_curve(y_true, y_proba, model_name="Model"):
    auc = roc_auc_score(y_true, y_proba)
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.plot([0, 1], [0, 1], 'r--', label='Random (AUC = 0.5)')
    plt.title(f"{model_name} ROC Curve (AUC = {auc:.3f})")
    plt.legend(loc="lower right")
    plt.show()


