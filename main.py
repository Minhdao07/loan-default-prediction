import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.preprocessing import preprocess_data
from src.visualization import plot_correlation_heatmap, plot_numeric_distributions
from src.model_training import (
    train_logistic_regression,
    train_random_forest,
    cross_validate_model,
    tune_random_forest
)
from src.evaluation import evaluate_model


if __name__ == "__main__":
    # 1. Data Preprocessing
    print("Starting data preprocessing...")
    preprocess_data(
        input_path="data/raw/Loan_Default.csv",
        output_path="data/processed/loan_default_cleaned.csv"
    )

    df_clean = pd.read_csv("data/processed/loan_default_cleaned.csv")
    print(f"Processed dataset shape: {df_clean.shape}")

    # 2. Exploratory Data Analysis (EDA)
    print("Generating correlation heatmap...")
    plot_correlation_heatmap(df_clean)

    print("Plotting numeric feature distributions...")
    plot_numeric_distributions(df_clean)

    # 3. Automatically detect suspected leakage to prevent optimistic bias 
    print("Detecting suspected leakage features...")

    suspected_leakage = []

    # Rule 1: High correlation with target (too good to be true)
    if 'status' in df_clean.columns:
        numeric_df = df_clean.select_dtypes(include=[np.number])
        corr_with_target = numeric_df.corr()['status'].drop('status')
        suspected_corr_leakage = corr_with_target[abs(corr_with_target) > 0.75].index.tolist()
        suspected_leakage.extend(suspected_corr_leakage)

    # Rule 2: Column names suggesting leakage
    leak_keywords = ['id', 'interest', 'charges', 'spread', 'rate', 'decision']
    # id: Unique identifiers should not be predictive of the target.
    # interest: This may correlate with the loan amount and other financial metrics, potentially leaking information about the loan's terms.
    # charges: Similar to interest, this could be related to the loan amount and terms, providing indirect information about the borrower's risk.
    # spread: This often relates to the difference between the interest rate and a benchmark rate, which could leak information about the loan's pricing.
    # rate: This is a direct measure of the loan's cost and could be highly correlated with the target variable.
    # decision: Similar to approval, this could relate to the final outcome of the loan application process.

    leak_from_name = [col for col in df_clean.columns if any(k in col.lower() for k in leak_keywords)]
    suspected_leakage.extend(leak_from_name)

    # Remove duplicates & ensure columns exist
    suspected_leakage = list(set(suspected_leakage))
    suspected_leakage = [col for col in suspected_leakage if col in df_clean.columns]

    if suspected_leakage:
        print(f"Removing suspected leakage features based on heuristics: {suspected_leakage}")
        df_clean.drop(columns=suspected_leakage, inplace=True)
    else:
        print("No suspected leakage features detected.")

   
    #4. Train/Test Split
    X = df_clean.drop(columns=["status"], errors="ignore")
    y = df_clean["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #5. One-hot encoding and alignment
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    #6. Multicollinearity check (only on continuous numeric features)
    continuous_features = X_train.select_dtypes(include=[np.number])
    vif_list = []
    for i in range(continuous_features.shape[1]):
        vif = variance_inflation_factor(continuous_features.values, i)
        vif_list.append(vif)

    vif_data = pd.DataFrame({
        "feature": continuous_features.columns,
        "VIF": vif_list
    }).sort_values(by="VIF", ascending=False)

    print("\nTop features with highest VIF:")
    print(vif_data.head(10))

    #7. Scaling for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 8. Model Training & Evaluation
    models = {
        "Logistic Regression": (train_logistic_regression, X_train_scaled, X_test_scaled),
        "Random Forest": (lambda Xt, yt: train_random_forest(Xt, yt, n_estimators=200, max_depth=5), X_train, X_test)
    }

    model_results = {}
    # 8a. Logistic Regression
    print(f"\nTraining Logistic Regression...")
    log_model = train_logistic_regression(X_train_scaled, y_train)
    cross_validate_model(log_model, X_train_scaled, y_train)
    evaluate_model(log_model, X_test_scaled, y_test, model_name="Logistic Regression", feature_names=X_train.columns)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_pred_log = log_model.predict(X_test_scaled)
    y_proba_log = log_model.predict_proba(X_test_scaled)[:, 1]
    model_results["Logistic Regression"] = {
        'accuracy': accuracy_score(y_test, y_pred_log),
        'precision': precision_score(y_test, y_pred_log),
        'recall': recall_score(y_test, y_pred_log),
        'f1_score': f1_score(y_test, y_pred_log),
        'roc_auc': roc_auc_score(y_test, y_proba_log)
    }
    # 8b. Random Forest with Hyperparameter Tuning
    print(f"\nTuning Random Forest (GridSearchCV)...")
    best_rf_model = tune_random_forest(X_train, y_train)
    cross_validate_model(best_rf_model, X_train, y_train)
    evaluate_model(best_rf_model, X_test, y_test, model_name="Random Forest", feature_names=X_train.columns)

    y_pred_rf = best_rf_model.predict(X_test)
    y_proba_rf = best_rf_model.predict_proba(X_test)[:, 1]
    model_results["Random Forest"] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1_score': f1_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_proba_rf)
    }

    #9. Model Performance Comparison
    print("Model Performance Comparision")
    if model_results:
        comparison_df = pd.DataFrame(model_results).T
        print(comparison_df.round(4))

    #10. Conclusion: Which model is better?
    print("\nModel Comparision Conclusion")
    best_metrics = comparison_df.idxmax()
    for metric, best_model in best_metrics.items():
        print(f"→ Best {metric}: {best_model}")

    # Define primary metric to pick best overall model by ROC_AUC
    primary_metric = 'roc_auc'
    best_overall_model = comparison_df[primary_metric].idxmax()
    print(f"\n Best Overall Model (by {primary_metric.upper()}): {best_overall_model}")