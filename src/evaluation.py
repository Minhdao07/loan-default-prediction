import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from src.visualization import plot_roc_curve

def plot_feature_importance(model, feature_names, model_name):
    #Plot and print top 10 features by importance or coefficient magnitude,
    #helping to interpret how the model makes predictions and to understand which factors contribute most to the output.
    
    if hasattr(model, "feature_importances_"):  #Random Forest
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):  # Linear models like Logistic Regression
        importances = np.abs(model.coef_[0])
    else:
        print(f"{model_name} does not support feature importance.")
        return

    feat_importance = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n Top 10 features by importance:")
    for feat, score in feat_importance[:10]:
        print(f"{feat}: {score:.4f}")

    #Graph
    top_feats = feat_importance[:10]
    names = [f[0] for f in top_feats]
    scores = [f[1] for f in top_feats]
    plt.figure(figsize=(8, 6))
    plt.barh(names[::-1], scores[::-1])
    plt.xlabel("Importance")
    plt.title(f"{model_name} - Top 10 Features")
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test, model_name, feature_names=None):
    #Evaluate a trained model:
    #- Print classification report  
    #- Calculate and plot ROC AUC
    #- Plot feature importance
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    if y_proba is not None:
        auc_score = roc_auc_score(y_test, y_proba)
        print(f"{model_name} ROC AUC: {auc_score:.4f}")
        plot_roc_curve(y_test, y_proba, model_name=model_name)
    
    if feature_names is not None:
        plot_feature_importance(model, feature_names, model_name)