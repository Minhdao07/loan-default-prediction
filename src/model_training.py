from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        oob_score=True,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"Random Forest trained with {n_estimators} trees, max_depth={max_depth}")
    print(f"OOB Score: {model.oob_score_:.4f}")
    return model

def cross_validate_model(model, X, y, cv=5, scoring='roc_auc'):
    #Returns mean cross-validation score.
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"Cross-validated {scoring} scores: {scores}")
    print(f"Mean {scoring}: {scores.mean():.4f}")
    return scores

def tune_random_forest(X_train, y_train):
    #Use GridSearchCV to find best parameters for RandomForest.
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print(f"Best Random Forest Params: {grid.best_params_}")
    return grid.best_estimator_