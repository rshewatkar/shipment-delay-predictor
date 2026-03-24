import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)
from utils import (load_processed_data, save_model, print_separator)

# ── Evaluate helper ──────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'Accuracy' : round(accuracy_score(y_test, y_pred)  * 100, 2),
        'Precision': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall'   : round(recall_score(y_test, y_pred)    * 100, 2),
        'F1 Score' : round(f1_score(y_test, y_pred)        * 100, 2),
        'AUC-ROC'  : round(roc_auc_score(y_test, y_prob)   * 100, 2)
    }

# ── Train XGBoost ────────────────────────────────────────────
def train_xgboost(X_train, y_train):
    params = {
        'n_estimators'    : [100, 200, 300],
        'max_depth'       : [3, 4, 5, 6],
        'learning_rate'   : [0.01, 0.05, 0.1, 0.2],
        'subsample'       : [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma'           : [0, 0.1, 0.2]
    }
    xgb = XGBClassifier(use_label_encoder=False,
                        eval_metric='logloss',
                        random_state=42)
    search = RandomizedSearchCV(xgb, params, n_iter=20,
                                cv=5, scoring='f1',
                                random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best XGBoost params: {search.best_params_}")
    return search.best_estimator_

# ── Train Random Forest ──────────────────────────────────────
def train_random_forest(X_train, y_train):
    params = {
        'n_estimators'     : [100, 200, 300],
        'max_depth'        : [4, 6, 8, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4],
        'class_weight'     : ['balanced', None]
    }
    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(rf, params, n_iter=20,
                                cv=5, scoring='f1',
                                random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best RF params: {search.best_params_}")
    return search.best_estimator_

# ── Main ─────────────────────────────────────────────────────
def run_training():
    print_separator("TRAINING PIPELINE")

    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"Data loaded: train={X_train.shape}, test={X_test.shape}")

    # Train both models
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_scores = evaluate(xgb_model, X_test, y_test)

    print("\nTraining Random Forest...")
    rf_model  = train_random_forest(X_train, y_train)
    rf_scores = evaluate(rf_model, X_test, y_test)

    # Compare
    print_separator("RESULTS")
    print(f"XGBoost       F1: {xgb_scores['F1 Score']}%")
    print(f"Random Forest F1: {rf_scores['F1 Score']}%")

    # Save best
    if xgb_scores['F1 Score'] >= rf_scores['F1 Score']:
        save_model(xgb_model)
        print(f"\nWinner: XGBoost — saved to models/best_model.pkl")
        return xgb_model, xgb_scores
    else:
        save_model(rf_model)
        print(f"\nWinner: Random Forest — saved to models/best_model.pkl")
        return rf_model, rf_scores

if __name__ == '__main__':
    run_training()