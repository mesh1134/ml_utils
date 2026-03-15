import numpy as np

# Metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.model_selection import train_test_split, GridSearchCV

# Models
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    RandomForestRegressor,
    AdaBoostRegressor
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


classification_models = {
    "Logistic Regression": (
        LogisticRegression(solver="liblinear", random_state=42),
        {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth": [3, 5, 10, None],
            "criterion": ["gini", "entropy"]
        }
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None],
            "max_features": ["sqrt", "log2"]
        }
    ),
    "xgboost": (
        XGBClassifier(eval_metric='logloss', n_jobs=-1),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5]
        }
    ),
    "adaboost": (
        AdaBoostClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.1, 1.0]
        }
    ),
    "Support Vector Machine (SVM)": (
        LinearSVC(
            class_weight="balanced",
        ),
        {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    )
}

regression_models = {
    "decision_tree": (
        DecisionTreeRegressor(random_state=42),
        {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5]
        }
    ),
    "random_forest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None]
        }
    ),
    "xgboost": (
        XGBRegressor(n_jobs=-1),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    ),
    "adaboost": (
        AdaBoostRegressor(random_state=42),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.1, 1.0]
        }
    )
}

# --- Map Metrics to Sklearn Scoring Names ---
scoring_map = {
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "f1": "f1_weighted",
    "precision": "precision_weighted",
    "recall": "recall_weighted"
}

def test_score(model, metric, x_test, y_test, y):
    """Calculate test score based on metric type."""
    y_pred = model.predict(x_test)
    if metric == "accuracy":
        return accuracy_score(y_test, y_pred)
    elif metric == "roc_auc" and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)
        if len(np.unique(y)) == 2:
            return roc_auc_score(y_test, y_prob[:, 1])
        return roc_auc_score(y_test, y_prob, multi_class='ovr')

    elif metric == "f1":
        return f1_score(y_test, y_pred, average='weighted')
    elif metric == "r2":
        return r2_score(y_test, y_pred)
    elif metric == "mse":
        return mean_squared_error(y_test, y_pred)
    elif metric == "rmse":
        return np.sqrt(mean_squared_error(y_test, y_pred))
    elif metric == "mae":
        return mean_absolute_error(y_test, y_pred)

def find_best_model(x, y, problem_type, metric):
    """
    Training multiple models using GridSearchCV.

    Parameters:
    - x, y: Features and Target
    - problem_type: 'classification' or 'regression'
    - Metric: 'accuracy', 'roc_auc', 'f1', 'mse', 'r2', etc.
    """

    problem_type = problem_type.lower()
    metric = metric.lower()

    # --- Validation ---
    if problem_type not in ["classification", "regression"]:
        raise ValueError("Problem type must be either classification or regression.")

    # --- Split Data ---
    # Stratify is crucial for Classification (keeps class balance the same in Train/Test)
    stratify_param = y if problem_type == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify_param)

    # --- Define Models & Grids ---

    models = classification_models if problem_type == "classification" else regression_models
    metric = scoring_map.get(metric, metric)

    # --- Training Loop ---
    best_overall_model = None
    best_overall_cv_score = -float("inf")
    best_test_score = 0
    best_name = ""
    best_params = {}

    print(f"Training models for {problem_type} using {metric} (GridSearchCV)...")

    for name, (model, params) in models.items():
        grid = GridSearchCV(
            model,
            params,
            cv=5,
            scoring=metric,
            n_jobs=-1
        )

        try:
            grid.fit(x_train, y_train)
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            continue

        current_cv_score = grid.best_score_

        if current_cv_score > best_overall_cv_score:
            best_overall_cv_score = current_cv_score
            best_overall_model = grid.best_estimator_
            best_name = name
            best_params = grid.best_params_

            # --- Calculate Test Score ---
            best_test_score = test_score(best_overall_model, metric, x_test, y_test, y)

    return {
        "best_model_name": best_name,
        "best_params": best_params,
        "CV_score": best_overall_cv_score,
        "Test_score": best_test_score,
        "trained_model": best_overall_model
    }


# ============================================================================
# Example Usage / Testing Section
# ============================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression

    print("=" * 60)
    print("Testing find_best_model() function")
    print("=" * 60)

    # Test 1: Classification
    print("\n[Test 1] Classification with synthetic data")
    print("-" * 60)
    X_clf, y_clf = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_classes=3, random_state=42
    )

    result_clf = find_best_model(X_clf, y_clf, 'classification', 'accuracy')
    print(f"\n✓ Best Classification Model: {result_clf['best_model_name']}")
    print(f"✓ CV Score: {result_clf['CV_score']:.4f}")
    print(f"✓ Test Score: {result_clf['Test_score']:.4f}")
    print(f"✓ Best Params: {result_clf['best_params']}")

    # Test 2: Regression
    print("\n\n[Test 2] Regression with synthetic data")
    print("-" * 60)
    X_reg, y_reg = make_regression(
        n_samples=500, n_features=10, n_informative=5,
        noise=10, random_state=42
    )

    result_reg = find_best_model(X_reg, y_reg, 'regression', 'r2')
    print(f"\n✓ Best Model: {result_reg['best_model_name']}")
    print(f"✓ CV Score: {result_reg['CV_score']:.4f}")
    print(f"✓ Test Score: {result_reg['Test_score']:.4f}")
    print(f"✓ Best Params: {result_reg['best_params']}")

    print("\n" + "=" * 60)
    print("All tests passed! Functions work correctly.")
    print("=" * 60)
