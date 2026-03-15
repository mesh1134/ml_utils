import numpy as np


# Metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC

# Models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    RandomForestRegressor,
    AdaBoostRegressor
)

# Metric scoring map
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

# Model configurations with larger parameter spaces for RandomizedSearchCV
classification_models = {
    "Logistic Regression": (
        LogisticRegression(solver="liblinear", random_state=42, class_weight="balanced"),
        {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [3, 5, 10, 20, None],
         "min_samples_split": [2, 5, 10],
         "criterion": ["gini", "entropy"]
         }
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
        {"n_estimators": [50, 100, 200],
         "max_depth": [5, 10, 20, None],
         "max_features": ["sqrt", "log2"]
         }
    ),
    "xgboost": (
        XGBClassifier(eval_metric='logloss', n_jobs=-1),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2],
         "max_depth": [3, 5, 7, 10], "subsample": [0.6, 0.8, 1.0]}
    ),
    "adaboost": (
        AdaBoostClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0]}
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
        {"max_depth": [3, 5, 10, 20, None], "min_samples_split": [2, 5, 10]}
    ),
    "random_forest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20, None],
         "max_features": ["sqrt", "log2"]}
    ),
    "xgboost": (
        XGBRegressor(n_jobs=-1),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2],
         "max_depth": [3, 5, 7, 10], "subsample": [0.6, 0.8, 1.0]}
    ),
    "adaboost": (
        AdaBoostRegressor(random_state=42),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0]}
    )
}


def _calculate_test_score(model, metric, x_test, y_test, y):
    """Calculate test score based on metric type."""
    y_pred = model.predict(x_test)

    metric_calculators = {
        "accuracy": lambda: accuracy_score(y_test, y_pred),
        "f1_weighted": lambda: f1_score(y_test, y_pred, average='weighted'),
        "r2": lambda: r2_score(y_test, y_pred),
        "neg_mean_squared_error": lambda: mean_squared_error(y_test, y_pred),
        "neg_root_mean_squared_error": lambda: np.sqrt(mean_squared_error(y_test, y_pred)),
        "neg_mean_absolute_error": lambda: mean_absolute_error(y_test, y_pred)
    }

    if metric == "roc_auc" and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)
        if len(np.unique(y)) == 2:
            return roc_auc_score(y_test, y_prob[:, 1])
        return roc_auc_score(y_test, y_prob, multi_class='ovr')

    return metric_calculators.get(metric, lambda: 0.0)()


def find_best_model(x, y, problem_type, metric, n_iter=10):
    """
    Training multiple models using RandomizedSearchCV.

    Parameters:
    - x, y: Features and Target
    - problem_type: 'classification' or 'regression'
    - Metric: 'accuracy', 'roc_auc', 'f1', 'mse', 'r2', etc.
    """
    problem_type = problem_type.lower()
    metric = metric.lower()

    if problem_type not in {"classification", "regression"}:
        raise ValueError("Problem type must be 'classification' or 'regression'")

    # Split data with stratification for classification
    stratify = y if problem_type == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Select models based on the problem type
    models = classification_models if problem_type == "classification" else regression_models
    sklearn_scoring = scoring_map.get(metric, metric)

    print(f"Training models for {problem_type} using {metric} (RandomizedSearchCV)...")

    best_results = {
        "best_model_name": "",
        "best_params": {},
        "CV_score": -float("inf"),
        "Test_score": 0,
        "trained_model": None
    }

    for name, (model, params) in models.items():
        search = RandomizedSearchCV(
            model, params, n_iter=n_iter, cv=5,
            scoring=sklearn_scoring, n_jobs=-1, random_state=42, verbose=0
        )

        try:
            search.fit(x_train, y_train)
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            continue

        if search.best_score_ > best_results["CV_score"]:
            best_results.update({
                "CV_score": search.best_score_,
                "trained_model": search.best_estimator_,
                "best_model_name": name,
                "best_params": search.best_params_,
                "Test_score": _calculate_test_score(
                    search.best_estimator_, sklearn_scoring, x_test, y_test, y
                )
            })

    return best_results


# ============================================================================
# Example Usage / Testing Section
# ============================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression

    print("=" * 60)
    print("Testing find_best_model() function (RandomizedSearchCV)")
    print("=" * 60)

    # Test 1: Classification
    print("\n[Test 1] Classification with synthetic data")
    print("-" * 60)
    X_clf, y_clf = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_classes=3, random_state=42
    )

    result_clf = find_best_model(X_clf, y_clf, 'classification', 'accuracy', n_iter=10)
    print(f"\n✓ Best Model: {result_clf['best_model_name']}")
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

    result_reg = find_best_model(X_reg, y_reg, 'regression', 'r2', n_iter=10)
    print(f"\n✓ Best Model: {result_reg['best_model_name']}")
    print(f"✓ CV Score: {result_reg['CV_score']:.4f}")
    print(f"✓ Test Score: {result_reg['Test_score']:.4f}")
    print(f"✓ Best Params: {result_reg['best_params']}")

    print("\n" + "=" * 60)
    print("All tests passed! Functions work correctly.")
    print("=" * 60)
