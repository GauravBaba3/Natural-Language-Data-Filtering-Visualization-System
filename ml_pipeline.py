from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class MLTrainingResult:
    best_estimator: Any
    best_score: float
    best_params: Dict[str, Any]


def _split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.isna().any():
        # For production: you may want imputation, but keeping it strict for now.
        y = y.dropna()
        X = X.loc[y.index]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def train_classifier(
    df: pd.DataFrame,
    target_col: str,
    model_type: Literal["logreg", "random_forest"] = "random_forest",
    search_type: Literal["randomized", "grid"] = "randomized",
    cv: int = 3,
    n_iter: int = 20,
    random_state: int = 42,
) -> MLTrainingResult:
    """
    Optional supervised ML pipeline training.
    - Auto-detect numeric vs categorical columns.
    - StandardScaler for numeric, OneHotEncoder for categorical.
    - LogisticRegression or RandomForestClassifier.
    - Hyperparameter tuning via GridSearchCV or RandomizedSearchCV.
    """
    X, y = _split_features_target(df, target_col)

    preprocessor, _, _ = build_preprocessor(X)

    if model_type == "logreg":
        clf = LogisticRegression(max_iter=2000, n_jobs=None)
        # Learning-rate is not applicable here; tune regularization strength instead.
        param_space: Dict[str, Any] = {
            "clf__C": np.logspace(-3, 2, 8),
        }
    elif model_type == "random_forest":
        clf = RandomForestClassifier(random_state=random_state)
        param_space = {
            # Required by prompt:
            "clf__n_estimators": [100, 250, 500],
            "clf__max_depth": [None, 5, 10, 20],
            # learning_rate not applicable for RandomForest.
        }
    else:
        raise ValueError("Unsupported model_type")

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

    if search_type == "grid":
        searcher = GridSearchCV(
            pipe,
            param_grid=param_space,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
        )
    else:
        searcher = RandomizedSearchCV(
            pipe,
            param_distributions=param_space,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            n_iter=n_iter,
            random_state=random_state,
        )

    searcher.fit(X, y)
    best_estimator = searcher.best_estimator_
    return MLTrainingResult(
        best_estimator=best_estimator,
        best_score=float(searcher.best_score_),
        best_params=dict(searcher.best_params_),
    )

