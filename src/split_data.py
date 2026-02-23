from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.utils import (
    ensure_dirs, PROCESSED_DIR, TRAIN_TEST_DIR, MODELS_DIR, REPORTS_DIR,
    load_csv, save_csv, save_json, info, warn
)

CLEAN_DATA = PROCESSED_DIR / "customers_clean.csv"
TARGET_COL = "Churn"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def infer_column_types(df: pd.DataFrame, target: str) -> tuple[List[str], List[str]]:
    """
    Decide which columns are numeric vs categorical.
    - numeric: int/float/bool
    - categorical: object/category
    """
    X = df.drop(columns=[target])

    # Treat bool as numeric (0/1). If your dataset has True/False, this works well.
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    - Numeric: StandardScaler
    - Categorical: OneHotEncoder(handle_unknown='ignore')
    """
    numeric_pipe = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """
    After fitting, build feature names:
    - numeric columns keep same names
    - categorical columns expand via onehot encoder categories
    """
    feature_names = []

    # numeric
    feature_names.extend(numeric_cols)

    # categorical
    if len(categorical_cols) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(ohe_names)

    return feature_names


def main() -> None:
    ensure_dirs()

    if not CLEAN_DATA.exists():
        warn(f"Missing clean dataset: {CLEAN_DATA}")
        warn("➡️ Run first: python -m src.preprocessing")
        return

    info(f"Loading clean dataset: {CLEAN_DATA}")
    df = load_csv(CLEAN_DATA)

    if TARGET_COL not in df.columns:
        warn(f"Target column '{TARGET_COL}' not found in dataset.")
        warn(f"Available columns example: {list(df.columns)[:15]} ...")
        return

    # Split X/y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Stratify if binary / classification
    stratify = y if y.nunique() <= 10 else None

    info(f"Splitting train/test (test_size={TEST_SIZE}, random_state={RANDOM_STATE})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    info(f"Train shape raw: {X_train.shape} | Test shape raw: {X_test.shape}")

    # Infer types and build preprocessor
    numeric_cols, categorical_cols = infer_column_types(df, TARGET_COL)
    info(f"Numeric cols: {len(numeric_cols)} | Categorical cols: {len(categorical_cols)}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Fit ONLY on train
    info("Fitting preprocessor on TRAIN only (no leakage)")
    preprocessor.fit(X_train)

    # Transform both
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Build feature names (after fitting)
    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)

    # Convert to DataFrame for saving
    X_train_df = pd.DataFrame(X_train_t, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_t, columns=feature_names)

    # Save CSVs
    save_csv(X_train_df, TRAIN_TEST_DIR / "X_train.csv", index=False)
    save_csv(X_test_df, TRAIN_TEST_DIR / "X_test.csv", index=False)
    save_csv(pd.DataFrame({TARGET_COL: y_train.values}), TRAIN_TEST_DIR / "y_train.csv", index=False)
    save_csv(pd.DataFrame({TARGET_COL: y_test.values}), TRAIN_TEST_DIR / "y_test.csv", index=False)

    # Save preprocessor for later (training + prediction)
    dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    # Report
    report: Dict[str, Any] = {
        "clean_data": str(CLEAN_DATA),
        "target": TARGET_COL,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_features_before": int(X_train.shape[1]),
        "test_features_before": int(X_test.shape[1]),
        "train_features_after": int(X_train_df.shape[1]),
        "test_features_after": int(X_test_df.shape[1]),
        "numeric_cols_count": int(len(numeric_cols)),
        "categorical_cols_count": int(len(categorical_cols)),
        "y_train_distribution": y_train.value_counts(dropna=False).to_dict(),
        "y_test_distribution": y_test.value_counts(dropna=False).to_dict(),
    }
    save_json(report, REPORTS_DIR / "split_report.json")

    info("Saved:")
    info(f"- {TRAIN_TEST_DIR / 'X_train.csv'}")
    info(f"- {TRAIN_TEST_DIR / 'X_test.csv'}")
    info(f"- {TRAIN_TEST_DIR / 'y_train.csv'}")
    info(f"- {TRAIN_TEST_DIR / 'y_test.csv'}")
    info(f"- {MODELS_DIR / 'preprocessor.joblib'}")
    info(f"- {REPORTS_DIR / 'split_report.json'}")

    info(f"Final train matrix: {X_train_df.shape} | Final test matrix: {X_test_df.shape}")
    info("Step 2 done ✅")


if __name__ == "__main__":
    main()