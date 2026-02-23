from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.utils import (
    ensure_dirs, RAW_DIR, PROCESSED_DIR, REPORTS_DIR,
    load_csv, save_csv, save_json, info, warn
)

RAW_FILENAME = "customers_raw.csv"          # raw file inside data/raw/
OUTPUT_FILENAME = "customers_clean.csv"     # output inside data/processed/
TARGET_COL = "Churn"

# Columns mentioned in the brief (if they exist in your dataset, we process them)
DATE_COL_CANDIDATES = ["RegistrationDate", "RegistDate"]
DROP_ALWAYS_IF_EXISTS = ["NewsletterSubscribed", "Newsletter"]


@dataclass
class PreprocessReport:
    rows_before: int
    cols_before: int
    rows_after: int
    cols_after: int
    dropped_columns: list
    parsed_date_column: str | None
    missing_before: Dict[str, int]
    missing_after: Dict[str, int]
    outliers_fixed: Dict[str, int]
    engineered_features: list


def pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_registration_date(df: pd.DataFrame, report_outliers: Dict[str, int]) -> Tuple[pd.DataFrame, str | None]:
    df = df.copy()
    date_col = pick_existing_column(df, DATE_COL_CANDIDATES)
    if not date_col:
        return df, None

    # Parse with dayfirst=True to handle UK format; invalid => NaT
    parsed = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df["RegistrationDate"] = parsed  # normalize name

    # Feature extraction
    df["RegYear"] = df["RegistrationDate"].dt.year
    df["RegMonth"] = df["RegistrationDate"].dt.month
    df["RegDay"] = df["RegistrationDate"].dt.day
    df["RegWeekday"] = df["RegistrationDate"].dt.weekday

    # If original was different name, drop it
    if date_col != "RegistrationDate":
        df = df.drop(columns=[date_col])

    return df, "RegistrationDate"


def fix_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Fix known outlier patterns from the brief:
    - SupportTickets might contain -1 (invalid) => set to NaN
    - Satisfaction might contain -1 or 0 => set to NaN
    (Then imputation will handle NaN)
    """
    df = df.copy()
    fixed = {}

    if "SupportTickets" in df.columns:
        bad = df["SupportTickets"].isin([-1])
        fixed["SupportTickets"] = int(bad.sum())
        df.loc[bad, "SupportTickets"] = np.nan

    if "Satisfaction" in df.columns:
        bad = df["Satisfaction"].isin([-1, 0])
        fixed["Satisfaction"] = int(bad.sum())
        df.loc[bad, "Satisfaction"] = np.nan

    return df, fixed


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, safe imputation:
    - numeric: median
    - categorical: mode (most frequent)
    """
    df = df.copy()

    # numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if c == TARGET_COL:
            continue
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # categorical columns
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for c in cat_cols:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            df[c] = df[c].fillna(fill)

    return df


def feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Adds useful features if source columns exist.
    """
    df = df.copy()
    created = []

    if "MonetaryTotal" in df.columns and "Recency" in df.columns:
        df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)
        created.append("MonetaryPerDay")

    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        # avoid division by zero
        denom = df["Frequency"].replace(0, np.nan)
        df["AvgBasketValue"] = df["MonetaryTotal"] / denom
        df["AvgBasketValue"] = df["AvgBasketValue"].fillna(df["AvgBasketValue"].median())
        created.append("AvgBasketValue")

    # Example: flag private IP (simple heuristic)
    if "LastLoginIP" in df.columns:
        ip = df["LastLoginIP"].astype(str)
        df["IsPrivateIP"] = ip.str.startswith(("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", "172.2"))
        created.append("IsPrivateIP")

    return df, created


def main() -> None:
    ensure_dirs()

    raw_path = RAW_DIR / RAW_FILENAME
    if not raw_path.exists():
        warn(f"Raw file not found: {raw_path}")
        warn("➡️ Put your raw CSV in data/raw/ and set RAW_FILENAME accordingly.")
        return

    info(f"Loading raw dataset: {raw_path}")
    df = load_csv(raw_path)

    rows_before, cols_before = df.shape
    missing_before = df.isna().sum().to_dict()

    # Drop useless/constant columns if present
    dropped = [c for c in DROP_ALWAYS_IF_EXISTS if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
        info(f"Dropped columns: {dropped}")

    # Fix known outliers -> NaN
    df, outliers_fixed = fix_outliers(df)

    # Parse date + extract date features
    df, parsed_date = parse_registration_date(df, outliers_fixed)

    # Feature engineering
    df, engineered = feature_engineering(df)

    # Impute missing
    df = impute_missing(df)

    rows_after, cols_after = df.shape
    missing_after = df.isna().sum().to_dict()

    # Save outputs
    out_csv = PROCESSED_DIR / OUTPUT_FILENAME
    save_csv(df, out_csv)

    report = PreprocessReport(
        rows_before=rows_before,
        cols_before=cols_before,
        rows_after=rows_after,
        cols_after=cols_after,
        dropped_columns=dropped,
        parsed_date_column=parsed_date,
        missing_before={k: int(v) for k, v in missing_before.items() if int(v) > 0},
        missing_after={k: int(v) for k, v in missing_after.items() if int(v) > 0},
        outliers_fixed=outliers_fixed,
        engineered_features=engineered,
    )

    out_report = REPORTS_DIR / "preprocessing_report.json"
    save_json(report.__dict__, out_report)

    info(f"Saved processed dataset: {out_csv}")
    info(f"Saved report: {out_report}")
    info(f"Done. Shape: {df.shape}")


if __name__ == "__main__":
    main()