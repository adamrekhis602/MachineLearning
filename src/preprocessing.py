"""
preprocessing.py — Nettoyage, encodage, normalisation et split du dataset
Pipeline complet : raw → processed → train_test
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
)
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings("ignore")

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH       = os.path.join(BASE_DIR, "data", "raw", "retail_customers.csv")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
TRAIN_TEST_DIR = os.path.join(BASE_DIR, "data", "train_test")
MODELS_DIR     = os.path.join(BASE_DIR, "models")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TRAIN_TEST_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Charge le CSV brut."""
    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. SUPPRESSION DES FEATURES INUTILES
# ══════════════════════════════════════════════════════════════════════════════

COLS_TO_DROP = [
    "CustomerID",          # identifiant unique, pas de valeur prédictive
    "NewsletterSubscribed",# constante (toujours "Yes")
    "LastLoginIP",         # adresse IP brute — feature engineering fait séparément
    "RegistrationDate",    # parsing fait ci-dessous → colonnes extraites
]

def drop_useless(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes inutiles ou à traiter autrement."""
    existing = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing)
    print(f"  🗑️  Colonnes supprimées : {existing}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def feature_engineering(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Crée de nouvelles features à partir des données brutes."""

    # --- Parsing RegistrationDate ---
    if "RegistrationDate" in raw_df.columns:
        reg = pd.to_datetime(raw_df["RegistrationDate"], dayfirst=True, errors="coerce")
        df["RegYear"]    = reg.dt.year.fillna(reg.dt.year.median()).astype(int)
        df["RegMonth"]   = reg.dt.month.fillna(6).astype(int)
        df["RegWeekday"] = reg.dt.weekday.fillna(0).astype(int)
        print("  📅 RegistrationDate parsée → RegYear, RegMonth, RegWeekday")

    # --- IP Engineering ---
    if "LastLoginIP" in raw_df.columns:
        def is_private_ip(ip):
            if not isinstance(ip, str):
                return 0
            parts = ip.split(".")
            if len(parts) != 4:
                return 0
            try:
                first = int(parts[0])
                second = int(parts[1])
                return int(first == 10 or
                           (first == 172 and 16 <= second <= 31) or
                           (first == 192 and second == 168))
            except Exception:
                return 0
        df["IsPrivateIP"] = raw_df["LastLoginIP"].apply(is_private_ip)
        print("  🌐 LastLoginIP → IsPrivateIP (0/1)")

    # --- Ratios métier ---
    if "MonetaryTotal" in df.columns and "Recency" in df.columns:
        df["MonetaryPerDay"]  = df["MonetaryTotal"] / (df["Recency"] + 1)
    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"]  = df["MonetaryTotal"] / (df["Frequency"] + 1)
    if "Recency" in df.columns and "CustomerTenureDays" in df.columns:
        df["TenureRatio"]     = df["Recency"] / (df["CustomerTenureDays"] + 1)
    if "CancelledTransactions" in df.columns and "TotalTransactions" in df.columns:
        df["CancelRate"]      = df["CancelledTransactions"] / (df["TotalTransactions"] + 1)

    print("  🔧 Features métier créées : MonetaryPerDay, AvgBasketValue, TenureRatio, CancelRate")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAITEMENT DES VALEURS ABERRANTES
# ══════════════════════════════════════════════════════════════════════════════

def fix_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """SupportTicketsCount : -1 et 999 sont des codes erreurs → NaN."""
    if "SupportTicketsCount" in df.columns:
        df["SupportTicketsCount"] = df["SupportTicketsCount"].replace({-1: np.nan, 999: np.nan})
    return df

def fix_satisfaction(df: pd.DataFrame) -> pd.DataFrame:
    """SatisfactionScore : -1 et 99 sont des codes erreurs → NaN. Valides : 0-5."""
    if "SatisfactionScore" in df.columns:
        col = df["SatisfactionScore"]
        df["SatisfactionScore"] = col.where(col.between(0, 5), other=np.nan)
    return df

def cap_outliers_iqr(df: pd.DataFrame, cols: list, factor: float = 3.0) -> pd.DataFrame:
    """Clipping IQR (winsorisation) sur les colonnes numériques spécifiées."""
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        if before > 0:
            print(f"     ✂️  {col} : {before} outliers cappés [{lower:.1f}, {upper:.1f}]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. IMPUTATION DES VALEURS MANQUANTES
# ══════════════════════════════════════════════════════════════════════════════

def impute_numeric(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """Imputation médiane pour les numériques."""
    imp = SimpleImputer(strategy="median")
    df[num_cols] = imp.fit_transform(df[num_cols])
    return df

def impute_age_knn(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """KNN Imputer spécifique pour Age (30% manquant)."""
    if "Age" not in df.columns:
        return df
    ref_cols = ["Age", "Recency", "Frequency", "MonetaryTotal"]
    ref_cols = [c for c in ref_cols if c in df.columns]
    imp = KNNImputer(n_neighbors=k)
    df[ref_cols] = imp.fit_transform(df[ref_cols])
    df["Age"] = df["Age"].clip(18, 81)
    print("  🔢 Age imputé via KNN (k=5)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. ENCODAGE DES FEATURES CATÉGORIELLES
# ══════════════════════════════════════════════════════════════════════════════

# Mappings ordinaux (ordre croissant de valeur)
ORDINAL_MAPS = {
    "RFMSegment":        ["Dormants", "Potentiels", "Fidèles", "Champions"],
    "AgeCategory":       ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Inconnu"],
    "SpendingCategory":  ["Low", "Medium", "High", "VIP"],
    "LoyaltyLevel":      ["Inconnu", "Nouveau", "Jeune", "Établi", "Ancien"],
    "ChurnRiskCategory": ["Faible", "Moyen", "Élevé", "Critique"],
    "BasketSizeCategory":["Inconnu", "Petit", "Moyen", "Grand"],
    "PreferredTimeOfDay":["Nuit", "Matin", "Midi", "Après-midi", "Soir"],
}

ONE_HOT_COLS = [
    "CustomerType", "FavoriteSeason", "Region",
    "WeekendPreference", "ProductDiversity", "Gender",
    "AccountStatus",
]

def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les catégorielles ordinales avec mapping explicite."""
    for col, order in ORDINAL_MAPS.items():
        if col not in df.columns:
            continue
        mapping = {v: i for i, v in enumerate(order)}
        df[col] = df[col].map(mapping).fillna(0).astype(int)
        print(f"  🔢 Ordinal : {col}")
    return df

def encode_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """One-Hot Encoding pour les catégorielles nominales."""
    cols_present = [c for c in ONE_HOT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=False, dtype=int)
    print(f"  🔠 One-Hot : {cols_present} → {df.shape[1]} colonnes totales")
    return df

def encode_country(df: pd.DataFrame, raw_df: pd.DataFrame,
                   target_col: str = "Churn") -> pd.DataFrame:
    """Target Encoding pour la variable Country (haute cardinalité)."""
    if "Country" not in raw_df.columns or target_col not in raw_df.columns:
        return df
    country_mean = raw_df.groupby("Country")[target_col].mean()
    df["Country_TargetEnc"] = raw_df["Country"].map(country_mean).fillna(
        country_mean.mean()
    )
    print("  🌍 Country → Target Encoding (Country_TargetEnc)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUPPRESSION MULTICOLINÉARITÉ
# ══════════════════════════════════════════════════════════════════════════════

HIGH_CORR_DROPS = [
    # Monetary : MonetaryTotal est plus synthétique
    "MonetaryMin", "MonetaryMax",
    # Quantity : TotalQuantity synthétise tout
    "MinQuantity", "MaxQuantity", "AvgQuantityPerTransaction",
    # Invoices/Trans : TotalTransactions suffit
    "UniqueInvoices",
    # UniqueDesc très corrélé à UniqueProducts
    "UniqueDescriptions",
    # FirstPurchaseDaysAgo très proche de CustomerTenureDays
    "FirstPurchaseDaysAgo",
]

def drop_multicollinear(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in HIGH_CORR_DROPS if c in df.columns]
    df = df.drop(columns=existing)
    print(f"  ❌ Colonnes multicolinéaires supprimées : {existing}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing(save: bool = True):
    print("\n" + "="*60)
    print("  PREPROCESSING — DÉBUT")
    print("="*60)

    # ── Chargement
    raw_df = load_data(RAW_PATH)
    df     = raw_df.copy()

    # ── Séparation target
    TARGET_CHURN = "Churn"
    y = df[TARGET_CHURN].copy()

    # ── Feature Engineering (avant suppression)
    df = feature_engineering(df, raw_df)

    # ── Suppression colonnes inutiles
    df = drop_useless(df)
    if TARGET_CHURN in df.columns:
        df = df.drop(columns=[TARGET_CHURN])  # on gère y séparément

    # ── Correction valeurs aberrantes
    print("\n  [Correction outliers]")
    df = fix_support_tickets(df)
    df = fix_satisfaction(df)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df = cap_outliers_iqr(df, num_cols, factor=3.0)

    # ── Imputation
    print("\n  [Imputation]")
    df = impute_age_knn(df)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df = impute_numeric(df, num_cols)

    # ── Encodage target country (avant drop)
    df = encode_country(df, raw_df, target_col=TARGET_CHURN)

    # ── Suppression multicolinéarité
    print("\n  [Multicolinéarité]")
    df = drop_multicollinear(df)

    # ── Encodage catégorielles
    print("\n  [Encodage]")
    df = encode_ordinal(df)
    df = encode_one_hot(df)

    # ── Suppression colonnes objet restantes (ex: Country texte)
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        df = df.drop(columns=obj_cols)
        print(f"  🗑️  Colonnes objet restantes supprimées : {obj_cols}")

    print(f"\n  📐 Shape finale avant split : {df.shape}")
    print(f"  🎯 Target Churn — distribution :\n{y.value_counts().to_string()}")

    # ── Sauvegarde données nettoyées
    df_full = df.copy()
    df_full[TARGET_CHURN] = y.values
    if save:
        out_path = os.path.join(PROCESSED_DIR, "retail_processed.csv")
        df_full.to_csv(out_path, index=False)
        print(f"\n  💾 Données nettoyées sauvegardées : {out_path}")

    # ── Split Train / Test (80/20, stratifié sur Churn)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Normalisation (fit sur train uniquement — éviter data leakage)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )

    if save:
        X_train_scaled.to_csv(os.path.join(TRAIN_TEST_DIR, "X_train.csv"), index=False)
        X_test_scaled.to_csv(os.path.join(TRAIN_TEST_DIR, "X_test.csv"),  index=False)
        y_train.to_csv(os.path.join(TRAIN_TEST_DIR, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(TRAIN_TEST_DIR,  "y_test.csv"),  index=False)
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
        print(f"  💾 Scaler sauvegardé : models/scaler.joblib")
        print(f"  💾 Splits sauvegardés dans data/train_test/")

    print(f"\n  Train : {X_train_scaled.shape} | Test : {X_test_scaled.shape}")
    print("\n" + "="*60)
    print("  PREPROCESSING — TERMINÉ ✅")
    print("="*60 + "\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_preprocessing(save=True)
