"""
train_model.py — Entraînement de tous les modèles ML
  1. Clustering K-Means (avec ACP)
  2. Classification Churn (Random Forest + Logistic Regression)
  3. Régression MonetaryTotal (Random Forest Regressor)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# Ajouter src/ au path pour importer utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    evaluate_classifier, evaluate_regressor, plot_feature_importance,
    plot_pca_variance, plot_pca_2d, plot_elbow_silhouette
)

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_TEST_DIR = os.path.join(BASE_DIR, "data", "train_test")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def load_splits():
    """Charge X_train, X_test, y_train, y_test depuis data/train_test/"""
    X_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "y_test.csv")).squeeze()
    print(f"✅ Splits chargés — Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"   Churn train : {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 : ACP + CLUSTERING K-MEANS
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   k_range=range(2, 9)):
    """
    1. Réduit la dimension via ACP
    2. Cherche le meilleur k via Elbow + Silhouette
    3. Entraîne K-Means avec le meilleur k
    4. Sauvegarde les artefacts
    """
    print("\n" + "="*60)
    print("  MODULE 1 — ACP + CLUSTERING K-MEANS")
    print("="*60)

    # ── ACP
    pca = PCA(n_components=0.95, random_state=42)   # 95% de variance
    X_pca_train = pca.fit_transform(X_train)
    X_pca_test  = pca.transform(X_test)
    n_comp = pca.n_components_
    print(f"\n  📉 ACP : {X_train.shape[1]} features → {n_comp} composantes (95% variance)")
    plot_pca_variance(pca)

    # Réduction 2D pour visualisation
    pca2d = PCA(n_components=2, random_state=42)
    X_2d  = pca2d.fit_transform(X_train)

    # ── Recherche du meilleur k
    inertias, silhouettes = [], []
    print("\n  🔍 Recherche du meilleur k ...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca_train)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_pca_train, labels, sample_size=1000, random_state=42)
        silhouettes.append(sil)
        print(f"    k={k}  |  Inertie={km.inertia_:.0f}  |  Silhouette={sil:.4f}")

    best_k = plot_elbow_silhouette(list(k_range), inertias, silhouettes)

    # ── Entraînement final
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    train_labels = kmeans.fit_predict(X_pca_train)
    final_sil = silhouette_score(X_pca_train, train_labels)
    print(f"\n  ✅ K-Means final : k={best_k}, Silhouette={final_sil:.4f}")

    # Visualisation 2D
    pca2d_labels = kmeans.predict(pca2d.fit_transform(X_train))
    plot_pca_2d(X_2d, train_labels, title=f"K-Means k={best_k} (ACP 2D)")

    # Profil des clusters
    df_cluster = X_train.copy()
    df_cluster["Cluster"] = train_labels
    print("\n  📊 Profil des clusters (moyennes features clés) :")
    key_features = ["Recency", "Frequency", "MonetaryTotal",
                    "CustomerTenureDays", "SatisfactionScore"]
    key_features = [f for f in key_features if f in df_cluster.columns]
    print(df_cluster.groupby("Cluster")[key_features].mean().round(2).to_string())

    # ── Sauvegarde
    joblib.dump(pca,    os.path.join(MODELS_DIR, "pca.joblib"))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.joblib"))
    print(f"\n  💾 Modèles sauvegardés : pca.joblib, kmeans.joblib")

    return pca, kmeans, X_pca_train, X_pca_test, train_labels


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 : CLASSIFICATION — PRÉDICTION CHURN
# ══════════════════════════════════════════════════════════════════════════════

def run_classification(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series):
    """
    Entraîne :
      - Logistic Regression (baseline)
      - Random Forest (modèle principal)
    Gère le déséquilibre de classes via SMOTE + class_weight.
    """
    print("\n" + "="*60)
    print("  MODULE 2 — CLASSIFICATION CHURN")
    print("="*60)

    # ── Rééquilibrage : SMOTE sur X_train
    print("\n  ⚖️  SMOTE pour rééquilibrage des classes ...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"    Avant SMOTE : {y_train.value_counts().to_dict()}")
    print(f"    Après SMOTE : {pd.Series(y_res).value_counts().to_dict()}")

    # ────────────────────────────────────
    # 2a. Logistic Regression (baseline)
    # ────────────────────────────────────
    print("\n  [Logistic Regression]")
    lr = LogisticRegression(
        C=1.0, max_iter=1000, random_state=42,
        class_weight="balanced", solver="lbfgs"
    )
    lr.fit(X_res, y_res)
    auc_lr = evaluate_classifier(lr, X_test, y_test, model_name="Logistic Regression")
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.joblib"))

    # ────────────────────────────────────
    # 2b. Random Forest (modèle principal)
    # ────────────────────────────────────
    print("\n  [Random Forest Classifier]")

    # Hyperparamètres optimaux (pré-tuned via GridSearch)
    rf_params = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_res, y_res)
    auc_rf = evaluate_classifier(rf, X_test, y_test, model_name="Random Forest Classifier")

    # Importance des features
    plot_feature_importance(rf, list(X_train.columns),
                            top_n=20, title="Random Forest Classifier")

    # Sauvegarde du meilleur modèle
    best_model = rf if (auc_rf or 0) >= (auc_lr or 0) else lr
    best_name  = "Random Forest" if best_model is rf else "Logistic Regression"
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest_classifier.joblib"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_classifier.joblib"))
    print(f"\n  🏆 Meilleur classifieur : {best_name} (AUC={max(auc_rf or 0, auc_lr or 0):.4f})")
    print(f"  💾 Modèles sauvegardés dans models/")

    return rf, lr


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 : RÉGRESSION — ESTIMATION MONÉTAIRE
# ══════════════════════════════════════════════════════════════════════════════

def run_regression(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   raw_train_target: pd.Series = None,
                   raw_test_target: pd.Series  = None):
    """
    Prédit MonetaryTotal à partir des features comportementales.
    La target est chargée depuis data/processed/retail_processed.csv si non fournie.
    """
    print("\n" + "="*60)
    print("  MODULE 3 — RÉGRESSION (MonetaryTotal)")
    print("="*60)

    # ── Chargement de la target de régression
    if raw_train_target is None or raw_test_target is None:
        processed_path = os.path.join(PROCESSED_DIR, "retail_processed.csv")
        if not os.path.exists(processed_path):
            print("  ⚠️  retail_processed.csv introuvable. Lancez d'abord preprocessing.py")
            return None

        df_full = pd.read_csv(processed_path)
        if "MonetaryTotal" not in df_full.columns:
            print("  ⚠️  MonetaryTotal absent du dataset traité.")
            return None

        # On recrée les indices train/test (même random_state=42)
        from sklearn.model_selection import train_test_split
        churn_col = df_full["Churn"] if "Churn" in df_full.columns else None
        y_monetary = df_full["MonetaryTotal"]

        if churn_col is not None:
            _, _, y_reg_train, y_reg_test = train_test_split(
                df_full.drop(columns=["Churn", "MonetaryTotal"], errors="ignore"),
                y_monetary, test_size=0.2, random_state=42, stratify=churn_col
            )
        else:
            _, _, y_reg_train, y_reg_test = train_test_split(
                df_full, y_monetary, test_size=0.2, random_state=42
            )
    else:
        y_reg_train = raw_train_target
        y_reg_test  = raw_test_target

    # On retire MonetaryTotal des features (sinon fuite triviale)
    X_train_reg = X_train.copy()
    X_test_reg  = X_test.copy()
    for drop_col in ["MonetaryTotal", "MonetaryAvg", "MonetaryStd",
                     "MonetaryPerDay", "AvgBasketValue"]:
        if drop_col in X_train_reg.columns:
            X_train_reg = X_train_reg.drop(columns=[drop_col])
            X_test_reg  = X_test_reg.drop(columns=[drop_col])

    print(f"\n  Features régression : {X_train_reg.shape[1]}")

    # ── Random Forest Regressor
    rfr = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_split=5, min_samples_leaf=2,
        max_features="sqrt", random_state=42, n_jobs=-1
    )
    rfr.fit(X_train_reg, y_reg_train)
    rmse, r2 = evaluate_regressor(rfr, X_test_reg, y_reg_test,
                                   model_name="Random Forest Regressor")
    plot_feature_importance(rfr, list(X_train_reg.columns),
                            top_n=15, title="Random Forest Regressor")

    joblib.dump(rfr, os.path.join(MODELS_DIR, "random_forest_regressor.joblib"))
    print(f"  💾 Régresseur sauvegardé : models/random_forest_regressor.joblib")

    return rfr


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    X_train, X_test, y_train, y_test = load_splits()

    # Clustering
    pca, kmeans, X_pca_train, X_pca_test, labels = run_clustering(X_train, X_test)

    # Classification
    rf_clf, lr_clf = run_classification(X_train, X_test, y_train, y_test)

    # Régression
    rf_reg = run_regression(X_train, X_test)

    print("\n" + "="*60)
    print("  TOUS LES MODÈLES ENTRAÎNÉS ✅")
    print("  Fichiers dans models/ :")
    for f in sorted(os.listdir(MODELS_DIR)):
        print(f"    • {f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all()
