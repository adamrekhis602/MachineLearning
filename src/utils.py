"""
utils.py — Fonctions utilitaires pour visualisation et évaluation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay, mean_squared_error, r2_score
)
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# Configuration globale des graphiques
# ─────────────────────────────────────────────
PALETTE = "viridis"
FIG_SIZE = (12, 6)
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def save_fig(name: str):
    """Sauvegarde la figure courante dans reports/"""
    path = os.path.join(REPORTS_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  ✅ Figure sauvegardée : {path}")
    plt.close()


# ─────────────────────────────────────────────
# EDA & Exploration
# ─────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame):
    """Affiche un bar chart des valeurs manquantes."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("  Aucune valeur manquante détectée.")
        return
    pct = (missing / len(df) * 100).round(2)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.bar(missing.index, pct, color=sns.color_palette(PALETTE, len(missing)))
    ax.set_title("Valeurs manquantes (%) par feature", fontsize=14)
    ax.set_ylabel("%")
    ax.set_xticklabels(missing.index, rotation=45, ha="right")
    for bar, val in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_fig("missing_values")


def plot_distributions(df: pd.DataFrame, cols: list, n_cols: int = 4):
    """Histogrammes pour les features numériques."""
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribution des features numériques", fontsize=15, y=1.01)
    plt.tight_layout()
    save_fig("distributions_numeriques")


def plot_correlation_heatmap(df: pd.DataFrame, threshold: float = 0.8):
    """Heatmap de corrélation avec marquage des corrélations élevées."""
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                center=0, linewidths=0.3, ax=ax, vmin=-1, vmax=1)
    ax.set_title(f"Matrice de corrélation (seuil = {threshold})", fontsize=14)
    plt.tight_layout()
    save_fig("correlation_heatmap")

    # Paires fortement corrélées
    high_corr = []
    for col in corr.columns:
        for row in corr.index:
            if col != row and abs(corr.loc[row, col]) >= threshold:
                pair = tuple(sorted([col, row]))
                if pair not in high_corr:
                    high_corr.append(pair)
    if high_corr:
        print(f"\n  ⚠️  {len(high_corr)} paires fortement corrélées (|r| ≥ {threshold}) :")
        for p in high_corr:
            print(f"     • {p[0]}  ↔  {p[1]}  (r={corr.loc[p[0], p[1]]:.2f})")
    return high_corr


def plot_class_balance(series: pd.Series, title: str = "Distribution des classes"):
    """Bar chart de la distribution d'une variable catégorielle."""
    counts = series.value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=sns.color_palette("Set2", len(counts)))
    ax.set_title(title, fontsize=13)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", fontsize=10)
    plt.tight_layout()
    save_fig(f"class_balance_{series.name}")


def plot_boxplots(df: pd.DataFrame, cols: list, n_cols: int = 4):
    """Boxplots pour détecter les outliers."""
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="#4C72B0", alpha=0.7))
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Boxplots — Détection d'outliers", fontsize=14, y=1.01)
    plt.tight_layout()
    save_fig("boxplots_outliers")


# ─────────────────────────────────────────────
# Évaluation Classification
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test, model_name: str = "Model"):
    """Affiche les métriques et courbes pour un classifieur."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n{'='*50}")
    print(f"  Évaluation : {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Fidèle (0)", "Churné (1)"]))

    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    if auc:
        print(f"  AUC-ROC : {auc:.4f}")

    # Matrice de confusion
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
        display_labels=["Fidèle", "Churné"],
        colorbar=False, ax=axes[0], cmap="Blues")
    axes[0].set_title(f"Matrice de confusion — {model_name}")

    # Courbe ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}", color="#4C72B0")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
        axes[1].set_title("Courbe ROC")
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    save_fig(f"eval_classification_{model_name.replace(' ', '_')}")
    return auc


def plot_feature_importance(model, feature_names: list, top_n: int = 20, title: str = ""):
    """Bar chart des importances de features (modèles arborescents)."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in indices[::-1]],
            importances[indices[::-1]],
            color=sns.color_palette(PALETTE, top_n))
    ax.set_title(f"Importance des features — {title}", fontsize=13)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    save_fig(f"feature_importance_{title.replace(' ', '_')}")


# ─────────────────────────────────────────────
# Évaluation Régression
# ─────────────────────────────────────────────

def evaluate_regressor(model, X_test, y_test, model_name: str = "Model"):
    """Métriques et graphiques pour un régresseur."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  Évaluation Régression : {model_name}")
    print(f"  RMSE : {rmse:.2f}  |  R² : {r2:.4f}")
    print(f"{'='*50}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.4, color="#4C72B0", edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set_xlabel("Valeurs réelles"); axes[0].set_ylabel("Prédictions")
    axes[0].set_title(f"Réel vs Prédit — {model_name}")

    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=40, color="#4C72B0", edgecolor="white")
    axes[1].set_title("Distribution des résidus")
    axes[1].set_xlabel("Résidu")

    plt.tight_layout()
    save_fig(f"eval_regression_{model_name.replace(' ', '_')}")
    return rmse, r2


# ─────────────────────────────────────────────
# ACP & Clustering
# ─────────────────────────────────────────────

def plot_pca_variance(pca: PCA):
    """Courbe de variance expliquée cumulée."""
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker="o", color="#4C72B0")
    ax.axhline(90, color="red", linestyle="--", label="90%")
    ax.axhline(95, color="orange", linestyle="--", label="95%")
    ax.set_xlabel("Nombre de composantes")
    ax.set_ylabel("Variance expliquée cumulée (%)")
    ax.set_title("ACP — Variance expliquée cumulée")
    ax.legend()
    plt.tight_layout()
    save_fig("pca_variance_explained")
    n90 = int(np.searchsorted(cumvar, 90)) + 1
    n95 = int(np.searchsorted(cumvar, 95)) + 1
    print(f"  → {n90} composantes pour 90% de variance")
    print(f"  → {n95} composantes pour 95% de variance")


def plot_pca_2d(X_pca: np.ndarray, labels, title: str = "Clustering ACP 2D"):
    """Scatter 2D coloré par labels."""
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=labels, cmap="tab10", alpha=0.6, s=15, edgecolors="none")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    save_fig(f"pca_2d_{title.replace(' ', '_')}")


def plot_elbow_silhouette(k_range, inertias: list, silhouettes: list):
    """Courbes elbow + silhouette pour choisir k."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(k_range, inertias, marker="o", color="#4C72B0")
    axes[0].set_title("Méthode Elbow (Inertie)")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertie")

    axes[1].plot(k_range, silhouettes, marker="s", color="#DD8452")
    axes[1].set_title("Score Silhouette")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")

    plt.tight_layout()
    save_fig("elbow_silhouette")
    best_k = k_range[int(np.argmax(silhouettes))]
    print(f"  → Meilleur k selon Silhouette : k={best_k} (score={max(silhouettes):.4f})")
    return best_k
