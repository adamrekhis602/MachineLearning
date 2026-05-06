"""
predict.py — Prédiction sur un ou plusieurs nouveaux clients
Usage :
  python src/predict.py                        → exemple interne
  python src/predict.py --file path/to/new.csv → batch depuis CSV
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Chargement des artefacts ───────────────────────────────────────────────────

def load_artifacts():
    """Charge tous les modèles et le scaler."""
    artefacts = {}
    files = {
        "scaler":     "scaler.joblib",
        "pca":        "pca.joblib",
        "kmeans":     "kmeans.joblib",
        "classifier": "best_classifier.joblib",
        "regressor":  "random_forest_regressor.joblib",
    }
    missing = []
    for name, fname in files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            artefacts[name] = joblib.load(path)
            print(f"  ✅ {fname} chargé")
        else:
            missing.append(fname)
            print(f"  ⚠️  {fname} introuvable — entraînez d'abord train_model.py")
    return artefacts, missing


# ── Prédiction sur un DataFrame pré-processé ──────────────────────────────────

def predict_all(df_raw: pd.DataFrame, artefacts: dict) -> pd.DataFrame:
    """
    Applique le scaler puis prédit :
      - Cluster K-Means
      - Probabilité de churn + label binaire
      - Montant monétaire estimé
    Retourne un DataFrame résultat.
    """
    results = pd.DataFrame()

    scaler = artefacts.get("scaler")
    if scaler is None:
        raise RuntimeError("Scaler manquant. Lancez preprocessing.py puis train_model.py.")

    # Alignement des colonnes avec celles vues à l'entraînement
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in df_raw.columns:
            df_raw[col] = 0  # colonne absente → valeur neutre
    df_aligned = df_raw[expected_cols]

    X_scaled = scaler.transform(df_aligned)
    X_scaled_df = pd.DataFrame(X_scaled, columns=expected_cols)

    # ── Clustering
    if "pca" in artefacts and "kmeans" in artefacts:
        X_pca    = artefacts["pca"].transform(X_scaled_df)
        clusters = artefacts["kmeans"].predict(X_pca)
        results["Cluster"] = clusters

    # ── Churn Classification
    if "classifier" in artefacts:
        clf = artefacts["classifier"]
        results["ChurnProbability"] = clf.predict_proba(X_scaled_df)[:, 1]
        results["ChurnPrediction"]  = clf.predict(X_scaled_df)
        results["ChurnLabel"]       = results["ChurnPrediction"].map({0: "Fidèle", 1: "Churné"})

    # ── Régression Monétaire
    if "regressor" in artefacts:
        reg = artefacts["regressor"]
        # Supprime les colonnes monétaires (comme à l'entraînement)
        monetary_drop = ["MonetaryTotal", "MonetaryAvg", "MonetaryStd",
                         "MonetaryPerDay", "AvgBasketValue"]
        X_reg = X_scaled_df.drop(columns=[c for c in monetary_drop if c in X_scaled_df.columns])
        # Aligner sur les features du régresseur
        try:
            results["PredictedMonetary"] = reg.predict(X_reg)
        except Exception as e:
            print(f"  ⚠️  Régression : {e}")

    return results


# ── Exemple : prédiction sur un client unique ─────────────────────────────────

def predict_single_client(client_dict: dict, artefacts: dict) -> dict:
    """Prédit pour un client passé en dictionnaire."""
    df_client = pd.DataFrame([client_dict])
    result_df = predict_all(df_client, artefacts)
    return result_df.iloc[0].to_dict()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prédictions Retail ML")
    parser.add_argument("--file", type=str, default=None,
                        help="Chemin vers un CSV de nouveaux clients")
    parser.add_argument("--out", type=str,
                        default=os.path.join(BASE_DIR, "reports", "predictions.csv"),
                        help="Chemin de sortie des prédictions")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  PRÉDICTIONS — DÉMARRAGE")
    print("="*60)

    artefacts, missing = load_artifacts()
    if len(missing) == len(artefacts) + len(missing):
        sys.exit("  ❌ Aucun artefact trouvé. Lancez d'abord preprocessing.py puis train_model.py.")

    if args.file:
        # ── Mode batch : CSV fourni
        if not os.path.exists(args.file):
            sys.exit(f"  ❌ Fichier introuvable : {args.file}")
        df_new = pd.read_csv(args.file)
        print(f"\n  📂 Fichier chargé : {df_new.shape[0]} clients")
        results = predict_all(df_new, artefacts)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        results.to_csv(args.out, index=False)
        print(f"\n  💾 Prédictions sauvegardées : {args.out}")
        print(results.head(10).to_string())
    else:
        # ── Mode démo : client exemple
        example_client = {
            "Recency": 45,
            "Frequency": 8,
            "MonetaryTotal": 1200.0,
            "MonetaryAvg": 150.0,
            "MonetaryStd": 40.0,
            "CustomerTenureDays": 300,
            "TotalQuantity": 90,
            "UniqueProducts": 15,
            "SatisfactionScore": 3.5,
            "SupportTicketsCount": 1,
            "Age": 35,
            "WeekendPurchaseRatio": 0.3,
            "ReturnRatio": 0.05,
            "CancelledTransactions": 0,
            "TotalTransactions": 50,
        }
        print("\n  📋 Client exemple :")
        for k, v in example_client.items():
            print(f"    {k}: {v}")

        prediction = predict_single_client(example_client, artefacts)
        print("\n  📊 Résultats de la prédiction :")
        for k, v in prediction.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    print("\n" + "="*60)
    print("  PRÉDICTIONS — TERMINÉES ✅")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
