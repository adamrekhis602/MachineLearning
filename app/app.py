"""
app.py — Application Flask pour prédictions ML en temps réel
Lancer : python app/app.py
Accès   : http://127.0.0.1:5000
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

# ── Ajout du répertoire src/ au path ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)

# ── Chargement des modèles (au démarrage) ─────────────────────────────────────

def load_model(name: str):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    print(f"  ⚠️  Modèle introuvable : {name}")
    return None

SCALER     = load_model("scaler.joblib")
PCA        = load_model("pca.joblib")
KMEANS     = load_model("kmeans.joblib")
CLASSIFIER = load_model("best_classifier.joblib")
REGRESSOR  = load_model("random_forest_regressor.joblib")

# ── Champs du formulaire avec leurs valeurs par défaut ────────────────────────

FORM_FIELDS = [
    {"name": "Recency",               "label": "Jours depuis dernier achat",  "default": 60,   "min": 0,   "max": 400,   "step": 1},
    {"name": "Frequency",             "label": "Nombre de commandes",         "default": 5,    "min": 1,   "max": 50,    "step": 1},
    {"name": "MonetaryTotal",         "label": "Total dépensé (£)",           "default": 1000, "min": 0,   "max": 15000, "step": 0.01},
    {"name": "MonetaryAvg",           "label": "Dépense moyenne / commande (£)", "default": 200, "min": 5, "max": 500,  "step": 0.01},
    {"name": "MonetaryStd",           "label": "Écart-type dépenses",         "default": 50,   "min": 0,   "max": 500,   "step": 0.01},
    {"name": "TotalQuantity",         "label": "Quantité totale d'articles",  "default": 100,  "min": 0,   "max": 10000, "step": 1},
    {"name": "CustomerTenureDays",    "label": "Ancienneté client (jours)",   "default": 300,  "min": 0,   "max": 730,   "step": 1},
    {"name": "UniqueProducts",        "label": "Produits distincts achetés",  "default": 20,   "min": 1,   "max": 1000,  "step": 1},
    {"name": "SatisfactionScore",     "label": "Score de satisfaction (0-5)", "default": 3.5,  "min": 0,   "max": 5,     "step": 0.1},
    {"name": "SupportTicketsCount",   "label": "Tickets support ouverts",     "default": 1,    "min": 0,   "max": 15,    "step": 1},
    {"name": "Age",                   "label": "Âge estimé",                  "default": 35,   "min": 18,  "max": 81,    "step": 1},
    {"name": "WeekendPurchaseRatio",  "label": "Ratio achats weekend (0-1)",  "default": 0.3,  "min": 0,   "max": 1,     "step": 0.01},
    {"name": "ReturnRatio",           "label": "Taux de retour (0-1)",        "default": 0.05, "min": 0,   "max": 1,     "step": 0.01},
    {"name": "CancelledTransactions", "label": "Transactions annulées",       "default": 0,    "min": 0,   "max": 50,    "step": 1},
    {"name": "TotalTransactions",     "label": "Total transactions",          "default": 50,   "min": 1,   "max": 10000, "step": 1},
]

CLUSTER_LABELS = {
    0: "Dormants",
    1: "Occasionnels",
    2: "Réguliers",
    3: "Champions",
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", fields=FORM_FIELDS)


@app.route("/predict", methods=["POST"])
def predict():
    """Reçoit les données du formulaire et retourne les prédictions."""
    try:
        # Récupération des inputs
        client = {}
        for field in FORM_FIELDS:
            raw = request.form.get(field["name"], field["default"])
            client[field["name"]] = float(raw)

        df_client = pd.DataFrame([client])

        # Aligner sur les features du scaler
        if SCALER is not None:
            expected = SCALER.feature_names_in_
            for col in expected:
                if col not in df_client.columns:
                    df_client[col] = 0
            df_aligned = df_client[expected]
            X_scaled = SCALER.transform(df_aligned)
            X_df     = pd.DataFrame(X_scaled, columns=expected)
        else:
            X_df = df_client

        result = {
            "success": True,
            "inputs": client,
        }

        # ── Clustering
        if PCA is not None and KMEANS is not None:
            X_pca   = PCA.transform(X_df)
            cluster = int(KMEANS.predict(X_pca)[0])
            result["cluster"]       = cluster
            result["cluster_label"] = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")

        # ── Classification Churn
        if CLASSIFIER is not None:
            proba  = float(CLASSIFIER.predict_proba(X_df)[0, 1])
            churn  = int(CLASSIFIER.predict(X_df)[0])
            result["churn_probability"] = round(proba * 100, 1)
            result["churn_prediction"]  = churn
            result["churn_label"]       = "🔴 Risque élevé de départ" if churn else "🟢 Client fidèle"

            # Niveau de risque
            if proba >= 0.7:
                risk = "Critique"
            elif proba >= 0.5:
                risk = "Élevé"
            elif proba >= 0.3:
                risk = "Modéré"
            else:
                risk = "Faible"
            result["risk_level"] = risk

        # ── Régression monétaire
        if REGRESSOR is not None:
            monetary_drop = ["MonetaryTotal", "MonetaryAvg", "MonetaryStd",
                             "MonetaryPerDay", "AvgBasketValue"]
            X_reg = X_df.drop(columns=[c for c in monetary_drop if c in X_df.columns])
            try:
                predicted_monetary = float(REGRESSOR.predict(X_reg)[0])
                result["predicted_monetary"] = round(predicted_monetary, 2)
            except Exception:
                result["predicted_monetary"] = None

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/health")
def health():
    """Endpoint de santé pour vérifier que l'API tourne."""
    models_status = {
        "scaler":     SCALER is not None,
        "pca":        PCA is not None,
        "kmeans":     KMEANS is not None,
        "classifier": CLASSIFIER is not None,
        "regressor":  REGRESSOR is not None,
    }
    return jsonify({
        "status": "ok",
        "models": models_status,
        "all_loaded": all(models_status.values())
    })


# ── Démarrage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Application Flask démarrée")
    print("   Accès : http://127.0.0.1:5000")
    print("   Health: http://127.0.0.1:5000/health\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
