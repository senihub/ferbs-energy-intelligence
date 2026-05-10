"""
FERBS ENERGY INTELLIGENCE — Outage Prediction Model
====================================================
Modèle XGBoost de prévision de pannes électriques
Cible : prédire P(panne) dans les 24-48h sur un réseau africain

USAGE dans Colab :
    from ferbs_outage_model import OutagePredictor
    model = OutagePredictor()
    model.train()
    risk = model.predict_risk()
    print(risk)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, 
                                  precision_score, recall_score, 
                                  roc_auc_score, confusion_matrix)
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
except ImportError:
    print("Installation des packages ML...")
    import subprocess
    subprocess.run(["pip", "install", "xgboost", "scikit-learn", "-q"])
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report,
                                  precision_score, recall_score,
                                  roc_auc_score, confusion_matrix)
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb


# ============================================================
# 1. GÉNÉRATEUR DE DONNÉES D'ENTRAÎNEMENT
# ============================================================

def generate_training_data(n_months: int = 72) -> pd.DataFrame:
    """
    Génère un dataset réaliste pour entraîner le modèle.
    Basé sur les patterns réels du réseau nigérian (NERC 2018-2023).
    
    Features utilisées :
    - generation_mw        : puissance générée actuelle
    - load_mw              : charge demandée
    - generation_gap       : écart offre/demande (clé !)
    - hour                 : heure de la journée
    - month                : mois (saisonnalité)
    - is_wet_season        : saison des pluies (meilleure hydro)
    - days_since_maintenance : jours depuis dernière maintenance
    - voltage_variance     : variance de tension (instabilité)
    - temperature          : température (impact climatisation)
    - oil_price            : prix pétrole (impact coût carburant)
    - consecutive_low_gen  : jours consécutifs de faible génération
    
    Target :
    - outage_24h : 1 = panne dans les 24h, 0 = pas de panne
    """
    np.random.seed(42)
    records = []

    start_date = datetime(2018, 1, 1)

    for day in range(n_months * 30):
        current_date = start_date + timedelta(days=day)
        month = current_date.month
        year = current_date.year

        # Saisonnalité Nigeria
        is_wet_season = month in [6, 7, 8, 9, 10, 11]

        # Génération de base avec tendance d'amélioration
        improvement = (year - 2018) * 0.8
        base_gen = 4200 if is_wet_season else 3600
        generation_mw = max(1200, base_gen + improvement + np.random.normal(0, 400))

        # Charge demandée (toujours supérieure à la génération = délestage)
        load_mw = generation_mw + np.random.uniform(500, 3000)

        # Gap offre/demande — plus c'est grand, plus le risque est élevé
        generation_gap = load_mw - generation_mw

        # Maintenance — cycle de 30-45 jours
        days_since_maintenance = (day % 35) + np.random.randint(0, 10)

        # Variance de tension — indicateur technique d'instabilité
        voltage_variance = np.random.exponential(2.5) + (generation_gap / 1000)

        # Température (Celsius) — impact sur climatisation donc sur charge
        temp_base = 28 if is_wet_season else 35
        temperature = temp_base + np.random.normal(0, 3)

        # Prix pétrole — impact sur coût du carburant des générateurs
        oil_price = 70 + np.random.normal(0, 15) + (10 if year >= 2021 else 0)

        # Jours consécutifs de faible génération
        consecutive_low_gen = max(0, int(np.random.exponential(2)))

        # ── CALCUL DE LA PROBABILITÉ DE PANNE ──────────────────
        # Basé sur les vrais facteurs de risque réseau Nigeria
        p_outage = 0.15  # Probabilité de base (15% par jour)

        # Gap élevé = risque plus grand
        if generation_gap > 3000:
            p_outage += 0.35
        elif generation_gap > 2000:
            p_outage += 0.20
        elif generation_gap > 1000:
            p_outage += 0.10

        # Maintenance dépassée = risque plus grand
        if days_since_maintenance > 30:
            p_outage += 0.15

        # Tension instable = risque plus grand
        if voltage_variance > 5:
            p_outage += 0.20

        # Génération très faible = risque critique
        if generation_mw < 2000:
            p_outage += 0.25

        # Saison sèche = moins d'hydro = plus de risque
        if not is_wet_season:
            p_outage += 0.10

        # Chaleur extrême = surcharge climatisation
        if temperature > 38:
            p_outage += 0.10

        # Plafonner à 95%
        p_outage = min(0.95, p_outage)

        # Générer le label (panne ou pas)
        outage = int(np.random.random() < p_outage)

        records.append({
            "date":                   current_date,
            "generation_mw":          round(generation_mw, 1),
            "load_mw":                round(load_mw, 1),
            "generation_gap":         round(generation_gap, 1),
            "hour":                   current_date.hour,
            "month":                  month,
            "day_of_week":            current_date.weekday(),
            "is_wet_season":          int(is_wet_season),
            "days_since_maintenance": days_since_maintenance,
            "voltage_variance":       round(voltage_variance, 3),
            "temperature":            round(temperature, 1),
            "oil_price":              round(oil_price, 2),
            "consecutive_low_gen":    consecutive_low_gen,
            "outage_24h":             outage,
        })

    df = pd.DataFrame(records)
    print(f"[Data] Dataset généré : {len(df)} jours")
    print(f"[Data] Taux de pannes : {df['outage_24h'].mean():.1%}")
    return df


# ============================================================
# 2. MODÈLE DE PRÉVISION — XGBoost
# ============================================================

class OutagePredictor:
    """
    Modèle XGBoost de prévision de pannes électriques.
    
    Pourquoi XGBoost et pas un réseau de neurones ?
    - Données tabulaires structurées → XGBoost gagne presque toujours
    - Interprétable (feature importance)
    - Entraîne en secondes, pas en heures
    - Utilisé en production par des équipes ML senior
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = [
            "generation_mw", "load_mw", "generation_gap",
            "month", "day_of_week", "is_wet_season",
            "days_since_maintenance", "voltage_variance",
            "temperature", "oil_price", "consecutive_low_gen"
        ]
        self.trained = False
        self.metrics = {}

    def train(self, df: pd.DataFrame = None) -> dict:
        """
        Entraîner le modèle XGBoost.
        Si df est None, génère des données synthétiques Nigeria.
        """
        if df is None:
            print("[Model] Génération des données d'entraînement Nigeria...")
            df = generate_training_data(n_months=72)

        # Features et target
        X = df[self.feature_cols]
        y = df["outage_24h"]

        # Split train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"[Model] Entraînement sur {len(X_train)} exemples...")
        print(f"[Model] Test sur {len(X_test)} exemples...")

        # ── MODÈLE XGBOOST ─────────────────────────────────────
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,  # Ajuster si données déséquilibrées
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── ÉVALUATION ──────────────────────────────────────────
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        auc       = roc_auc_score(y_test, y_prob)

        self.metrics = {
            "precision":  round(precision, 3),
            "recall":     round(recall, 3),
            "auc_roc":    round(auc, 3),
            "n_train":    len(X_train),
            "n_test":     len(X_test),
            "outage_rate": round(y.mean(), 3),
        }

        self.trained = True

        # ── RÉSULTATS ───────────────────────────────────────────
        print("\n" + "=" * 50)
        print("FERBS — Résultats du modèle de prévision")
        print("=" * 50)
        print(f"  Précision  : {precision:.1%}  (sur les pannes prédites, combien sont réelles)")
        print(f"  Rappel     : {recall:.1%}  (sur les vraies pannes, combien on détecte)")
        print(f"  AUC-ROC    : {auc:.3f}  (1.0 = parfait, 0.5 = aléatoire)")
        print("=" * 50)

        return self.metrics

    def predict_risk(self, input_data: dict = None) -> dict:
        """
        Prédire le risque de panne pour un réseau donné.
        
        input_data : dict avec les features du réseau
        Si None, utilise un scénario Nigeria typique.
        
        Retourne : score de risque 0-100 + niveau + recommandation
        """
        if not self.trained:
            raise ValueError("Le modèle n'est pas encore entraîné. Lance .train() d'abord.")

        # Scénario par défaut — Nigeria jour typique
        if input_data is None:
            input_data = {
                "generation_mw":          3800,
                "load_mw":                6200,
                "generation_gap":         2400,
                "month":                  datetime.now().month,
                "day_of_week":            datetime.now().weekday(),
                "is_wet_season":          1 if datetime.now().month in [6,7,8,9,10,11] else 0,
                "days_since_maintenance": 15,
                "voltage_variance":       3.2,
                "temperature":            32,
                "oil_price":              78,
                "consecutive_low_gen":    2,
            }

        # Préparer les données
        X = pd.DataFrame([input_data])[self.feature_cols]

        # Probabilité de panne
        prob = self.model.predict_proba(X)[0][1]
        risk_score = round(prob * 100, 1)

        # Niveau de risque
        if risk_score >= 75:
            level = "CRITIQUE"
            color = "🔴"
            action = "Intervention immédiate requise. Activer les générateurs de secours."
        elif risk_score >= 50:
            level = "ÉLEVÉ"
            color = "🟠"
            action = "Alerter les équipes de maintenance. Réduire la charge non-essentielle."
        elif risk_score >= 25:
            level = "MODÉRÉ"
            color = "🟡"
            action = "Surveillance renforcée. Planifier une maintenance préventive."
        else:
            level = "FAIBLE"
            color = "🟢"
            action = "Réseau stable. Continuer le monitoring standard."

        result = {
            "risk_score":    risk_score,
            "risk_level":    level,
            "probability":   f"{prob:.1%}",
            "action":        action,
            "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
            "input_summary": {
                "generation_mw":  input_data["generation_mw"],
                "load_mw":        input_data["load_mw"],
                "gap_mw":         input_data["generation_gap"],
                "temperature":    input_data["temperature"],
            }
        }

        print(f"\n{color} FERBS RISK SCORE : {risk_score}/100 — {level}")
        print(f"   Probabilité de panne dans 24h : {prob:.1%}")
        print(f"   Action recommandée : {action}")

        return result

    def feature_importance(self) -> pd.DataFrame:
        """
        Quelles variables influencent le plus les pannes ?
        Insight clé pour les utilities.
        """
        if not self.trained:
            raise ValueError("Modèle non entraîné.")

        importance = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        print("\n=== FACTEURS DE RISQUE — par ordre d'importance ===")
        for _, row in importance.iterrows():
            bar = "█" * int(row["importance"] * 100)
            print(f"  {row['feature']:<28} {bar} {row['importance']:.3f}")

        return importance

    def simulate_scenario(self, scenario_name: str) -> dict:
        """
        Simuler des scénarios de risque prédéfinis.
        Utile pour les démonstrations aux utilities et fonds.
        """
        scenarios = {
            "nigeria_normal": {
                "generation_mw": 4200, "load_mw": 6000,
                "generation_gap": 1800, "month": 8,
                "day_of_week": 2, "is_wet_season": 1,
                "days_since_maintenance": 10, "voltage_variance": 2.1,
                "temperature": 28, "oil_price": 75,
                "consecutive_low_gen": 0,
            },
            "nigeria_crise": {
                "generation_mw": 1800, "load_mw": 7000,
                "generation_gap": 5200, "month": 3,
                "day_of_week": 1, "is_wet_season": 0,
                "days_since_maintenance": 45, "voltage_variance": 8.5,
                "temperature": 38, "oil_price": 95,
                "consecutive_low_gen": 7,
            },
            "gabon_normal": {
                "generation_mw": 350, "load_mw": 420,
                "generation_gap": 70, "month": 6,
                "day_of_week": 3, "is_wet_season": 1,
                "days_since_maintenance": 20, "voltage_variance": 1.5,
                "temperature": 26, "oil_price": 75,
                "consecutive_low_gen": 0,
            },
            "gabon_crise": {
                "generation_mw": 180, "load_mw": 450,
                "generation_gap": 270, "month": 8,
                "day_of_week": 0, "is_wet_season": 1,
                "days_since_maintenance": 60, "voltage_variance": 6.2,
                "temperature": 33, "oil_price": 90,
                "consecutive_low_gen": 4,
            },
        }

        if scenario_name not in scenarios:
            print(f"Scénarios disponibles : {list(scenarios.keys())}")
            return {}

        print(f"\n[Simulation] Scénario : {scenario_name}")
        return self.predict_risk(scenarios[scenario_name])


# ============================================================
# 3. DÉTECTION D'ANOMALIES — Isolation Forest
# ============================================================

class AnomalyDetector:
    """
    Détecte les comportements anormaux du réseau
    AVANT qu'ils ne causent une panne.
    Complément au modèle XGBoost.
    """

    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1,  # 10% des points considérés anormaux
            random_state=42,
            n_estimators=100,
        )
        self.trained = False

    def train(self, df: pd.DataFrame):
        feature_cols = ["generation_mw", "generation_gap",
                        "voltage_variance", "temperature"]
        X = df[feature_cols].dropna()
        self.model.fit(X)
        self.trained = True
        print("[Anomaly] Détecteur entraîné sur données normales")

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retourne les lignes anormales détectées"""
        feature_cols = ["generation_mw", "generation_gap",
                        "voltage_variance", "temperature"]
        X = df[feature_cols].dropna()
        scores = self.model.decision_function(X)
        predictions = self.model.predict(X)

        df = df.copy()
        df["anomaly_score"] = scores
        df["is_anomaly"] = predictions == -1

        anomalies = df[df["is_anomaly"]]
        print(f"[Anomaly] {len(anomalies)} anomalies détectées sur {len(df)} points ({len(anomalies)/len(df):.1%})")
        return anomalies


# ============================================================
# 4. USAGE COMPLET DANS COLAB
# ============================================================
"""
Copie ces cellules dans Google Colab :

# ── Cellule 1 : Installation ──
!pip install xgboost scikit-learn -q

# ── Cellule 2 : Entraînement ──
from ferbs_outage_model import OutagePredictor, generate_training_data

model = OutagePredictor()
metrics = model.train()
print(metrics)

# ── Cellule 3 : Prédiction scénario Nigeria normal ──
model.simulate_scenario("nigeria_normal")

# ── Cellule 4 : Prédiction scénario Nigeria en crise ──
model.simulate_scenario("nigeria_crise")

# ── Cellule 5 : Scénario Gabon ──
model.simulate_scenario("gabon_normal")
model.simulate_scenario("gabon_crise")

# ── Cellule 6 : Facteurs de risque ──
importance = model.feature_importance()

# ── Cellule 7 : Prédiction personnalisée ──
# Remplace les valeurs par les vraies données de ton réseau
risk = model.predict_risk({
    "generation_mw":          3200,
    "load_mw":                5800,
    "generation_gap":         2600,
    "month":                  11,
    "day_of_week":            0,
    "is_wet_season":          1,
    "days_since_maintenance": 25,
    "voltage_variance":       4.1,
    "temperature":            31,
    "oil_price":              82,
    "consecutive_low_gen":    3,
})

# ── Cellule 8 : Détection d'anomalies ──
from ferbs_outage_model import AnomalyDetector, generate_training_data

df = generate_training_data()
detector = AnomalyDetector()
detector.train(df)
anomalies = detector.detect(df)
print(anomalies[["date", "generation_mw", "generation_gap", "anomaly_score"]].head(10))
"""
