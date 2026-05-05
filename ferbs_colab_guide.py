# FERBS ENERGY INTELLIGENCE — Notebook Colab
# Copie chaque bloc dans une cellule Colab séparée
# =====================================================

# ── CELLULE 1 : Installation ──────────────────────
"""
!pip install requests pandas numpy plotly sqlalchemy beautifulsoup4 lxml openpyxl -q
print("✅ Packages installés")
"""

# ── CELLULE 2 : Télécharger les fichiers depuis GitHub ou upload ──
"""
# Option A — si tu héberges tes fichiers sur GitHub:
# !wget https://raw.githubusercontent.com/TON_USER/ferbs/main/ferbs_pipeline.py
# !wget https://raw.githubusercontent.com/TON_USER/ferbs/main/ferbs_dashboard.py

# Option B — upload manuel dans Colab (bouton 📁 à gauche)
# Glisse ferbs_pipeline.py et ferbs_dashboard.py dans le panneau fichiers
"""

# ── CELLULE 3 : Lancer le pipeline complet ────────
"""
from ferbs_pipeline import run_full_pipeline, FerbsAnalytics

db = run_full_pipeline()
# Ce pipeline va:
# 1. Charger les données OWID (190 pays, données annuelles)
# 2. Récupérer World Bank API (accès, renouvelables, per capita)
# 3. Générer les données Nigeria estimées
# Durée: ~2-3 minutes (téléchargement OWID ~10 Mo)
"""

# ── CELLULE 4 : Statistiques de la base ───────────
"""
stats = db.stats()
print(f"📊 Total enregistrements : {stats['total_rows']}")
print(f"📅 Période : {stats['date_range']}")
print(f"🌍 Pays : {list(stats['by_country'].keys())}")
"""

# ── CELLULE 5 : Requête directe ───────────────────
"""
# Voir les données Nigeria
df_nigeria = db.query(
    country_codes=["NGA"],
    metrics=["generation_mw", "access_pct", "per_capita_kwh"]
)
print(df_nigeria.tail(20))
"""

# ── CELLULE 6 : Analyses clés ─────────────────────
"""
analytics = FerbsAnalytics(db)

# Gap d'accès électricité
print("\\n=== GAP ACCÈS ÉLECTRICITÉ ===")
gap = analytics.energy_access_gap()
print(gap.sort_values("value").to_string(index=False))

# Volatilité Nigeria
print("\\n=== VOLATILITÉ RÉSEAU NIGERIA ===")
vol = analytics.demand_volatility("NGA")
for k, v in vol.items():
    print(f"  {k}: {v}")

# Score transition énergétique
print("\\n=== SCORE TRANSITION ÉNERGÉTIQUE ===")
score = analytics.renewable_transition_score()
print(score.sort_values("value", ascending=False).head(10).to_string(index=False))
"""

# ── CELLULE 7 : Dashboard visualisation ───────────
"""
from ferbs_dashboard import FerbsDashboard

dash = FerbsDashboard(db)

# Graphique 1 : Accès électricité Afrique
fig1 = dash.chart_access_gap()
fig1.show()
"""

# ── CELLULE 8 : Nigeria timeline ──────────────────
"""
fig2 = dash.chart_nigeria_generation()
fig2.show()
"""

# ── CELLULE 9 : Mix énergétique ───────────────────
"""
fig3 = dash.chart_energy_mix()
fig3.show()
"""

# ── CELLULE 10 : Export dashboard HTML ────────────
"""
# Exporte un fichier HTML standalone que tu peux ouvrir dans n'importe quel navigateur
path = dash.export_full_dashboard("ferbs_dashboard.html")

# Dans Colab, télécharger le fichier:
from google.colab import files
files.download("ferbs_dashboard.html")
"""

# ── CELLULE 11 : Ajouter RTE France (optionnel) ───
"""
# 1. Inscris-toi sur https://data.rte-france.com (gratuit)
# 2. Crée une application, récupère ton token OAuth
# 3. Lance le pipeline avec le token:

from ferbs_pipeline import run_full_pipeline
db = run_full_pipeline(rte_token="TON_TOKEN_ICI")
"""

# ── CELLULE 12 : Prochaines étapes ────────────────
"""
# NEXT STEPS pour faire évoluer Ferbs:

# 1. DONNÉES TEMPS RÉEL
#    - Activer RTE (consommation France toutes les 30 min)
#    - Ajouter ENTSO-E (Europe entière, API gratuite)
#    - Scraping NERC Nigeria (rapports PDF mensuels)

# 2. MODÈLE DE PRÉVISION (Phase 2 de ton plan d'apprentissage)
#    from sklearn.ensemble import RandomForestRegressor
#    # Entraîner sur données historiques Nigeria
#    # Prédire génération J+1, J+7

# 3. ALERTES AUTOMATIQUES
#    # Détecter anomalies (chutes soudaines de génération)
#    from sklearn.ensemble import IsolationForest

# 4. DÉPLOIEMENT
#    # FastAPI sur Railway/Render (gratuit)
#    # Dashboard Streamlit (gratuit)
#    # Base PostgreSQL sur Supabase (gratuit 500 Mo)
"""
