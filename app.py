"""
FERBS ENERGY INTELLIGENCE — Web App (Streamlit)
================================================
Interface web deployable sur Railway / Render / Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import sys
import os

# Configuration page
st.set_page_config(
    page_title="Ferbs Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS custom
st.markdown("""
<style>
    .main { background-color: #0D1117; }
    .stMetric { background: #161B22; padding: 12px; border-radius: 8px; border: 0.5px solid #2D333B; }
    h1 { color: #0F6E56; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚡ Ferbs Energy Intelligence")
st.markdown("**Plateforme d'intelligence énergétique — Afrique & Global**")
st.divider()

# Import pipeline
@st.cache_resource
def load_pipeline():
    try:
        from ferbs_pipeline import run_full_pipeline, FerbsAnalytics
        db = run_full_pipeline()
        analytics = FerbsAnalytics(db)
        return db, analytics
    except Exception as e:
        return None, str(e)

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/Made%20in-Gabon%20🇬🇦-009A44", width=180)
    st.markdown("### Navigation")
    page = st.radio("", [
        "🌍 Vue d'ensemble",
        "📊 Accès électricité",
        "⚡ Production Nigeria",
        "🌱 Transition énergétique",
        "📈 Per capita"
    ])
    st.divider()
    st.markdown("**v0.1 MVP** · [GitHub](https://github.com/senihub/ferbs-energy-intelligence)")

# Chargement données
with st.spinner("Chargement des données énergétiques..."):
    result = load_pipeline()
    db, analytics = result

if db is None:
    st.error(f"Erreur pipeline: {analytics}")
    st.stop()

# Métriques globales
col1, col2, col3, col4 = st.columns(4)

stats = db.stats()
with col1:
    st.metric("Enregistrements", f"{stats['total_rows']:,}")
with col2:
    st.metric("Pays couverts", len(stats['by_country']))
with col3:
    st.metric("Sources de données", "4")
with col4:
    st.metric("Région principale", "Afrique")

st.divider()

# Pages
if page == "🌍 Vue d'ensemble":
    st.subheader("Vue d'ensemble — Déficit énergétique africain")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Le problème fondamental**
        - 600M personnes sans électricité en Afrique subsaharienne
        - Seulement 3.1% de la consommation mondiale
        - Gap d'investissement : 90 Mds$/an
        - Coût des pannes : 6-16% du CA des entreprises
        """)

    with col2:
        st.success("""
        **Ce que Ferbs résout**
        - Agrégation de données multi-sources normalisées
        - Score de risque réseau par pays
        - Prévision de pannes (roadmap v0.2)
        - Intelligence pour investisseurs et utilities
        """)

    # Table données disponibles
    st.subheader("Données disponibles par pays")
    gap = analytics.energy_access_gap()
    if not gap.empty:
        st.dataframe(
            gap[["country_name", "value", "no_access_pct", "risk_tier"]]
              .rename(columns={
                  "country_name": "Pays",
                  "value": "Accès électricité (%)",
                  "no_access_pct": "Sans accès (%)",
                  "risk_tier": "Niveau de risque"
              })
              .sort_values("Accès électricité (%)"),
            use_container_width=True,
            hide_index=True
        )

elif page == "📊 Accès électricité":
    st.subheader("Accès à l'électricité — Afrique")

    try:
        from ferbs_dashboard import FerbsDashboard
        dash = FerbsDashboard(db)
        fig = dash.chart_access_gap()
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur graphique: {e}")

    gap = analytics.energy_access_gap()
    if not gap.empty:
        worst = gap.sort_values("value").head(3)
        st.warning(f"⚠️ Pays critiques (accès < 30%) : {', '.join(worst['country_name'].tolist())}")

elif page == "⚡ Production Nigeria":
    st.subheader("Production électrique — Nigeria")

    st.info("""
    **Contexte** : Le Nigeria a 13 GW de capacité installée mais seulement ~4 GW disponibles.
    Les entreprises perdent en moyenne 6h d'électricité par jour.
    """)

    try:
        from ferbs_dashboard import FerbsDashboard
        dash = FerbsDashboard(db)
        fig = dash.chart_nigeria_generation()
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur graphique: {e}")

    vol = analytics.demand_volatility("NGA")
    if vol:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Production moyenne", f"{vol.get('mean_mw', 0):,.0f} MW")
        with col2:
            st.metric("Volatilité (CV)", f"{vol.get('cv_pct', 0):.1f}%")
        with col3:
            st.metric("Swing max", f"{vol.get('max_swing_mw', 0):,.0f} MW")

elif page == "🌱 Transition énergétique":
    st.subheader("Score de transition énergétique")

    try:
        from ferbs_dashboard import FerbsDashboard
        dash = FerbsDashboard(db)
        fig = dash.chart_energy_mix()
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur graphique: {e}")

    score = analytics.renewable_transition_score()
    if not score.empty:
        st.dataframe(
            score[["country_name", "value", "transition_score"]]
              .rename(columns={
                  "country_name": "Pays",
                  "value": "% Renouvelable",
                  "transition_score": "Score"
              })
              .sort_values("% Renouvelable", ascending=False),
            use_container_width=True,
            hide_index=True
        )

elif page == "📈 Per capita":
    st.subheader("Consommation per capita — Afrique vs Monde")

    try:
        from ferbs_dashboard import FerbsDashboard
        dash = FerbsDashboard(db)
        fig = dash.chart_per_capita_gap()
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur graphique: {e}")

# Footer
st.divider()
st.markdown("""
<div style='text-align:center;color:#4A5568;font-size:12px'>
Ferbs Energy Intelligence · Made in Gabon 🇬🇦 · 
<a href='https://github.com/senihub/ferbs-energy-intelligence' style='color:#0F6E56'>GitHub</a>
</div>
""", unsafe_allow_html=True)
