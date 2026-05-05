"""
FERBS ENERGY INTELLIGENCE — Dashboard Visualisation
====================================================
Génère des graphiques interactifs Plotly à partir de la base Ferbs.
Compatible Google Colab (affichage inline) + export HTML standalone.

USAGE dans Colab:
    from ferbs_pipeline import run_full_pipeline
    from ferbs_dashboard import FerbsDashboard
    
    db = run_full_pipeline()
    dash = FerbsDashboard(db)
    dash.show_all()
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3

try:
    from ferbs_pipeline import FerbsDatabase, FerbsAnalytics
except ImportError:
    pass  # Import optionnel pour usage standalone


# Palette Ferbs — couleurs de marque
COLORS = {
    "primary":    "#0F6E56",   # Vert foncé — couleur principale
    "secondary":  "#F5A623",   # Ambre — accent
    "danger":     "#D85A30",   # Rouge-orange — alertes
    "neutral":    "#4A5568",   # Gris
    "background": "#0D1117",   # Fond sombre
    "africa":     "#F5A623",
    "europe":     "#0F6E56",
    "global":     "#4A5568",
}

AFRICA_COLORS = px.colors.qualitative.Set2


class FerbsDashboard:
    def __init__(self, db):
        self.db = db
        self.analytics = FerbsAnalytics(db)

    # ------------------------------------------------------------------
    # GRAPHIQUE 1 : Accès électricité par pays africain (bar chart)
    # ------------------------------------------------------------------
    def chart_access_gap(self) -> go.Figure:
        """Bar chart horizontal — accès électricité Afrique"""
        df = self.analytics.energy_access_gap()
        
        if df.empty:
            print("[Dashboard] Pas de données access_pct")
            return go.Figure()
        
        df = df.sort_values("value")
        
        # Couleur selon le tier
        color_map = {
            "Critique (<30%)":      COLORS["danger"],
            "Faible (30-60%)":      "#E8A838",
            "Moyen (60-85%)":       COLORS["secondary"],
            "Bon (>85%)":           COLORS["primary"],
        }
        colors = [color_map.get(str(t), COLORS["neutral"]) for t in df["risk_tier"]]
        
        fig = go.Figure(go.Bar(
            x=df["value"],
            y=df["country_name"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in df["value"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Accès: %{x:.1f}%<extra></extra>",
        ))
        
        # Ligne référence monde
        fig.add_vline(x=91, line_dash="dash", line_color="white",
                      annotation_text="Monde: 91%", annotation_position="top right")
        
        fig.update_layout(
            title={
                "text": "Accès à l'électricité en Afrique<br><sup>% de la population — Source: World Bank / OWID</sup>",
                "font": {"size": 18, "color": "white"}
            },
            paper_bgcolor=COLORS["background"],
            plot_bgcolor="#161B22",
            font={"color": "white"},
            xaxis={"title": "% population avec accès", "range": [0, 110], "gridcolor": "#2D333B"},
            yaxis={"title": "", "gridcolor": "#2D333B"},
            height=500,
            margin={"l": 150, "r": 80, "t": 80, "b": 40},
        )
        
        return fig

    # ------------------------------------------------------------------
    # GRAPHIQUE 2 : Évolution production Nigeria (time series)
    # ------------------------------------------------------------------
    def chart_nigeria_generation(self) -> go.Figure:
        """Time series — génération électrique Nigeria avec zones saisonnières"""
        df = self.db.query(country_codes=["NGA"], metrics=["generation_mw"])
        
        if df.empty:
            print("[Dashboard] Pas de données Nigeria")
            return go.Figure()
        
        df = df.sort_values("timestamp")
        
        fig = go.Figure()
        
        # Zone saison des pluies (ombrage)
        for year in df["timestamp"].dt.year.unique():
            wet_start = pd.Timestamp(f"{year}-06-01", tz="UTC")
            wet_end   = pd.Timestamp(f"{year}-11-30", tz="UTC")
            fig.add_vrect(
                x0=wet_start, x1=wet_end,
                fillcolor="#0F6E56", opacity=0.08,
                layer="below", line_width=0,
            )
        
        # Courbe principale
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["value"],
            mode="lines",
            name="Génération MW",
            line={"color": COLORS["secondary"], "width": 2},
            fill="tozeroy",
            fillcolor="rgba(245, 166, 35, 0.1)",
            hovertemplate="%{x|%b %Y}<br><b>%{y:.0f} MW</b><extra></extra>",
        ))
        
        # Ligne capacité installée théorique
        fig.add_hline(y=13000, line_dash="dash", line_color=COLORS["danger"],
                      annotation_text="Capacité installée théorique: 13 GW",
                      annotation_font_color=COLORS["danger"])
        
        # Ligne capacité réelle disponible
        fig.add_hline(y=4500, line_dash="dot", line_color=COLORS["primary"],
                      annotation_text="Capacité disponible moyenne: ~4.5 GW")
        
        fig.update_layout(
            title={
                "text": "Production électrique Nigeria<br><sup>MW — Saisons des pluies en vert (meilleure hydro)</sup>",
                "font": {"size": 18, "color": "white"}
            },
            paper_bgcolor=COLORS["background"],
            plot_bgcolor="#161B22",
            font={"color": "white"},
            xaxis={"title": "", "gridcolor": "#2D333B"},
            yaxis={"title": "MW", "gridcolor": "#2D333B"},
            legend={"bgcolor": "rgba(0,0,0,0)"},
            height=420,
        )
        
        return fig

    # ------------------------------------------------------------------
    # GRAPHIQUE 3 : Mix énergétique comparatif (stacked bar)
    # ------------------------------------------------------------------
    def chart_energy_mix(self) -> go.Figure:
        """Mix renouvelable vs fossile par pays"""
        df = self.analytics.renewable_transition_score()
        
        if df.empty:
            print("[Dashboard] Pas de données mix énergétique")
            return go.Figure()
        
        df = df.dropna(subset=["value"]).sort_values("value", ascending=False).head(15)
        df["fossil_pct"] = 100 - df["value"]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name="Renouvelable",
            x=df["country_name"],
            y=df["value"],
            marker_color=COLORS["primary"],
            hovertemplate="%{x}<br>Renouvelable: <b>%{y:.1f}%</b><extra></extra>",
        ))
        
        fig.add_trace(go.Bar(
            name="Fossile",
            x=df["country_name"],
            y=df["fossil_pct"],
            marker_color=COLORS["danger"],
            hovertemplate="%{x}<br>Fossile: <b>%{y:.1f}%</b><extra></extra>",
        ))
        
        fig.update_layout(
            barmode="stack",
            title={
                "text": "Mix Énergétique — Part Renouvelable<br><sup>% de la production totale — Source: OWID</sup>",
                "font": {"size": 18, "color": "white"}
            },
            paper_bgcolor=COLORS["background"],
            plot_bgcolor="#161B22",
            font={"color": "white"},
            xaxis={"tickangle": -35, "gridcolor": "#2D333B"},
            yaxis={"title": "%", "gridcolor": "#2D333B"},
            legend={"bgcolor": "rgba(0,0,0,0)", "orientation": "h", "y": 1.1},
            height=450,
        )
        
        return fig

    # ------------------------------------------------------------------
    # GRAPHIQUE 4 : Consommation per capita — Afrique vs monde
    # ------------------------------------------------------------------
    def chart_per_capita_gap(self) -> go.Figure:
        """Scatter plot consommation per capita vs accès"""
        df_kwh = self.db.query(metrics=["per_capita_kwh"])
        df_access = self.db.query(metrics=["access_pct"])
        
        if df_kwh.empty or df_access.empty:
            return go.Figure()
        
        # Prendre la dernière valeur par pays
        latest_kwh = (df_kwh.sort_values("timestamp")
                            .groupby("country_code")
                            .last()
                            .reset_index()
                            .rename(columns={"value": "kwh_per_capita"}))
        
        latest_access = (df_access.sort_values("timestamp")
                                  .groupby("country_code")
                                  .last()
                                  .reset_index()
                                  .rename(columns={"value": "access_pct"}))
        
        merged = latest_kwh.merge(latest_access[["country_code", "access_pct"]], on="country_code", how="inner")
        merged = merged.dropna(subset=["kwh_per_capita", "access_pct"])
        
        if merged.empty:
            return go.Figure()
        
        fig = px.scatter(
            merged,
            x="access_pct",
            y="kwh_per_capita",
            text="country_name",
            size="kwh_per_capita",
            color="region",
            color_discrete_map={
                "africa":  COLORS["secondary"],
                "europe":  COLORS["primary"],
                "global":  COLORS["neutral"],
            },
            hover_data={"country_code": True},
        )
        
        fig.update_traces(textposition="top center", marker={"opacity": 0.8})
        
        # Quadrants
        fig.add_vline(x=85, line_dash="dash", line_color="#2D333B")
        fig.add_hline(y=1000, line_dash="dash", line_color="#2D333B",
                      annotation_text="Seuil 1 000 kWh/hab")
        
        fig.update_layout(
            title={
                "text": "Accès vs Consommation per capita<br><sup>Chaque point = un pays — taille proportionnelle à la conso</sup>",
                "font": {"size": 18, "color": "white"}
            },
            paper_bgcolor=COLORS["background"],
            plot_bgcolor="#161B22",
            font={"color": "white"},
            xaxis={"title": "% population avec accès", "gridcolor": "#2D333B"},
            yaxis={"title": "kWh par habitant/an", "gridcolor": "#2D333B"},
            height=480,
        )
        
        return fig

    # ------------------------------------------------------------------
    # DASHBOARD COMPLET
    # ------------------------------------------------------------------
    def show_all(self, export_html: bool = False):
        """Afficher tous les graphiques (Colab) et optionnellement exporter"""
        charts = [
            ("Access Gap",       self.chart_access_gap),
            ("Nigeria Timeline", self.chart_nigeria_generation),
            ("Energy Mix",       self.chart_energy_mix),
            ("Per Capita Gap",   self.chart_per_capita_gap),
        ]
        
        for name, fn in charts:
            print(f"\n--- {name} ---")
            try:
                fig = fn()
                if fig.data:
                    fig.show()
                    if export_html:
                        fname = f"ferbs_{name.lower().replace(' ', '_')}.html"
                        fig.write_html(fname)
                        print(f"Exporté: {fname}")
            except Exception as e:
                print(f"Erreur {name}: {e}")

    def export_full_dashboard(self, path: str = "ferbs_dashboard.html"):
        """Exporter un dashboard HTML standalone (une seule page)"""
        charts = [
            self.chart_access_gap(),
            self.chart_nigeria_generation(),
            self.chart_energy_mix(),
            self.chart_per_capita_gap(),
        ]
        
        from plotly.io import to_html
        
        html_parts = ["""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Ferbs Energy Intelligence</title>
  <style>
    body { background: #0D1117; color: white; font-family: monospace; margin: 0; padding: 20px; }
    h1 { color: #0F6E56; border-bottom: 1px solid #0F6E56; padding-bottom: 10px; }
    .chart { margin-bottom: 40px; }
    .subtitle { color: #4A5568; font-size: 12px; }
  </style>
</head>
<body>
<h1>⚡ FERBS ENERGY INTELLIGENCE</h1>
<p class="subtitle">Plateforme d'analyse énergétique Afrique + Global</p>
"""]
        
        for fig in charts:
            if fig.data:
                html_parts.append(f'<div class="chart">')
                html_parts.append(to_html(fig, full_html=False, include_plotlyjs="cdn"))
                html_parts.append("</div>")
        
        html_parts.append("</body></html>")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))
        
        print(f"Dashboard exporté: {path}")
        return path


# ============================================================
# USAGE RAPIDE DANS COLAB
# ============================================================
"""
Copier-coller dans Colab:

# Cellule 1 — Installation
!pip install requests pandas numpy plotly sqlalchemy beautifulsoup4 -q

# Cellule 2 — Pipeline
from ferbs_pipeline import run_full_pipeline
db = run_full_pipeline()

# Cellule 3 — Dashboard
from ferbs_dashboard import FerbsDashboard
dash = FerbsDashboard(db)
dash.show_all()

# Cellule 4 — Export HTML standalone
dash.export_full_dashboard("mon_dashboard.html")

# Cellule 5 — Requête directe
df = db.query(country_codes=["NGA", "ZAF"], metrics=["access_pct"])
print(df.tail(20))
"""
