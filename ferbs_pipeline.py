"""
FERBS ENERGY INTELLIGENCE — Data Pipeline
==========================================
Architecture: Ingest → Normalize → Store → Analyze → Export
Compatible: Google Colab + Python 3.9+

INSTALLATION (première cellule Colab):
!pip install requests pandas numpy plotly sqlalchemy beautifulsoup4 lxml openpyxl
"""

# ============================================================
# 0. SETUP & IMPORTS
# ============================================================
import requests
import pandas as pd
import numpy as np
import sqlite3
import json
import os
import time
from datetime import datetime, timedelta
from io import StringIO, BytesIO
from typing import Optional

# ============================================================
# 1. CONFIGURATION CENTRALE
# ============================================================
CONFIG = {
    "db_path": "ferbs_energy.db",
    "raw_data_dir": "data/raw",
    "clean_data_dir": "data/clean",
    
    # RTE France (inscription gratuite sur data.rte-france.com)
    "rte_token": "VOTRE_TOKEN_RTE",  # Remplacer après inscription
    "rte_base_url": "https://digital.iservices.rte-france.com/open_api",
    
    # World Bank (pas de clé requise)
    "worldbank_url": "https://api.worldbank.org/v2",
    
    # Our World in Data (CSV directs, pas d'API key)
    "owid_base": "https://raw.githubusercontent.com/owid/energy-data/master",
}

# Schéma de normalisation — toutes les sources convergeront vers ce format
STANDARD_SCHEMA = {
    "timestamp":     "datetime64[ns]",   # Date/heure UTC
    "country_code":  "str",              # ISO 3166 (NG, GA, FR, ZA...)
    "country_name":  "str",
    "region":        "str",              # africa / europe / global
    "metric":        "str",              # demand_mwh / production_mwh / price_eur_mwh
    "value":         "float64",
    "unit":          "str",              # MWh, MW, USD, etc.
    "source":        "str",              # rte / worldbank / owid / nerc / manual
    "quality_flag":  "str",             # ok / estimated / missing
}


# ============================================================
# 2. COUCHE D'INGESTION — Une classe par source
# ============================================================

class RTEIngester:
    """
    Données France — RTE Open Data
    Inscription gratuite : https://data.rte-france.com
    Endpoints utilisés : consumption, generation, physical_flows
    """
    def __init__(self, token: str):
        self.token = token
        self.base = CONFIG["rte_base_url"]
        self.headers = {"Authorization": f"Bearer {token}"}

    def get_token_oauth(self, client_id: str, client_secret: str) -> str:
        """OAuth2 — obtenir un bearer token depuis client_id/secret RTE"""
        import base64
        credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        resp = requests.post(
            "https://digital.iservices.rte-france.com/token/oauth/",
            headers={"Authorization": f"Basic {credentials}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials"}
        )
        return resp.json().get("access_token", "")

    def fetch_consumption(self, start: str, end: str) -> pd.DataFrame:
        """
        Consommation électrique française en MW (pas de 30 min)
        start/end format: "2024-01-01T00:00:00+01:00"
        """
        url = f"{self.base}/consumption/v1/consumption"
        params = {"start_date": start, "end_date": end, "type": "D"}
        
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records = []
            for item in data.get("consumption", []):
                for val in item.get("values", []):
                    records.append({
                        "timestamp": val["start_date"],
                        "value": val["value"],
                        "type": item.get("type", "D"),
                    })
            
            df = pd.DataFrame(records)
            if df.empty:
                print("[RTE] Aucune donnée reçue — vérifier le token")
                return pd.DataFrame()
            
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df
        
        except Exception as e:
            print(f"[RTE] Erreur: {e}")
            return pd.DataFrame()

    def to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convertir vers le schéma normalisé Ferbs"""
        if df.empty:
            return df
        return pd.DataFrame({
            "timestamp":    df["timestamp"],
            "country_code": "FR",
            "country_name": "France",
            "region":       "europe",
            "metric":       "demand_mw",
            "value":        df["value"],
            "unit":         "MW",
            "source":       "rte",
            "quality_flag": "ok",
        })


class OWIDIngester:
    """
    Our World in Data — Energy Dataset
    Source parfaite pour l'Afrique : Nigeria, Gabon, Afrique du Sud, Kenya...
    Données annuelles : consommation, production, mix énergétique, accès
    PAS DE CLÉ API REQUISE — CSV public sur GitHub
    """
    def __init__(self):
        self.url = f"{CONFIG['owid_base']}/owid-energy-data.csv"
        self._cache = None

    def fetch(self) -> pd.DataFrame:
        """Charger le dataset complet (190 pays, 1900-2023)"""
        if self._cache is not None:
            return self._cache
        
        print("[OWID] Chargement du dataset énergétique mondial...")
        try:
            df = pd.read_csv(self.url, low_memory=False)
            self._cache = df
            print(f"[OWID] {len(df)} lignes chargées — {df['country'].nunique()} pays")
            return df
        except Exception as e:
            print(f"[OWID] Erreur: {e}")
            return pd.DataFrame()

    def get_africa(self, countries: list = None) -> pd.DataFrame:
        """
        Extraire les données africaines
        Métriques clés disponibles dans OWID:
        - electricity_demand        : consommation totale TWh
        - electricity_generation    : production totale TWh
        - electricity_share_renewables
        - access_to_electricity     : % population avec accès
        - per_capita_electricity    : kWh par habitant
        - fossil_electricity        : TWh fossile
        - solar_electricity, wind_electricity, hydro_electricity
        """
        df = self.fetch()
        if df.empty:
            return df
        
        african_countries = countries or [
            "Nigeria", "Gabon", "South Africa", "Kenya", "Ethiopia",
            "Ghana", "Ivory Coast", "Senegal", "Cameroon", "Tanzania",
            "Democratic Republic of Congo", "Angola", "Egypt", "Morocco"
        ]
        
        african_cols = [
            "country", "year", "iso_code",
            "electricity_demand", "electricity_generation",
            "access_to_electricity", "per_capita_electricity",
            "fossil_electricity", "solar_electricity",
            "wind_electricity", "hydro_electricity",
            "electricity_share_renewables",
            "population", "gdp"
        ]
        
        # Garder seulement les colonnes qui existent
        available_cols = [c for c in african_cols if c in df.columns]
        
        mask = df["country"].isin(african_countries)
        return df[mask][available_cols].copy()

    def to_standard_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transformer format wide → long pour normalisation"""
        if df.empty:
            return df
        
        metric_cols = [c for c in df.columns 
                      if c not in ["country", "year", "iso_code", "population", "gdp"]]
        
        records = []
        for _, row in df.iterrows():
            for metric in metric_cols:
                if pd.notna(row.get(metric)):
                    records.append({
                        "timestamp":    pd.Timestamp(f"{int(row['year'])}-01-01", tz="UTC"),
                        "country_code": str(row.get("iso_code", ""))[:3],
                        "country_name": row["country"],
                        "region":       "africa",
                        "metric":       metric,
                        "value":        float(row[metric]),
                        "unit":         "TWh" if "electricity" in metric else "pct",
                        "source":       "owid",
                        "quality_flag": "ok",
                    })
        
        return pd.DataFrame(records)


class WorldBankIngester:
    """
    World Bank Open Data — Indicateurs énergie
    Indicateurs utiles:
    - EG.USE.ELEC.KH.PC : consommation électrique kWh/habitant
    - EG.ELC.ACCS.ZS    : accès électricité (% population)
    - EG.ELC.RNEW.ZS    : électricité renouvelable (% total)
    - EG.IMP.CONS.ZS    : importations nettes énergie
    PAS DE CLÉ API REQUISE
    """
    def __init__(self):
        self.base = CONFIG["worldbank_url"]

    def fetch_indicator(self, indicator: str, countries: list, 
                        start_year: int = 2000, end_year: int = 2023) -> pd.DataFrame:
        """Récupérer un indicateur pour une liste de pays"""
        country_str = ";".join(countries)
        url = f"{self.base}/country/{country_str}/indicator/{indicator}"
        params = {
            "format": "json",
            "date": f"{start_year}:{end_year}",
            "per_page": 1000,
        }
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if len(data) < 2 or not data[1]:
                return pd.DataFrame()
            
            records = []
            for item in data[1]:
                if item.get("value") is not None:
                    records.append({
                        "country_code": item["countryiso3code"],
                        "country_name": item["country"]["value"],
                        "year":         int(item["date"]),
                        "value":        float(item["value"]),
                        "indicator":    indicator,
                    })
            
            return pd.DataFrame(records)
        
        except Exception as e:
            print(f"[WorldBank] Erreur {indicator}: {e}")
            return pd.DataFrame()

    def fetch_energy_bundle(self, africa_iso: list) -> pd.DataFrame:
        """Récupérer tous les indicateurs énergie pour l'Afrique"""
        indicators = {
            "EG.USE.ELEC.KH.PC": ("per_capita_kwh", "kWh"),
            "EG.ELC.ACCS.ZS":    ("access_pct", "pct"),
            "EG.ELC.RNEW.ZS":    ("renewables_pct", "pct"),
        }
        
        all_dfs = []
        for indicator_code, (metric_name, unit) in indicators.items():
            print(f"[WorldBank] Récupération {metric_name}...")
            df = self.fetch_indicator(indicator_code, africa_iso)
            if not df.empty:
                df["metric"] = metric_name
                df["unit"] = unit
                all_dfs.append(df)
            time.sleep(0.5)  # Respecter le rate limiting
        
        if not all_dfs:
            return pd.DataFrame()
        
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Normaliser vers schéma Ferbs
        return pd.DataFrame({
            "timestamp":    pd.to_datetime(combined["year"].astype(str) + "-01-01", utc=True),
            "country_code": combined["country_code"],
            "country_name": combined["country_name"],
            "region":       "africa",
            "metric":       combined["metric"],
            "value":        combined["value"],
            "unit":         combined["unit"],
            "source":       "worldbank",
            "quality_flag": "ok",
        })


class NERECIngester:
    """
    NERC Nigeria — données électricité Nigeria
    Source: rapports mensuels PDF + données publiques
    Stratégie: téléchargement CSV quand disponible, 
               simulation réaliste sinon (pour prototype)
    """
    def get_nigerian_data_synthetic(self, years: range = range(2018, 2024)) -> pd.DataFrame:
        """
        Données synthétiques réalistes pour Nigeria basées sur:
        - Capacité installée: ~13 GW (dont ~4 GW disponible réellement)
        - Consommation moyenne: 4,000-5,000 MW (délestages fréquents)
        - Forte saisonnalité: saison des pluies (hydro fort) vs sèche
        - Source: rapports NERC 2018-2023 publics
        """
        np.random.seed(42)
        records = []
        
        for year in years:
            for month in range(1, 13):
                # Saisonnalité Nigeria: hydro fort en juil-nov
                is_wet_season = month in [6, 7, 8, 9, 10, 11]
                base_mw = 4200 if is_wet_season else 3600
                
                # Variabilité + tendance de légère amélioration
                improvement = (year - 2018) * 80
                noise = np.random.normal(0, 300)
                generation = max(1500, base_mw + improvement + noise)
                
                records.append({
                    "timestamp":    pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC"),
                    "country_code": "NGA",
                    "country_name": "Nigeria",
                    "region":       "africa",
                    "metric":       "generation_mw",
                    "value":        round(generation, 1),
                    "unit":         "MW",
                    "source":       "nerc_estimated",
                    "quality_flag": "estimated",
                })
        
        return pd.DataFrame(records)


# ============================================================
# 3. COUCHE DE STOCKAGE — SQLite (portable, zéro serveur)
# ============================================================

class FerbsDatabase:
    """
    Base SQLite locale — suffisante pour prototype et MVP.
    Migration vers PostgreSQL/TimescaleDB pour production.
    """
    def __init__(self, db_path: str = CONFIG["db_path"]):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        """Créer les tables si elles n'existent pas"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS energy_data (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                country_code TEXT NOT NULL,
                country_name TEXT,
                region       TEXT,
                metric       TEXT NOT NULL,
                value        REAL,
                unit         TEXT,
                source       TEXT,
                quality_flag TEXT,
                created_at   TEXT DEFAULT (datetime('now')),
                UNIQUE(timestamp, country_code, metric, source)
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_country_metric 
            ON energy_data(country_code, metric, timestamp)
        """)
        
        self.conn.commit()
        print(f"[DB] Base initialisée: {self.db_path}")

    def insert(self, df: pd.DataFrame) -> int:
        """Insérer des données (ignore les doublons)"""
        if df.empty:
            return 0
        
        df = df.copy()
        df["timestamp"] = df["timestamp"].astype(str)
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                self.conn.execute("""
                    INSERT OR IGNORE INTO energy_data 
                    (timestamp, country_code, country_name, region, metric, value, unit, source, quality_flag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["timestamp"], row["country_code"], row.get("country_name"),
                    row.get("region"), row["metric"], row.get("value"),
                    row.get("unit"), row.get("source"), row.get("quality_flag", "ok")
                ))
                inserted += 1
            except Exception:
                pass
        
        self.conn.commit()
        return inserted

    def query(self, country_codes: list = None, metrics: list = None,
              start: str = None, end: str = None) -> pd.DataFrame:
        """Requête flexible avec filtres"""
        sql = "SELECT * FROM energy_data WHERE 1=1"
        params = []
        
        if country_codes:
            placeholders = ",".join(["?" for _ in country_codes])
            sql += f" AND country_code IN ({placeholders})"
            params.extend(country_codes)
        
        if metrics:
            placeholders = ",".join(["?" for _ in metrics])
            sql += f" AND metric IN ({placeholders})"
            params.extend(metrics)
        
        if start:
            sql += " AND timestamp >= ?"
            params.append(start)
        
        if end:
            sql += " AND timestamp <= ?"
            params.append(end)
        
        sql += " ORDER BY timestamp"
        
        df = pd.read_sql(sql, self.conn, params=params)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def stats(self) -> dict:
        """Statistiques de la base"""
        cursor = self.conn.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM energy_data")
        count, min_ts, max_ts = cursor.fetchone()
        
        cursor = self.conn.execute("SELECT country_code, COUNT(*) FROM energy_data GROUP BY country_code")
        by_country = dict(cursor.fetchall())
        
        return {
            "total_rows": count,
            "date_range": f"{min_ts} → {max_ts}",
            "by_country": by_country,
        }


# ============================================================
# 4. COUCHE ANALYTIQUE — Métriques clés
# ============================================================

class FerbsAnalytics:
    """Calcul des métriques stratégiques Ferbs"""

    def __init__(self, db: FerbsDatabase):
        self.db = db

    def energy_access_gap(self) -> pd.DataFrame:
        """
        Gap d'accès à l'électricité par pays africain
        Retourne: pays, % accès, population sans électricité estimée
        """
        df = self.db.query(metrics=["access_pct"], country_codes=None)
        if df.empty:
            return pd.DataFrame()
        
        latest = (df.sort_values("timestamp")
                    .groupby("country_code")
                    .last()
                    .reset_index())
        
        latest["no_access_pct"] = 100 - latest["value"]
        latest["risk_tier"] = pd.cut(
            latest["value"],
            bins=[0, 30, 60, 85, 100],
            labels=["Critique (<30%)", "Faible (30-60%)", "Moyen (60-85%)", "Bon (>85%)"]
        )
        return latest[["country_code", "country_name", "value", "no_access_pct", "risk_tier"]]

    def demand_volatility(self, country_code: str) -> dict:
        """
        Calcul de la volatilité de la demande — indicateur clé de risque réseau
        Retourne: std, cv (coefficient de variation), max_swing
        """
        df = self.db.query(country_codes=[country_code], metrics=["generation_mw", "demand_mw"])
        if df.empty:
            return {}
        
        values = df["value"].dropna()
        if len(values) < 3:
            return {}
        
        return {
            "country": country_code,
            "mean_mw":  round(values.mean(), 1),
            "std_mw":   round(values.std(), 1),
            "cv_pct":   round((values.std() / values.mean()) * 100, 2),  # + élevé = + instable
            "max_swing_mw": round(values.max() - values.min(), 1),
            "n_observations": len(values),
        }

    def renewable_transition_score(self) -> pd.DataFrame:
        """Score de transition énergétique pour chaque pays"""
        df = self.db.query(metrics=["renewables_pct"])
        if df.empty:
            return pd.DataFrame()
        
        latest = (df.sort_values("timestamp")
                    .groupby("country_code")
                    .last()
                    .reset_index())
        
        latest["transition_score"] = pd.cut(
            latest["value"],
            bins=[0, 10, 30, 60, 100],
            labels=["Fossile dominant", "Début transition", "Transition avancée", "Leader renouvelable"]
        )
        return latest[["country_code", "country_name", "value", "transition_score"]]

    def per_capita_energy_ranking(self) -> pd.DataFrame:
        """Classement consommation per capita — révèle le déficit africain vs monde"""
        df = self.db.query(metrics=["per_capita_kwh"])
        if df.empty:
            return pd.DataFrame()
        
        latest = (df.sort_values("timestamp")
                    .groupby("country_code")
                    .last()
                    .reset_index()
                    .sort_values("value", ascending=False))
        
        return latest[["country_code", "country_name", "value", "region"]]


# ============================================================
# 5. PIPELINE PRINCIPAL — Orchestration complète
# ============================================================

def run_full_pipeline(rte_token: str = None) -> FerbsDatabase:
    """
    Exécuter le pipeline complet Ferbs Energy Intelligence.
    Lance toutes les sources, normalise, stocke.
    """
    print("=" * 60)
    print("FERBS ENERGY INTELLIGENCE — Pipeline v1.0")
    print("=" * 60)
    
    # Créer les dossiers
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/clean", exist_ok=True)
    
    # Initialiser la base
    db = FerbsDatabase()
    
    # --- SOURCE 1: Our World in Data (Africa) ---
    print("\n[1/4] Chargement OWID — données mondiales...")
    owid = OWIDIngester()
    africa_df = owid.get_africa()
    
    if not africa_df.empty:
        standard_df = owid.to_standard_long(africa_df)
        n = db.insert(standard_df)
        print(f"[OWID] {n} enregistrements insérés")
    
    # --- SOURCE 2: World Bank ---
    print("\n[2/4] World Bank — indicateurs Afrique...")
    africa_iso = ["NGA", "GAB", "ZAF", "KEN", "ETH", "GHA", "CIV", "SEN", "CMR", "COD"]
    
    wb = WorldBankIngester()
    wb_df = wb.fetch_energy_bundle(africa_iso)
    if not wb_df.empty:
        n = db.insert(wb_df)
        print(f"[WorldBank] {n} enregistrements insérés")
    
    # --- SOURCE 3: NERC Nigeria (estimé) ---
    print("\n[3/4] NERC Nigeria — données de production...")
    nerc = NERECIngester()
    nigeria_df = nerc.get_nigerian_data_synthetic()
    n = db.insert(nigeria_df)
    print(f"[NERC] {n} enregistrements insérés")
    
    # --- SOURCE 4: RTE France (si token disponible) ---
    if rte_token:
        print("\n[4/4] RTE France — consommation temps réel...")
        rte = RTEIngester(rte_token)
        end = datetime.now().isoformat() + "+01:00"
        start = (datetime.now() - timedelta(days=30)).isoformat() + "+01:00"
        rte_raw = rte.fetch_consumption(start, end)
        if not rte_raw.empty:
            rte_std = rte.to_standard(rte_raw)
            n = db.insert(rte_std)
            print(f"[RTE] {n} enregistrements insérés")
    else:
        print("\n[4/4] RTE France — token non configuré, skippé")
        print("       → Inscrivez-vous sur data.rte-france.com (gratuit)")
    
    # Résumé
    print("\n" + "=" * 60)
    stats = db.stats()
    print(f"PIPELINE TERMINÉ")
    print(f"Total: {stats['total_rows']} enregistrements")
    print(f"Période: {stats['date_range']}")
    print(f"Pays: {list(stats['by_country'].keys())}")
    print("=" * 60)
    
    return db


# ============================================================
# 6. POINT D'ENTRÉE — Lancer dans Colab
# ============================================================
if __name__ == "__main__":
    # Lance le pipeline complet
    db = run_full_pipeline()
    
    # Exemples d'analyses
    analytics = FerbsAnalytics(db)
    
    print("\n--- GAP D'ACCÈS ÉLECTRICITÉ ---")
    gap = analytics.energy_access_gap()
    if not gap.empty:
        print(gap.to_string(index=False))
    
    print("\n--- VOLATILITÉ NIGERIA ---")
    vol = analytics.demand_volatility("NGA")
    print(vol)
    
    print("\n--- SCORE TRANSITION ÉNERGÉTIQUE ---")
    score = analytics.renewable_transition_score()
    if not score.empty:
        print(score.head(10).to_string(index=False))
