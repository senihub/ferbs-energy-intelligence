"""
FERBS ENERGY INTELLIGENCE — EIA Data Ingester
==============================================
Source : U.S. Energy Information Administration (eia.gov)
Données : Prix pétrole, gaz naturel, électricité — mondial
Clé API : Gratuite sur https://www.eia.gov/opendata/

USAGE :
    import os
    from ferbs_eia import EIAIngester
    
    ingester = EIAIngester(api_key=os.environ.get("EIA_API_KEY"))
    df = ingester.fetch_all()
"""

import os
import requests
import pandas as pd
from datetime import datetime

class EIAIngester:
    """
    Récupère les prix des matières premières énergétiques via l'API EIA v2.
    Données clés pour Ferbs :
    - Prix pétrole brut (WTI + Brent)
    - Prix gaz naturel (Henry Hub)
    - Prix électricité (marchés USA comme référence mondiale)
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("EIA_API_KEY")
        self.base_url = "https://api.eia.gov/v2"

        if not self.api_key:
            raise ValueError(
                "Clé EIA manquante. "
                "Inscris-toi sur https://www.eia.gov/opendata/ "
                "puis ajoute EIA_API_KEY dans tes variables d'environnement Render."
            )

    def _get(self, endpoint: str, params: dict) -> dict:
        """Appel API générique avec gestion d'erreur"""
        params["api_key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            print(f"[EIA] Timeout sur {endpoint}")
            return {}
        except requests.exceptions.HTTPError as e:
            print(f"[EIA] Erreur HTTP {e.response.status_code} sur {endpoint}")
            return {}
        except Exception as e:
            print(f"[EIA] Erreur inattendue: {e}")
            return {}

    # ------------------------------------------------------------------
    # PÉTROLE BRUT
    # ------------------------------------------------------------------
    def fetch_crude_oil_prices(self, frequency: str = "monthly") -> pd.DataFrame:
        """
        Prix pétrole brut — WTI et Brent
        WTI  = West Texas Intermediate (référence USA)
        Brent = référence internationale (plus pertinent pour Afrique)
        frequency: 'daily', 'weekly', 'monthly'
        """
        print("[EIA] Récupération prix pétrole brut...")

        data = self._get("petroleum/pri/spt/data", {
            "frequency": frequency,
            "data[0]": "value",
            "facets[product][]": ["EPCBRENT", "EPCWTI"],
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 120,  # 10 ans de données mensuelles
            "offset": 0,
        })

        records = []
        for item in data.get("response", {}).get("data", []):
            records.append({
                "timestamp":    pd.Timestamp(item["period"] + "-01", tz="UTC")
                               if len(item["period"]) == 7
                               else pd.Timestamp(item["period"], tz="UTC"),
                "country_code": "GLOBAL",
                "country_name": "Global",
                "region":       "global",
                "metric":       f"oil_price_{item.get('product', 'crude').lower()}",
                "value":        float(item["value"]) if item.get("value") else None,
                "unit":         "USD_per_barrel",
                "source":       "eia",
                "quality_flag": "ok",
            })

        df = pd.DataFrame(records).dropna(subset=["value"])
        print(f"[EIA] {len(df)} enregistrements pétrole récupérés")
        return df

    # ------------------------------------------------------------------
    # GAZ NATUREL
    # ------------------------------------------------------------------
    def fetch_natural_gas_prices(self) -> pd.DataFrame:
        """
        Prix gaz naturel — Henry Hub (référence mondiale)
        Pertinent pour l'Afrique : Nigeria, Mozambique, Tanzanie
        sont de gros producteurs de gaz
        """
        print("[EIA] Récupération prix gaz naturel...")

        data = self._get("natural-gas/pri/fut/data", {
            "frequency": "monthly",
            "data[0]": "value",
            "facets[series][]": ["RNGC1"],  # Henry Hub Natural Gas Futures
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 120,
            "offset": 0,
        })

        records = []
        for item in data.get("response", {}).get("data", []):
            records.append({
                "timestamp":    pd.Timestamp(item["period"] + "-01", tz="UTC")
                               if len(item["period"]) == 7
                               else pd.Timestamp(item["period"], tz="UTC"),
                "country_code": "GLOBAL",
                "country_name": "Global",
                "region":       "global",
                "metric":       "gas_price_henry_hub",
                "value":        float(item["value"]) if item.get("value") else None,
                "unit":         "USD_per_mmbtu",
                "source":       "eia",
                "quality_flag": "ok",
            })

        df = pd.DataFrame(records).dropna(subset=["value"])
        print(f"[EIA] {len(df)} enregistrements gaz récupérés")
        return df

    # ------------------------------------------------------------------
    # PRODUCTION PÉTROLE AFRIQUE
    # ------------------------------------------------------------------
    def fetch_africa_oil_production(self) -> pd.DataFrame:
        """
        Production pétrolière par pays africain
        Clé pour Ferbs : corréler production pétrolière
        avec stabilité du réseau électrique
        """
        print("[EIA] Récupération production pétrole Afrique...")

        # Pays africains producteurs de pétrole (codes EIA)
        african_producers = {
            "NG": ("NGA", "Nigeria"),
            "GA": ("GAB", "Gabon"),
            "AO": ("AGO", "Angola"),
            "LY": ("LBY", "Libya"),
            "DZ": ("DZA", "Algeria"),
            "GQ": ("GNQ", "Equatorial Guinea"),
            "CG": ("COG", "Congo"),
        }

        data = self._get("petroleum/supply/monthly/data", {
            "frequency": "monthly",
            "data[0]": "value",
            "facets[countryRegionId][]": list(african_producers.keys()),
            "facets[productId][]": ["53"],  # Crude oil production
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 200,
            "offset": 0,
        })

        records = []
        for item in data.get("response", {}).get("data", []):
            country_eia = item.get("countryRegionId", "")
            iso, name = african_producers.get(country_eia, ("UNK", "Unknown"))

            records.append({
                "timestamp":    pd.Timestamp(item["period"] + "-01", tz="UTC")
                               if len(item["period"]) == 7
                               else pd.Timestamp(item["period"], tz="UTC"),
                "country_code": iso,
                "country_name": name,
                "region":       "africa",
                "metric":       "oil_production_kbd",
                "value":        float(item["value"]) if item.get("value") else None,
                "unit":         "thousand_barrels_per_day",
                "source":       "eia",
                "quality_flag": "ok",
            })

        df = pd.DataFrame(records).dropna(subset=["value"])
        print(f"[EIA] {len(df)} enregistrements production africaine récupérés")
        return df

    # ------------------------------------------------------------------
    # PIPELINE COMPLET EIA
    # ------------------------------------------------------------------
    def fetch_all(self) -> pd.DataFrame:
        """
        Récupérer toutes les données EIA et les fusionner
        dans le schéma Ferbs unifié
        """
        print("\n[EIA] Lancement pipeline complet...")
        print("=" * 50)

        all_dfs = []

        # 1. Prix pétrole
        oil_df = self.fetch_crude_oil_prices()
        if not oil_df.empty:
            all_dfs.append(oil_df)

        # 2. Prix gaz
        gas_df = self.fetch_natural_gas_prices()
        if not gas_df.empty:
            all_dfs.append(gas_df)

        # 3. Production africaine
        prod_df = self.fetch_africa_oil_production()
        if not prod_df.empty:
            all_dfs.append(prod_df)

        if not all_dfs:
            print("[EIA] Aucune donnée récupérée — vérifier la clé API")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n[EIA] Total : {len(combined)} enregistrements")
        print(f"[EIA] Métriques : {combined['metric'].unique().tolist()}")
        print("=" * 50)

        return combined


# ------------------------------------------------------------------
# ANALYSE CORRÉLATION — Prix pétrole vs Stabilité réseau
# ------------------------------------------------------------------
def analyze_oil_grid_correlation(db, eia_df: pd.DataFrame) -> dict:
    """
    Insight clé de Ferbs : les pays producteurs de pétrole
    ont-ils un réseau plus stable quand les prix sont élevés ?
    
    Hypothèse : prix pétrole élevé → revenus État → investissement réseau
    Réalité africaine : souvent l'inverse (Dutch disease)
    """
    if eia_df.empty:
        return {}

    # Prix Brent mensuel
    brent = eia_df[eia_df["metric"] == "oil_price_epcbrent"][["timestamp", "value"]].copy()
    brent.columns = ["timestamp", "brent_price"]
    brent["timestamp"] = pd.to_datetime(brent["timestamp"], utc=True)
    brent = brent.sort_values("timestamp")

    # Production Nigeria
    nigeria_prod = eia_df[
        (eia_df["country_code"] == "NGA") &
        (eia_df["metric"] == "oil_production_kbd")
    ][["timestamp", "value"]].copy()
    nigeria_prod.columns = ["timestamp", "nigeria_production"]
    nigeria_prod["timestamp"] = pd.to_datetime(nigeria_prod["timestamp"], utc=True)

    if brent.empty or nigeria_prod.empty:
        return {"error": "Données insuffisantes pour la corrélation"}

    # Fusion
    merged = brent.merge(nigeria_prod, on="timestamp", how="inner")

    if len(merged) < 5:
        return {"error": "Pas assez de points communs"}

    # Corrélation simple
    correlation = merged["brent_price"].corr(merged["nigeria_production"])

    return {
        "insight": "Corrélation prix Brent vs production Nigeria",
        "correlation": round(correlation, 3),
        "interpretation": (
            "Forte corrélation positive — Nigeria produit plus quand les prix sont hauts"
            if correlation > 0.5
            else "Faible corrélation — la production nigériane est indépendante des prix"
            if correlation < 0.2
            else "Corrélation modérée"
        ),
        "n_points": len(merged),
        "brent_mean": round(merged["brent_price"].mean(), 2),
        "nigeria_prod_mean": round(merged["nigeria_production"].mean(), 1),
    }


# ------------------------------------------------------------------
# INTÉGRATION DANS LE PIPELINE FERBS PRINCIPAL
# ------------------------------------------------------------------
"""
COMMENT INTÉGRER DANS ferbs_pipeline.py :

Ajoute ces lignes dans la fonction run_full_pipeline() :

    # --- SOURCE EIA ---
    print("\\n[EIA] Récupération prix matières premières...")
    try:
        from ferbs_eia import EIAIngester
        eia = EIAIngester()  # Lit EIA_API_KEY depuis l'environnement
        eia_df = eia.fetch_all()
        if not eia_df.empty:
            n = db.insert(eia_df)
            print(f"[EIA] {n} enregistrements insérés")
    except Exception as e:
        print(f"[EIA] Skippé: {e}")
"""


# ------------------------------------------------------------------
# TEST RAPIDE DANS COLAB
# ------------------------------------------------------------------
"""
Dans Google Colab :

import os
os.environ["EIA_API_KEY"] = "ta_clé_eia_ici"  # Seulement pour tester

from ferbs_eia import EIAIngester, analyze_oil_grid_correlation
from ferbs_pipeline import run_full_pipeline

# Test EIA seul
eia = EIAIngester()
df = eia.fetch_all()
print(df.head(20))
print(df["metric"].value_counts())

# Pipeline complet avec EIA
db = run_full_pipeline()
correlation = analyze_oil_grid_correlation(db, df)
print(correlation)
"""
