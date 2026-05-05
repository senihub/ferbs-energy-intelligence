# ⚡ Ferbs Energy Intelligence

> **AI-powered energy data platform for Africa and emerging markets**  
> Predict grid failures. Quantify energy risk. Empower investment decisions.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-MVP%20v0.1-orange)]()
[![Made in Gabon](https://img.shields.io/badge/Made%20in-Gabon%20🇬🇦-009A44)]()

---

## 🌍 The Problem

**600 million people in Sub-Saharan Africa lack reliable electricity.**

- Africa has only **3.1% of global electricity consumption** despite 19% of world population
- Businesses lose **6–16% of revenue** to power outages every year
- Investors pay **2× higher risk premiums** on African energy projects — largely due to lack of reliable data
- **No unified intelligence layer** exists for real-time energy risk in Africa

## 💡 What Ferbs Does

Ferbs aggregates heterogeneous energy data (APIs, CSV, scraping) from Africa and Europe, normalizes it into a unified schema, and applies machine learning to:

- **Predict grid failures** 6–48h in advance
- **Score energy risk** per country and region
- **Generate investment intelligence** for funds and DFIs investing in African energy infrastructure

---

## 🏗️ Architecture

```
Data Sources          Ingestion Layer        Storage         Analytics
─────────────         ───────────────        ───────         ─────────
OWID Energy     ──►   OWIDIngester     ──►              ──►  Access Gap
World Bank API  ──►   WorldBankIngester ──►  SQLite DB  ──►  Volatility Score  
NERC Nigeria    ──►   NERECIngester    ──►              ──►  Transition Score
RTE France      ──►   RTEIngester      ──►              ──►  Per Capita Gap
```

**Unified schema:** `timestamp | country_code | metric | value | unit | source | quality_flag`

---

## 📁 Repository Structure

```
ferbs-energy-intelligence/
│
├── ferbs_pipeline.py      # Core data pipeline (ingestion + storage + analytics)
├── ferbs_dashboard.py     # Plotly visualizations and dashboard export
├── ferbs_colab_guide.py   # Step-by-step Google Colab notebook
│
├── data/
│   ├── raw/               # Raw data files (gitignored)
│   └── clean/             # Normalized data (gitignored)
│
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md
```

---

## 🚀 Quick Start

### Option A — Google Colab (recommended, no install needed)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `ferbs_pipeline.py` and `ferbs_dashboard.py`
3. Run in order:

```python
# Cell 1 — Install dependencies
!pip install requests pandas numpy plotly sqlalchemy beautifulsoup4 lxml openpyxl -q

# Cell 2 — Run full pipeline
from ferbs_pipeline import run_full_pipeline
db = run_full_pipeline()

# Cell 3 — Launch dashboard
from ferbs_dashboard import FerbsDashboard
dash = FerbsDashboard(db)
dash.show_all()
```

### Option B — Local Python

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ferbs-energy-intelligence.git
cd ferbs-energy-intelligence

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python ferbs_pipeline.py
```

---

## 📊 Data Sources

| Source | Coverage | Type | API Key Required |
|--------|----------|------|-----------------|
| [Our World in Data](https://github.com/owid/energy-data) | 190 countries, 1900–2023 | CSV (GitHub) | ❌ No |
| [World Bank Open Data](https://data.worldbank.org) | 200+ countries | REST API | ❌ No |
| [RTE France](https://data.rte-france.com) | France, 30-min intervals | REST API | ✅ Free registration |
| NERC Nigeria | Nigeria, monthly | Estimated model | ❌ No |

### Key Metrics Available

- `access_pct` — % population with electricity access
- `per_capita_kwh` — electricity consumption per capita (kWh/year)
- `renewables_pct` — share of renewable electricity (%)
- `generation_mw` — grid generation capacity (MW)
- `demand_mw` — electricity demand (MW)

---

## 🔑 Optional: RTE France API Setup

```python
# 1. Register free at https://data.rte-france.com
# 2. Create an application and get your OAuth credentials
# 3. Run with token:

from ferbs_pipeline import run_full_pipeline
db = run_full_pipeline(rte_token="YOUR_TOKEN_HERE")
```

---

## 📈 Analytics & Key Metrics

```python
from ferbs_pipeline import run_full_pipeline, FerbsAnalytics

db  = run_full_pipeline()
analytics = FerbsAnalytics(db)

# Energy access gap by country
gap = analytics.energy_access_gap()

# Grid volatility score (higher = more unstable)
vol = analytics.demand_volatility("NGA")  # Nigeria

# Renewable transition score
score = analytics.renewable_transition_score()

# Per capita consumption ranking
ranking = analytics.per_capita_energy_ranking()
```

---

## 🗺️ Roadmap

- [x] Multi-source data ingestion (OWID, World Bank, RTE, NERC)
- [x] Unified normalized schema
- [x] SQLite storage with deduplication
- [x] Core analytics (access gap, volatility, transition score)
- [x] Plotly interactive dashboard
- [ ] **v0.2** — XGBoost outage prediction model (24–48h forecast)
- [ ] **v0.3** — Anomaly detection with Isolation Forest
- [ ] **v0.4** — FastAPI REST endpoint for risk scores
- [ ] **v0.5** — Real-time pipeline (ENTSO-E + RTE streaming)
- [ ] **v1.0** — Deployable SaaS dashboard for utilities

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Data manipulation | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost *(coming v0.2)* |
| Visualization | Plotly |
| Storage | SQLite → PostgreSQL *(production)* |
| Cloud | Google Colab / AWS |
| API *(coming)* | FastAPI |

---

## 🤝 Contributing

This project is in active early development. Contributions welcome:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add: your feature'`
4. Push and open a Pull Request

Areas where help is needed:
- Additional African data sources (SNEL DRC, ECG Ghana, ENEO Cameroon)
- PDF scraping for NERC Nigeria monthly reports
- ML model for outage prediction

---

## 👤 Author

**Ferbs** — Founder, Ferbs Energy Intelligence  
🇬🇦 Libreville, Gabon  
Building the intelligence layer for African energy markets.

> *"Africa needs someone building this from the inside — not from San Francisco."*

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ⭐ If this project is useful to you

Give it a star on GitHub — it helps the project get visibility in the African tech ecosystem.
