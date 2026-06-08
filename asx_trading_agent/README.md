# ASX Equity Trading Agent

A multi-agent AI system built in Python that fetches ASX market data, forecasts equity prices, assesses portfolio risk, and generates daily trading recommendations using LLM-written briefings.

This project demonstrates agentic AI design, time series forecasting, and financial decision-making applied to Australian equities.

> **Status:** In active development. Agents 1 and 2 complete.

---

## Planned Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Orchestrator                      │
│                    (LangGraph)                       │
└──────┬──────────┬──────────┬──────────┬─────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
  Market Data  Forecast    Risk     Strategy    Report
    Agent       Agent      Agent     Agent      Writer
       │          │          │          │         │
       └──────────┴──────────┴──────────┴─────────┘
                            │
                     DuckDB (local)
                   asx_market_data.db
```

Each agent has a single responsibility and exposes a `run()` method. The orchestrator calls them in sequence and passes outputs downstream.

---

## Agents

| # | Agent | Status | Responsibility |
|---|-------|--------|----------------|
| 1 | Market Data Agent | Complete | Fetch and store ASX price, fundamental, and macro data |
| 2 | Forecasting Agent | Complete | Predict next-day returns for each watchlist stock |
| 3 | Risk Agent | Next | Calculate VaR, drawdown, beta, and position sizing |
| 4 | Trading Strategy Agent | Planned | Generate buy/hold/reduce signals from forecast + risk |
| 5 | Report Writer Agent | Planned | LLM-generated daily briefing |
| — | Orchestrator | Planned | Wire all agents together with LangGraph |

---

## Agent 1: Market Data Agent

The foundation of the system. Fetches, validates, and stores all market data that downstream agents depend on.

**What it does:**
- Downloads 12 months of OHLCV price history for 10 ASX stocks via `yfinance`
- Fetches fundamental snapshots per stock (market cap, P/E ratio, beta, sector)
- Fetches macro context: ASX 200 index, AUD/USD, Gold, and Crude Oil
- Validates all incoming data with Pydantic — suspicious rows are skipped, not stored
- Persists everything to a local DuckDB database across three tables

**Database schema:**

| Table | Contents |
|-------|----------|
| `ohlcv` | Daily open, high, low, close, volume, and return per ticker |
| `fundamentals` | Timestamped snapshot of each stock's fundamental data |
| `macro` | Daily close and return for ASX 200, AUD/USD, Gold, and Oil |

---

## Agent 2: Forecasting Agent

Reads OHLCV and macro data from the database, engineers features, trains a model per stock, and predicts next-day returns.

**What it does:**
- Engineers 11 features from price and volume history — momentum, volatility, RSI, volume ratio, mean reversion signals, and macro context
- Trains a Random Forest Regressor per stock using walk-forward validation to avoid lookahead bias
- Predicts the next trading day's return for each watchlist stock
- Derives a confidence score from the spread of individual tree predictions
- Outputs a BUY / HOLD / REDUCE signal and identifies the top driving feature

**Key design decisions:**
- Walk-forward train/validate split (80/20, chronological) — never shuffled, never peeked at future data
- One model per stock — each stock has different dynamics and shouldn't share a model
- Confidence derived from tree disagreement — low confidence flags unreliable predictions rather than hiding them

**Sample output:**
```
ticker forecast_date predicted_return confidence   mae signal    top_feature
WBC.AX    2026-06-05           -0.35%        46% 1.25%   HOLD   dist_52w_low
CSL.AX    2026-06-05           +1.05%        12% 1.56%    BUY  asx200_return
TLS.AX    2026-06-05           -0.09%        63% 0.82%   HOLD volatility_20d
CBA.AX    2026-06-05           -0.70%        55% 1.36% REDUCE   volume_ratio
```

---

## Watchlist

| Ticker | Company | Sector |
|--------|---------|--------|
| BHP.AX | BHP Group | Materials |
| CBA.AX | Commonwealth Bank | Financials |
| CSL.AX | CSL Limited | Healthcare |
| WDS.AX | Woodside Energy | Energy |
| WES.AX | Wesfarmers | Consumer Discretionary |
| ANZ.AX | ANZ Banking Group | Financials |
| RIO.AX | Rio Tinto | Materials |
| WBC.AX | Westpac Banking | Financials |
| MQG.AX | Macquarie Group | Financials |
| TLS.AX | Telstra | Telecommunications |

---

## Tech Stack

| Purpose | Library |
|---------|---------|
| Market data | `yfinance` |
| Database | `duckdb` |
| Data processing | `pandas`, `numpy` |
| Data validation | `pydantic` |
| Forecasting | `scikit-learn` |
| LLM orchestration (Agent 5) | TBD |
| Agent framework | `langgraph` |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/asx-trading-agent.git
cd asx-trading-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# LLM API key required for Agent 5 only — not needed yet
```

---

## Running the Agents

**Agent 1 — fetch and store market data:**
```bash
python agents/market_data_agent.py
```

**Agent 2 — forecast next-day returns:**
```bash
python agents/forecasting_agent.py
```

Agent 1 must be run before Agent 2. The database file `asx_market_data.db` is created automatically on first run.

---