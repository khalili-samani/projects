# ASX Equity Trading Agent

A multi-agent AI system built in Python that fetches ASX market data, forecasts equity prices, assesses portfolio risk, and generates daily trading recommendations using LLM-written briefings.

This project demonstrates agentic AI design, time series forecasting, and financial decision-making applied to Australian equities.

> **Status:** In active development. Agent 1 (Market Data) is complete.

---

## Planned Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Orchestrator                     │
│                    (LangGraph)                      │
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
| 2 | Forecasting Agent | Next | Predict next-day returns for each watchlist stock |
| 3 | Risk Agent | Planned | Calculate VaR, drawdown, beta, and position sizing |
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
| Forecasting (Agent 2) | `scikit-learn`, `statsmodels` |
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

## Running Agent 1

```bash
python agents/market_data_agent.py
```

Expected output:

```
MarketDataAgent | INFO | Database initialised at asx_market_data.db
MarketDataAgent | INFO | Fetching OHLCV for BHP.AX...
MarketDataAgent | INFO |   ✓ BHP.AX: 251 records
...
MarketDataAgent | INFO | Run complete.
MarketDataAgent | INFO |   Successful tickers : 10
MarketDataAgent | INFO |   Total OHLCV records: 2510

Watchlist Summary:
ticker                           name        date   close daily_return
BHP.AX           BHP Group (Materials)  2026-06-01   62.48       +0.27%
CBA.AX   Commonwealth Bank (Financials) 2026-06-01  172.14       +0.45%
...
```

---

## Skills Demonstrated

- Multi-agent system design
- Financial data engineering (OHLCV, fundamentals, macro context)
- Data validation and quality control with Pydantic
- Time series storage with DuckDB
- ASX market structure and Australian equities
- Production-grade Python (structured logging, modular design, error handling)