# ASX Equity Trading Agent

A multi-agent AI system built in Python that fetches ASX market data, forecasts equity prices, assesses portfolio risk, and generates daily trading recommendations using LLM-written briefings.

This project demonstrates agentic AI design, time series forecasting, and financial decision-making applied to Australian equities.

> **Status:** In active development. Agents 1, 2 and 3 complete.

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
| 2 | Forecasting Agent | Complete | Predict next-day returns for each watchlist stock |
| 3 | Risk Agent | Complete | Calculate VaR, drawdown, beta, and sector concentration |
| 4 | Trading Strategy Agent | Next | Generate final signals from forecast + risk combined |
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

## Agent 3: Risk Agent

Takes forecasts from Agent 2 and assesses the risk profile of each stock and the overall watchlist portfolio.

**What it does:**
- Calculates historical Value at Risk (VaR) at 95% and 99% confidence — using actual return distributions rather than assuming normality
- Measures maximum drawdown over the full price history
- Calculates beta against the ASX 200 index
- Derives annualised volatility and a risk-adjusted Sharpe score
- Assesses sector concentration across the watchlist
- Outputs a recommendation modifier (CONFIRM / DOWNGRADE / BLOCK) that the Strategy Agent uses to filter signals

**Key design decisions:**
- Historical (non-parametric) VaR — stock returns have fat tails, so assuming a normal distribution underestimates real-world risk
- Recommendation modifier bridges risk assessment to strategy — a BUY signal with HIGH risk and low confidence gets downgraded before it reaches the Strategy Agent
- Portfolio-level risk assessed separately from individual stocks — sector concentration is a portfolio problem, not a stock problem

**Sample output:**
```
ticker signal predicted_ret confidence volatility  var_95 max_drawdown  beta risk_level modifier sharpe_score
CSL.AX    BUY        +1.05%        12%     18.2%  -1.14%      -22.31%  0.15        LOW  DOWNGRADE        0.021
TLS.AX   HOLD        -0.09%        63%     14.8%  -0.93%      -18.44%  0.13        LOW    CONFIRM       -0.027
CBA.AX REDUCE        -0.70%        55%     22.1%  -1.38%      -28.54%  0.83     MEDIUM    CONFIRM       -0.142

Portfolio: avg beta 0.59 | avg volatility 24.3% | sector concentration HIGH (Financials 40%)
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

Agents must be run in order — each depends on the output of the previous.

```bash
# Agent 1 — fetch and store market data
python agents/market_data_agent.py

# Agent 2 — forecast next-day returns
python agents/forecasting_agent.py

# Agent 3 — assess risk (also re-runs Agent 2 internally)
python agents/risk_agent.py
```

The database file `asx_market_data.db` is created automatically on first run.

---