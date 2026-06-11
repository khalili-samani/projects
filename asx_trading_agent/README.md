# ASX Equity Trading Agent

A multi-agent AI system built in Python that fetches ASX market data, forecasts equity prices, assesses portfolio risk, and generates daily trading recommendations using LLM-written briefings.

This project demonstrates agentic AI design, time series forecasting, and financial decision-making applied to Australian equities.

> **Status:** In active development. Agents 1–5 complete. Orchestrator next.

---

## Architecture

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
| 3 | Risk Agent | Complete | Calculate VaR, drawdown, beta, and sector concentration |
| 4 | Trading Strategy Agent | Complete | Generate final signals from forecast + risk combined |
| 5 | Report Writer Agent | Complete | LLM-generated daily briefing via Google Gemini |
| — | Orchestrator | Next | Wire all agents together with LangGraph |

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
WBC.AX    2026-06-11           +1.15%        58% 1.29%    BUY   dist_52w_low
TLS.AX    2026-06-11           +1.07%        69% 0.89%    BUY volatility_20d
ANZ.AX    2026-06-11           +0.61%        65% 1.10%    BUY   dist_52w_low
CBA.AX    2026-06-11           +0.17%        53% 1.31%   HOLD   dist_52w_low
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
ticker vol    var_95 max_drawdown  beta risk_level modifier
WBC.AX 21.5% -2.15%      -28.3%  -0.07     MEDIUM  CONFIRM
TLS.AX 13.2% -1.11%      -18.4%  -0.05        LOW  CONFIRM
CSL.AX 38.1% -2.50%      -31.2%  -0.40       HIGH DOWNGRADE

Portfolio: avg beta -0.06 | avg volatility 24.9% | sector concentration MEDIUM (Financials 40%)
```

---

## Agent 4: Trading Strategy Agent

Takes forecasts from Agent 2 and risk assessments from Agent 3 and produces a final actionable recommendation for each stock.

**What it does:**
- Resolves forecast signals against risk modifiers using a signal resolution table to produce a final BUY / HOLD / REDUCE decision
- Sizes positions based on confidence and risk level — high-confidence, low-risk signals get larger allocations
- Ranks all recommendations by Sharpe score so the highest-opportunity signals are prioritised
- Derives an overall market bias (BULLISH / NEUTRAL / BEARISH) from the balance of signals across the watchlist
- Generates a plain-English rationale for each recommendation, passed directly to the Report Writer

**Signal resolution logic:**

| Forecast | Risk Modifier | Final Signal |
|----------|--------------|--------------|
| BUY | CONFIRM | BUY |
| BUY | DOWNGRADE | HOLD |
| BUY | BLOCK | HOLD |
| REDUCE | CONFIRM | REDUCE |
| REDUCE | DOWNGRADE | HOLD |
| HOLD | anything | HOLD |

**Key design decisions:**
- Resolution table over if/else logic — clean, readable, easy to modify and explain
- Conservative by default — ambiguous signals resolve to HOLD rather than a marginal BUY or REDUCE
- Position sizing is volatility-scaled — higher-risk stocks receive smaller allocations to equalise risk contribution across the portfolio

**Sample output:**
```
ticker forecast_signal modifier final_signal predicted_ret confidence risk_level position
TLS.AX            BUY  CONFIRM          BUY        +1.07%        69%        LOW     FULL
WBC.AX            BUY  CONFIRM          BUY        +1.15%        58%     MEDIUM  REDUCED
ANZ.AX            BUY  CONFIRM          BUY        +0.61%        65%     MEDIUM  REDUCED
CSL.AX            BUY DOWNGRADE        HOLD        +0.88%        26%       HIGH     NONE

Market bias: NEUTRAL | BUY: 4 | HOLD: 6 | REDUCE: 0
```

---

## Agent 5: Report Writer Agent

Takes the full output of all four upstream agents and generates a professional daily trading briefing using an LLM.

**What it does:**
- Assembles macro context, risk summary, watchlist recommendations, and actionable signals into a structured prompt
- Calls Google Gemini to generate a narrative daily briefing
- Saves the report as a markdown file in the `reports/` folder, named by date
- LLM provider is swappable — comments in the code show how to switch to Claude or Groq

**Key design decisions:**
- Structured prompt with explicit section headings — produces consistent output across runs
- All data passed to the LLM comes from upstream agents — the LLM writes, it doesn't decide
- Mandatory disclaimer appended to every report

**Sample output:**
```
ASX MORNING BRIEF — 2026-06-11

MACRO OVERVIEW
The ASX 200 opened lower, down 0.24% to 8604.2, alongside a 0.25% depreciation
in AUD/USD. Gold fell sharply by 3.56% while crude oil surged 2.07%...

ACTIONABLE RECOMMENDATIONS
TLS.AX (BUY): Predicted +1.07% (69% confidence). Low volatility of 13.2% makes
this a low-beta anchor for the portfolio. FULL position recommended...

DISCLAIMER
This report is generated by an automated system for educational purposes only
and does not constitute financial advice.
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
| LLM | `google-genai` (Gemini 2.5 Flash Lite) |
| Agent framework | `langgraph` (orchestrator) |

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
# Add your Gemini API key — free at aistudio.google.com
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

# Agent 4 — generate final recommendations (also re-runs Agents 2 and 3 internally)
python agents/strategy_agent.py

# Agent 5 — generate daily briefing (also re-runs Agents 2, 3 and 4 internally)
python agents/report_writer_agent.py
```

The database file `asx_market_data.db` is created automatically on first run.
Reports are saved to the `reports/` folder as `asx_brief_YYYYMMDD.md`.

---