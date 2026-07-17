# ASX Equity Intelligence Agent

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A multi-agent system that automates daily ASX equity analysis end to end: collecting market data, forecasting next-day returns, assessing portfolio risk, generating trading recommendations, and producing a written market briefing via an LLM. It combines data engineering, machine learning, risk analytics and agent orchestration into a single automated pipeline.

---

## Overview

Producing a daily market brief typically means an analyst manually pulling data, running forecasts, checking risk exposure, and writing up recommendations for stakeholders. It's repetitive, slow to scale, and hard to keep consistent across a watchlist.

This project replaces that manual workflow with five specialised agents, each responsible for one stage of the analysis, orchestrated through LangGraph. The goal wasn't to build another stock predictor, but a decision-support pipeline that mirrors how a quant research or investment team would actually operate: forecasting, risk-checking, recommending, and reporting, with clear separation of responsibility between stages.

---

## Key Features

- **Automated data collection** for a 10-stock ASX watchlist plus macro indicators (ASX 200, AUD/USD, gold, crude oil), stored in DuckDB
- **Per-stock forecasting** using Random Forest models with walk-forward validation to avoid look-ahead bias
- **Portfolio risk assessment**: historical VaR, annualised volatility, beta, drawdown, and sector concentration
- **Rule-based recommendation engine** producing BUY / HOLD / REDUCE signals, conservative by design (defaults to HOLD under high uncertainty)
- **LLM-generated market briefing** (Gemini 2.5 Flash Lite) that converts structured outputs into readable commentary, constrained to prevent it from altering numbers or recommendations
- **Scheduled execution** ahead of ASX market open

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Database | DuckDB |
| Machine learning | Scikit-learn (Random Forest) |
| Workflow orchestration | LangGraph |
| Market data | yfinance |
| LLM reporting | Gemini 2.5 Flash Lite |
| Scheduling | `schedule` |
| Config | python-dotenv |

---

## Architecture

```text
Market Data Agent
        │
        ▼
Forecasting Agent
        │
        ▼
Risk Agent
        │
        ▼
Strategy Agent
        │
        ▼
Report Writer Agent
        │
        ▼
ASX Morning Brief
```

| Stage | Agent | Responsibility |
|---|---|---|
| 1 | Market Data Agent | Collect OHLCV and macro data, store in DuckDB |
| 2 | Forecasting Agent | Predict next-day returns per stock, estimate confidence |
| 3 | Risk Agent | Assess stock and portfolio-level risk |
| 4 | Strategy Agent | Combine forecasts and risk into a final signal |
| 5 | Report Writer Agent | Turn structured output into a market brief |

Each agent passes structured output to the next, which keeps the pipeline explainable and makes it straightforward to swap out or extend individual stages.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment variables**

Create a `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

**3. Run the full pipeline**
```bash
python orchestrator.py --mode run
```

**4. Or schedule daily execution** (before ASX market open)
```bash
python orchestrator.py --mode schedule
```

**Outputs:**

| Output | Location |
|---|---|
| Market database | `asx_market_data.db` |
| Daily report | `reports/asx_brief_YYYYMMDD.md` |

---

## Results: Sample Pipeline Run (23 June 2026)

A full end-to-end run completed successfully with the following output:

| Stage | Result |
|---|---|
| Market data collection | 10/10 tickers succeeded, 2,540 OHLCV records loaded |
| Forecasting | 10 forecasts generated |
| Risk assessment | 10 stocks assessed |
| Recommendations | 1 BUY, 8 HOLD, 1 REDUCE |
| Overall market bias | NEUTRAL |
| Report generation | Completed successfully |

**Portfolio risk snapshot:**

| Metric | Value |
|---|---|
| Average beta | 0.84 |
| Average annualised volatility | 25.0% |
| Average 1-day VaR (95%) | -2.28% |
| Largest sector exposure | Financial Services (40%) |
| Concentration risk | Medium |

**Sample recommendations:**

| Ticker | Signal | Predicted Return | Confidence | Risk Level |
|---|---|---:|---:|---|
| CBA.AX | BUY | +0.54% | 60% | Medium |
| MQG.AX | REDUCE | -0.68% | 64% | Medium |

![Pipeline Execution](screenshots/pipeline_execution.png)
![Generated Report](screenshots/final_report.png)

---

## LLM Design Decisions

Gemini 2.5 Flash Lite is used only for the reporting layer, not for forecasting, risk modelling, or recommendations. It receives structured outputs from the pipeline and converts them into commentary. The prompt explicitly restricts the model to using only supplied data and preserving numbers, signals and confidence scores exactly, which keeps the analytical output auditable and reduces hallucination risk.

---

## Skills Demonstrated

- **Data engineering:** financial data ingestion, DuckDB schema design, automated pipelines, data validation
- **Machine learning:** time-series forecasting, feature engineering, walk-forward validation, confidence estimation
- **Risk analytics:** VaR, volatility, beta, drawdown, sector concentration analysis
- **Agentic AI / orchestration:** multi-agent design with LangGraph, shared state, conditional routing
- **LLM engineering:** prompt design for constrained, hallucination-resistant reporting

---

## Watchlist

| Ticker | Company | Sector |
|---|---|---|
| BHP.AX | BHP Group | Materials |
| CBA.AX | Commonwealth Bank | Financial Services |
| CSL.AX | CSL Limited | Healthcare |
| WDS.AX | Woodside Energy | Energy |
| WES.AX | Wesfarmers | Consumer Discretionary |
| ANZ.AX | ANZ Banking Group | Financial Services |
| RIO.AX | Rio Tinto | Materials |
| WBC.AX | Westpac Banking Corporation | Financial Services |
| MQG.AX | Macquarie Group | Financial Services |
| TLS.AX | Telstra Corporation | Telecommunications |

---

## Limitations

- Uses yfinance rather than an institutional market data feed
- Forecasting models are intentionally simple (no deep learning, no ensembling across models)
- No transaction cost or slippage modelling
- No live order execution or portfolio optimisation
- No historical backtesting framework, so performance has not been validated over time (see note below)

This is an educational portfolio project, not a production trading system.

---

## Future Improvements

- Backtesting engine to validate forecast accuracy over time
- Portfolio optimisation
- Streamlit or Power BI dashboard
- Benchmark comparison reporting
- Automated email delivery
- Model performance monitoring and feature importance visualisations

---

## License

MIT License, see [LICENSE](LICENSE) for details.