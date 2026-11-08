# ASX Equity Intelligence Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A Python workflow that collects Australian equity market data, forecasts next-session returns, calculates risk measures, applies deterministic recommendation rules, and generates a daily Markdown briefing.

The project is designed as a decision-support and engineering demonstration. It does not execute trades, optimise a live portfolio, or establish that the forecasts are profitable.

## Project summary

An equity research workflow often requires several separate tasks:

1. collecting and validating market data;
2. engineering predictive features;
3. evaluating forecast uncertainty and downside risk;
4. translating model output into consistent decisions; and
5. preparing a readable briefing.

This project separates those responsibilities into five LangGraph nodes with a shared typed state. Each analytical stage produces structured output for the next stage. Gemini is used only after forecasts, risk metrics and recommendations have already been calculated.

### What the project demonstrates

* ingestion and validation of financial time-series data;
* local analytical storage with DuckDB;
* time-ordered model training and validation;
* feature engineering for tabular market data;
* historical volatility, VaR, beta and drawdown calculations;
* deterministic decision rules and position-sizing logic;
* LangGraph state management and conditional failure routing;
* constrained LLM-based report generation.

## Workflow

```text
                         ┌──────────────────┐
                         │ Market Data Node │
                         └────────┬─────────┘
                                  │ data available
                                  ▼
                         ┌──────────────────┐
                         │ Forecasting Node │
                         └────────┬─────────┘
                                  │ forecasts available
                                  ▼
                         ┌──────────────────┐
                         │    Risk Node     │
                         └────────┬─────────┘
                                  │ risk output available
                                  ▼
                         ┌──────────────────┐
                         │  Strategy Node   │
                         └────────┬─────────┘
                                  │ recommendations available
                                  ▼
                         ┌──────────────────┐
                         │Report Writer Node│
                         └────────┬─────────┘
                                  ▼
                         Markdown morning brief

Any critical stage failure ───────────────► Error node ─► End
```

| Stage          | Implementation                  | Responsibility                                                                       |
| -------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| Market data    | `agents/market_data_agent.py`   | Downloads equity and macro data, validates records and writes them to DuckDB         |
| Forecasting    | `agents/forecasting_agent.py`   | Engineers features, trains one Random Forest per ticker and predicts the next return |
| Risk           | `agents/risk_agent.py`          | Calculates stock-level and watchlist-level risk measures                             |
| Strategy       | `agents/strategy_agent.py`      | Converts forecasts and risk modifiers into BUY, HOLD or REDUCE recommendations       |
| Report writing | `agents/report_writer_agent.py` | Supplies structured results to Gemini and saves a Markdown briefing                  |
| Orchestration  | `orchestrator.py`               | Defines shared state, graph routing, execution and application-level scheduling      |

LangGraph is used for explicit state transitions and conditional routing. This is primarily a structured multi-stage workflow rather than a collection of autonomous agents.

## Data

### Equity watchlist

The default watchlist contains ten ASX-listed securities.

| Ticker   | Company           | Sector                 |
| -------- | ----------------- | ---------------------- |
| `BHP.AX` | BHP Group         | Materials              |
| `CBA.AX` | Commonwealth Bank | Financials             |
| `CSL.AX` | CSL Limited       | Healthcare             |
| `WDS.AX` | Woodside Energy   | Energy                 |
| `WES.AX` | Wesfarmers        | Consumer Discretionary |
| `ANZ.AX` | ANZ Banking Group | Financials             |
| `RIO.AX` | Rio Tinto         | Materials              |
| `WBC.AX` | Westpac Banking   | Financials             |
| `MQG.AX` | Macquarie Group   | Financials             |
| `TLS.AX` | Telstra           | Telecommunications     |

### Macro series

| Symbol     | Series                |
| ---------- | --------------------- |
| `^AXJO`    | ASX 200 Index         |
| `AUDUSD=X` | AUD/USD exchange rate |
| `GC=F`     | Gold futures          |
| `CL=F`     | WTI crude oil futures |

Data is downloaded with `yfinance`. Equity prices use `auto_adjust=True`, so the stored OHLC values are adjusted for corporate actions handled by the provider.

The default ingestion window is the previous 365 calendar days. Because market data contains non-trading days, the resulting number of observations is lower than 365.

### Validation and storage

Incoming equity records are validated with Pydantic before insertion:

* open, high, low and close prices must be positive;
* daily returns outside ±50% are rejected as suspicious;
* records are keyed by ticker and date;
* repeated runs use `INSERT OR REPLACE`.

DuckDB contains three tables:

| Table          | Contents                                               |
| -------------- | ------------------------------------------------------ |
| `ohlcv`        | Adjusted equity OHLCV data and daily returns           |
| `macro`        | Closing values and daily returns for macro series      |
| `fundamentals` | Timestamped company metadata and fundamental snapshots |

The supplied database contains 2,630 equity OHLCV rows, with 263 observations for each watchlist ticker from 10 June 2025 to 22 June 2026.

## Forecasting approach

A separate `RandomForestRegressor` is trained for each stock. The prediction target is the following observation’s daily return:

```python
target = daily_return.shift(-1)
```

### Features

The model uses eleven engineered features.

| Feature            | Definition                                                                     |
| ------------------ | ------------------------------------------------------------------------------ |
| `return_5d`        | Five-session percentage price change                                           |
| `return_10d`       | Ten-session percentage price change                                            |
| `return_20d`       | Twenty-session percentage price change                                         |
| `volatility_20d`   | Rolling 20-session standard deviation of daily returns                         |
| `rsi_14`           | Fourteen-session Relative Strength Index                                       |
| `volume_ratio`     | Current volume divided by the 20-session average                               |
| `dist_52w_high`    | Distance from the rolling 252-session high                                     |
| `dist_52w_low`     | Distance from the rolling 252-session low                                      |
| `ma_crossover`     | Ten-session moving average divided by the 50-session moving average, minus one |
| `asx200_return`    | ASX 200 daily return                                                           |
| `asx200_return_5d` | Five-session rolling mean of ASX 200 returns                                   |

Rows with missing features or targets are excluded. At least 100 usable rows are required for a ticker to be modelled.

### Model configuration

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
)
```

Features are standardised before training. Scaling is not required for Random Forests, but the implementation applies the same fitted transformation to training and validation data.

### Validation design

The observations remain in chronological order:

* the first 80% are used for training;
* the final 20% are used for validation;
* validation performance is measured with mean absolute error;
* the model is then retrained on all available labelled rows before producing the latest forecast.

This is a time-ordered holdout and avoids randomly mixing future observations into the training set. It is not a repeated walk-forward backtest, so the validation result should be treated as a limited diagnostic rather than evidence of stable out-of-sample performance.

For the recorded 23 June 2026 run, the per-stock validation MAEs ranged from 0.85 to 1.86 percentage points, with an unweighted mean of approximately 1.37 percentage points.

### Forecast confidence

The displayed confidence value is derived from disagreement among the 200 individual trees:

```python
confidence = clip(1 - tree_prediction_std / 0.02, 0, 1)
```

A smaller spread produces a higher value. This is a model-agreement heuristic, not a calibrated probability that a forecast or signal will be correct.

### Forecast signals

Raw signals use fixed next-return thresholds:

|       Predicted next return | Raw signal |
| --------------------------: | ---------- |
|        Greater than `+0.5%` | `BUY`      |
| Between `-0.5%` and `+0.5%` | `HOLD`     |
|           Less than `-0.5%` | `REDUCE`   |

## Risk calculations

Risk measures use up to 252 stored observations per security.

### Stock-level metrics

| Metric                | Calculation                                                                |
| --------------------- | -------------------------------------------------------------------------- |
| Annualised volatility | Daily return standard deviation multiplied by `sqrt(252)`                  |
| Historical VaR, 95%   | Fifth percentile of observed daily returns                                 |
| Historical VaR, 99%   | First percentile of observed daily returns                                 |
| Maximum drawdown      | Worst peak-to-trough decline in closing prices                             |
| Beta                  | Covariance of stock and ASX 200 returns divided by ASX 200 return variance |
| Risk level            | Classification based on annualised volatility                              |

Risk classifications are:

| Annualised volatility | Risk level |
| --------------------: | ---------- |
|             Below 20% | `LOW`      |
|      20% to below 35% | `MEDIUM`   |
|          35% or above | `HIGH`     |

The code also calculates a custom ranking score:

```text
(predicted daily return / estimated daily volatility) × confidence
```

The source code names this value `sharpe_score`, but it is not a conventional Sharpe ratio because it does not use realised portfolio returns or a risk-free rate.

### Risk modifiers

| Condition                                  | Modifier    |
| ------------------------------------------ | ----------- |
| Historical 95% VaR is below `-5%`          | `BLOCK`     |
| Risk is high and confidence is below 40%   | `DOWNGRADE` |
| Risk is medium and confidence is below 25% | `DOWNGRADE` |
| Otherwise                                  | `CONFIRM`   |

### Watchlist-level summary

The project reports:

* average beta across assessed securities;
* average annualised volatility;
* average historical 95% VaR;
* the most common sector in the watchlist;
* sector concentration risk.

Sector weights are calculated by counting securities, not by market value or invested capital. A 40% Financials figure means four of the ten watchlist companies are classified as Financials.

Concentration is classified as:

|    Largest sector share | Classification |
| ----------------------: | -------------- |
|               Above 40% | `HIGH`         |
| Above 25% and up to 40% | `MEDIUM`       |
|            25% or below | `LOW`          |

## Strategy rules

The strategy node combines the raw forecast signal with the risk modifier.

| Forecast | `CONFIRM` | `DOWNGRADE` | `BLOCK` |
| -------- | --------- | ----------- | ------- |
| `BUY`    | `BUY`     | `HOLD`      | `HOLD`  |
| `HOLD`   | `HOLD`    | `HOLD`      | `HOLD`  |
| `REDUCE` | `REDUCE`  | `HOLD`      | `HOLD`  |

Forecasts with confidence below 30% default to `HOLD`, regardless of their raw signal.

### Position adjustments

Position sizes describe suggested adjustments rather than construction of a complete portfolio. `HOLD` recommendations receive a zero adjustment.

The default base adjustment is 10%. It is scaled using forecast confidence and risk level:

* confidence of at least 60%: multiply by 1.5;
* confidence from 45% to below 60%: no confidence adjustment;
* confidence below 45%: multiply by 0.5;
* high risk: multiply by 0.5;
* medium risk: multiply by 0.75;
* final size capped at 15%.

The output is labelled `FULL`, `REDUCED`, `MINIMAL` or `NONE`.

No optimisation is performed across recommendations, and the suggested adjustments are not checked against a portfolio-level capital constraint.

## LLM reporting

Gemini 2.5 Flash Lite is used only by the report-writing node.

The prompt supplies:

* latest stored macro values;
* portfolio-risk context;
* all final watchlist recommendations;
* actionable BUY and REDUCE recommendations;
* exact confidence, risk and position labels;
* pre-generated rationales from the deterministic strategy code.

The prompt instructs the model not to:

* invent market movements, prices, returns or explanations;
* infer unsupported macro trends;
* modify supplied numbers or percentages;
* change recommendations, confidence values, risk levels or position labels.

These controls reduce the scope for unsupported output, but they do not provide a programmatic guarantee that every generated sentence will comply. The saved report should still be reviewed before operational use.

## Verified sample run

A recorded run on 23 June 2026 reached the `report_complete` state.

| Pipeline output             |                          Result |
| --------------------------- | ------------------------------: |
| Watchlist tickers processed |                              10 |
| Forecasts generated         |                              10 |
| Risk assessments completed  |                              10 |
| Final recommendations       |         1 BUY, 8 HOLD, 1 REDUCE |
| Market bias                 |                       `NEUTRAL` |
| Report                      | `reports/asx_brief_20260623.md` |

### Portfolio-risk snapshot

| Metric                              |              Value |
| ----------------------------------- | -----------------: |
| Average beta                        |               0.84 |
| Average annualised volatility       |              25.0% |
| Average one-day historical VaR, 95% |             -2.28% |
| Largest watchlist sector            | Financial Services |
| Securities in largest sector        |                40% |
| Concentration classification        |           `MEDIUM` |

### Actionable recommendations

| Ticker   | Signal   | Predicted return | Confidence heuristic | Risk     | Position label |
| -------- | -------- | ---------------: | -------------------: | -------- | -------------- |
| `CBA.AX` | `BUY`    |           +0.54% |                  60% | `MEDIUM` | `REDUCED`      |
| `MQG.AX` | `REDUCE` |           -0.68% |                  64% | `MEDIUM` | `REDUCED`      |

These results confirm that the pipeline executed and produced internally consistent outputs. They do not establish forecast accuracy, profitability or risk-adjusted investment performance.

![Pipeline execution](screenshots/pipeline_execution.png)

![Forecasting output](screenshots/forecasting_agent.png)

![Generated report](screenshots/final_report.png)

## Repository structure

```text
.
├── agents/
│   ├── forecasting_agent.py
│   ├── market_data_agent.py
│   ├── report_writer_agent.py
│   ├── risk_agent.py
│   └── strategy_agent.py
├── reports/
│   └── asx_brief_YYYYMMDD.md
├── screenshots/
├── .env.example
├── asx_market_data.db
├── orchestrator.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

### Requirements

* Python 3.10 or later
* internet access for Yahoo Finance and Gemini
* a Gemini API key for report generation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Add the API key:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not commit the populated `.env` file.

## Running the project

Run the full pipeline once:

```bash
python orchestrator.py --mode run
```

Use a different DuckDB path:

```bash
python orchestrator.py --mode run --db data/asx_market_data.db
```

Run the application-level scheduler:

```bash
python orchestrator.py --mode schedule
```

The scheduler calls the pipeline at 07:00 according to the host machine’s local clock. The process must remain running. The current implementation schedules every calendar day, including weekends, and does not explicitly handle AEST/AEDT, exchange holidays or daylight-saving transitions.

For unattended operation, use an operating-system or cloud scheduler and configure its timezone and trading-day rules explicitly.

## Outputs

| Output              | Default location                       |
| ------------------- | -------------------------------------- |
| DuckDB database     | `asx_market_data.db`                   |
| Generated report    | `reports/asx_brief_YYYYMMDD.md`        |
| Runtime diagnostics | Standard output through Python logging |

The pipeline accumulates error messages in shared state. It halts through an explicit error node when a critical stage cannot provide the data required by the next stage.

## Limitations

* Yahoo Finance is a convenient public source, not an institutional market-data feed.
* Data availability, revisions, adjusted prices and metadata depend on the upstream provider.
* Validation uses one chronological holdout rather than rolling or nested backtesting.
* The project does not report directional accuracy, hit rate, benchmark-relative performance or trading returns.
* The confidence score is not statistically calibrated.
* Feature importance is global Random Forest impurity importance, not a local explanation of a particular prediction.
* Risk estimates use historical observations and may not represent future tail behaviour.
* VaR is calculated independently for each security and averaged for the watchlist. It is not a portfolio VaR based on positions and correlations.
* Sector concentration is based on security counts rather than capital weights.
* The custom risk-adjusted ranking score is not a standard Sharpe ratio.
* Position-size suggestions are heuristic and are not produced by a portfolio optimiser.
* Transaction costs, bid-ask spreads, liquidity constraints, slippage and taxes are not modelled.
* There is no order-management, brokerage or live-execution integration.
* The generated narrative is constrained by prompting but is not automatically validated against the structured inputs.
* Automated scheduling depends on the local process remaining active.

## Next development priorities

The most useful extensions would be:

1. a rolling backtest with untouched test periods and benchmark comparisons;
2. directional accuracy, MAE stability and signal-level precision reporting;
3. transaction-cost and turnover modelling;
4. calibrated uncertainty estimates;
5. portfolio-level VaR using weights and cross-asset correlations;
6. tests for feature timing, database writes, decision thresholds and prompt formatting;
7. machine-readable run artefacts for forecasts, validation metrics and recommendations;
8. schema validation of the LLM output against supplied values;
9. timezone-aware scheduling with ASX trading calendars;
10. model-drift and data-quality monitoring.

## Disclaimer

This repository is an educational portfolio project. It is not financial advice, an investment recommendation, or a production trading system.

## Licence

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).