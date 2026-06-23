"""
Market Data Agent
=================
Responsibility: Fetch, validate, and store ASX market data for the watchlist.
This is the foundation layer — all other agents consume data from here.

Design decisions:
- yfinance for OHLCV (free, reliable, ASX-compatible with .AX suffix)
- DuckDB for local time series storage (fast, SQL-native, zero infra)
- Pydantic for data validation before storage
- Structured logging so the orchestrator can track agent health
"""

import logging
import duckdb
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Logging — every agent uses structured logs so the orchestrator can monitor
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("MarketDataAgent")


# ---------------------------------------------------------------------------
# Watchlist — 10 liquid ASX stocks across key sectors
# Format: yfinance requires the .AX suffix for ASX-listed securities
# ---------------------------------------------------------------------------
DEFAULT_WATCHLIST = {
    "BHP.AX":  "BHP Group (Materials)",
    "CBA.AX":  "Commonwealth Bank (Financials)",
    "CSL.AX":  "CSL Limited (Healthcare)",
    "WDS.AX":  "Woodside Energy (Energy)",
    "WES.AX":  "Wesfarmers (Consumer Discretionary)",
    "ANZ.AX":  "ANZ Banking Group (Financials)",
    "RIO.AX":  "Rio Tinto (Materials)",
    "WBC.AX":  "Westpac Banking (Financials)",
    "MQG.AX":  "Macquarie Group (Financials)",
    "TLS.AX":  "Telstra (Telecommunications)",
}

# Macro context tickers — these give the risk agent broader market signals
MACRO_TICKERS = {
    "^AXJO":   "ASX 200 Index",
    "AUDUSD=X": "AUD/USD Exchange Rate",
    "GC=F":    "Gold Futures (USD)",
    "CL=F":    "Crude Oil Futures (WTI)",
}


# ---------------------------------------------------------------------------
# Data Models — Pydantic validates incoming data before we trust it
# ---------------------------------------------------------------------------
class OHLCVRecord(BaseModel):
    ticker: str
    date: str           # ISO format: YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: float
    daily_return: float  # (close - prev_close) / prev_close

    @field_validator("close", "open", "high", "low")
    @classmethod
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Price must be positive, got {v}")
        return round(v, 4)

    @field_validator("daily_return")
    @classmethod
    def return_must_be_sane(cls, v):
        # Flag if daily return is outside ±50% — likely a data error
        if abs(v) > 0.5:
            raise ValueError(f"Suspicious daily return: {v:.2%}")
        return round(v, 6)


class FundamentalSnapshot(BaseModel):
    ticker: str
    fetched_at: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    dividend_yield: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


# ---------------------------------------------------------------------------
# Database Layer — DuckDB stores everything locally as a .db file
# We use two tables: ohlcv for price history, fundamentals for snapshots
# ---------------------------------------------------------------------------
class MarketDatabase:
    def __init__(self, db_path: str = "asx_market_data.db"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._initialise_tables()

    def _initialise_tables(self):
        """Create tables if they don't already exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                ticker        VARCHAR,
                date          DATE,
                open          DOUBLE,
                high          DOUBLE,
                low           DOUBLE,
                close         DOUBLE,
                volume        DOUBLE,
                daily_return  DOUBLE,
                PRIMARY KEY (ticker, date)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker              VARCHAR,
                fetched_at          TIMESTAMP,
                market_cap          DOUBLE,
                pe_ratio            DOUBLE,
                forward_pe          DOUBLE,
                dividend_yield      DOUBLE,
                fifty_two_week_high DOUBLE,
                fifty_two_week_low  DOUBLE,
                beta                DOUBLE,
                sector              VARCHAR,
                industry            VARCHAR,
                PRIMARY KEY (ticker, fetched_at)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS macro (
                ticker        VARCHAR,
                date          DATE,
                close         DOUBLE,
                daily_return  DOUBLE,
                PRIMARY KEY (ticker, date)
            )
        """)
        logger.info(f"Database initialised at {self.db_path}")

    def upsert_ohlcv(self, records: list[OHLCVRecord]):
        """Insert or replace OHLCV records (handles re-runs cleanly)."""
        if not records:
            return
        data = [(r.ticker, r.date, r.open, r.high, r.low,
                  r.close, r.volume, r.daily_return) for r in records]
        self.conn.executemany("""
            INSERT OR REPLACE INTO ohlcv
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        logger.info(f"Upserted {len(records)} OHLCV records")

    def upsert_fundamentals(self, snap: FundamentalSnapshot):
        self.conn.execute("""
            INSERT OR REPLACE INTO fundamentals VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            snap.ticker, snap.fetched_at, snap.market_cap, snap.pe_ratio,
            snap.forward_pe, snap.dividend_yield, snap.fifty_two_week_high,
            snap.fifty_two_week_low, snap.beta, snap.sector, snap.industry
        ))

    def upsert_macro(self, ticker: str, df: pd.DataFrame):
        """Store macro ticker history (ASX200, AUD/USD, etc.)"""
        for date, row in df.iterrows():
            self.conn.execute("""
                INSERT OR REPLACE INTO macro VALUES (?, ?, ?, ?)
            """, (ticker, str(date.date()), float(row["Close"]),
                  float(row.get("daily_return", 0.0))))

    def get_ohlcv(self, ticker: str, days: int = 60) -> pd.DataFrame:
        """Retrieve recent OHLCV history for a ticker."""
        return self.conn.execute("""
            SELECT * FROM ohlcv
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """, (ticker, days)).df()

    def get_latest_fundamentals(self, ticker: str) -> dict:
        result = self.conn.execute("""
            SELECT * FROM fundamentals
            WHERE ticker = ?
            ORDER BY fetched_at DESC
            LIMIT 1
        """, (ticker,)).df()
        return result.to_dict("records")[0] if not result.empty else {}

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Market Data Agent — the agent itself
# ---------------------------------------------------------------------------
class MarketDataAgent:
    """
    Fetches and stores market data for the ASX watchlist.

    In the multi-agent system, this agent runs first and its outputs
    are consumed by the Forecasting Agent and Risk Agent.
    """

    def __init__(
        self,
        watchlist: dict = DEFAULT_WATCHLIST,
        macro_tickers: dict = MACRO_TICKERS,
        db_path: str = "asx_market_data.db",
        lookback_days: int = 365,
    ):
        self.watchlist = watchlist
        self.macro_tickers = macro_tickers
        self.db = MarketDatabase(db_path)
        self.lookback_days = lookback_days
        self.start_date = (
            datetime.today() - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")
        self.end_date = datetime.today().strftime("%Y-%m-%d")

    def _fetch_ohlcv(self, ticker: str) -> list[OHLCVRecord]:
        """
        Download price history from Yahoo Finance and convert to validated records.
        The .AX suffix tells yfinance this is an ASX-listed security.
        """
        logger.info(f"Fetching OHLCV for {ticker}...")
        try:
            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True,   # adjusts for splits and dividends
            )

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return []

            # Flatten MultiIndex columns if yfinance returns them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Calculate daily returns — needed by forecasting and risk agents
            df["daily_return"] = df["Close"].pct_change().fillna(0)

            records = []
            for date, row in df.iterrows():
                try:
                    record = OHLCVRecord(
                        ticker=ticker,
                        date=str(date.date()),
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                        daily_return=float(row["daily_return"]),
                    )
                    records.append(record)
                except Exception as e:
                    # Skip bad rows but keep going — don't let one bad day
                    # break the whole fetch
                    logger.warning(f"Skipping row {date} for {ticker}: {e}")

            logger.info(f"  ✓ {ticker}: {len(records)} records")
            return records

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return []

    def _fetch_fundamentals(self, ticker: str) -> Optional[FundamentalSnapshot]:
        """
        Fetch fundamental data from yfinance .info dict.
        This is a snapshot — we store it with a timestamp so we can
        track how fundamentals change over time.
        """
        try:
            info = yf.Ticker(ticker).info
            snap = FundamentalSnapshot(
                ticker=ticker,
                fetched_at=datetime.now().isoformat(),
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                dividend_yield=info.get("dividendYield"),
                fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                fifty_two_week_low=info.get("fiftyTwoWeekLow"),
                beta=info.get("beta"),
                sector=info.get("sector"),
                industry=info.get("industry"),
            )
            logger.info(f"  ✓ Fundamentals fetched for {ticker}")
            return snap
        except Exception as e:
            logger.warning(f"Could not fetch fundamentals for {ticker}: {e}")
            return None

    def _fetch_macro(self):
        """
        Fetch macro context: ASX 200 index, AUD/USD, Gold, Oil.
        The risk agent uses these to assess broader market conditions.
        """
        logger.info("Fetching macro context...")
        for ticker, name in self.macro_tickers.items():
            try:
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    logger.warning(f"No macro data for {ticker}")
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df["daily_return"] = df["Close"].pct_change().fillna(0)
                self.db.upsert_macro(ticker, df)
                logger.info(f"  ✓ Macro: {name} ({ticker})")
            except Exception as e:
                logger.error(f"Macro fetch failed for {ticker}: {e}")

    def run(self) -> dict:
        """
        Main entry point for the agent.
        Called by the orchestrator at the start of each daily run.
        Returns a status summary the orchestrator can log.
        """
        logger.info("=" * 60)
        logger.info("Market Data Agent — starting run")
        logger.info(f"Watchlist: {len(self.watchlist)} tickers")
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info("=" * 60)

        results = {"success": [], "failed": [], "total_records": 0}

        # --- Step 1: Fetch OHLCV for each watchlist ticker ---
        for ticker in self.watchlist:
            records = self._fetch_ohlcv(ticker)
            if records:
                self.db.upsert_ohlcv(records)
                results["success"].append(ticker)
                results["total_records"] += len(records)
            else:
                results["failed"].append(ticker)

            # Fetch fundamentals alongside price data
            snap = self._fetch_fundamentals(ticker)
            if snap:
                self.db.upsert_fundamentals(snap)

        # --- Step 2: Fetch macro context ---
        self._fetch_macro()

        # --- Step 3: Summary ---
        logger.info("=" * 60)
        logger.info(f"Run complete.")
        logger.info(f"  Successful tickers : {len(results['success'])}")
        logger.info(f"  Failed tickers     : {len(results['failed'])}")
        logger.info(f"  Total OHLCV records: {results['total_records']}")
        if results["failed"]:
            logger.warning(f"  Failed: {results['failed']}")
        logger.info("=" * 60)

        return results

    def get_watchlist_summary(self) -> pd.DataFrame:
        """
        Returns a quick summary DataFrame of the latest close prices
        and returns for all watchlist tickers. Used by the orchestrator
        to confirm data is fresh before triggering downstream agents.
        """
        rows = []
        for ticker, name in self.watchlist.items():
            df = self.db.get_ohlcv(ticker, days=2)
            if df.empty:
                continue
            latest = df.iloc[0]
            rows.append({
                "ticker": ticker,
                "name": name,
                "date": latest["date"],
                "close": latest["close"],
                "daily_return": f"{latest['daily_return']:.2%}",
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run directly for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = MarketDataAgent()
    results = agent.run()

    print("\n📊 Watchlist Summary:")
    print(agent.get_watchlist_summary().to_string(index=False))