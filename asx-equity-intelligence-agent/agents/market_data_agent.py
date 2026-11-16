"""Market data ingestion, validation and DuckDB persistence."""

import argparse
import logging
import math
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)


logger = logging.getLogger(__name__)


DEFAULT_DB_PATH = "asx_market_data.db"
DEFAULT_LOOKBACK_DAYS = 365
MAX_DATA_AGE_DAYS = 4
RETURN_BUFFER_DAYS = 14
MAX_ABSOLUTE_EQUITY_RETURN = 0.50

ASX_TIMEZONE = "Australia/Sydney"
BENCHMARK_TICKER = "^AXJO"


DEFAULT_WATCHLIST = {
    "BHP.AX": "BHP Group (Materials)",
    "CBA.AX": "Commonwealth Bank (Financials)",
    "CSL.AX": "CSL Limited (Healthcare)",
    "WDS.AX": "Woodside Energy (Energy)",
    "WES.AX": "Wesfarmers (Consumer Discretionary)",
    "ANZ.AX": "ANZ Banking Group (Financials)",
    "RIO.AX": "Rio Tinto (Materials)",
    "WBC.AX": "Westpac Banking (Financials)",
    "MQG.AX": "Macquarie Group (Financials)",
    "TLS.AX": "Telstra (Telecommunications)",
}


MACRO_TICKERS = {
    "^AXJO": "ASX 200 Index",
    "AUDUSD=X": "AUD/USD Exchange Rate",
    "GC=F": "Gold Futures (USD)",
    "CL=F": "Crude Oil Futures (WTI)",
}


MACRO_TIMEZONES = {
    "^AXJO": ASX_TIMEZONE,
    "AUDUSD=X": "Europe/London",
    "GC=F": "America/New_York",
    "CL=F": "America/New_York",
}


class OHLCVRecord(BaseModel):
    """Validated daily equity market observation."""

    model_config = ConfigDict(allow_inf_nan=False)

    ticker: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    daily_return: float

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        value = value.strip().upper()

        if not value:
            raise ValueError("Ticker cannot be empty.")

        return value

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        return date.fromisoformat(value).isoformat()

    @field_validator("open", "high", "low", "close")
    @classmethod
    def validate_price(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Prices must be positive.")

        return value

    @field_validator("volume")
    @classmethod
    def validate_volume(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Volume cannot be negative.")

        return value

    @field_validator("daily_return")
    @classmethod
    def validate_return(cls, value: float) -> float:
        if abs(value) > MAX_ABSOLUTE_EQUITY_RETURN:
            raise ValueError(
                f"Suspicious daily equity return: {value:.2%}"
            )

        return value

    @model_validator(mode="after")
    def validate_price_relationships(self):
        tolerance = max(self.high, self.close) * 1e-6

        if self.low > self.high + tolerance:
            raise ValueError("Low price exceeds high price.")

        if self.open > self.high + tolerance:
            raise ValueError("Open price exceeds high price.")

        if self.close > self.high + tolerance:
            raise ValueError("Close price exceeds high price.")

        if self.open < self.low - tolerance:
            raise ValueError("Open price is below low price.")

        if self.close < self.low - tolerance:
            raise ValueError("Close price is below low price.")

        return self


class FundamentalSnapshot(BaseModel):
    """Timestamped company fundamentals and sector metadata."""

    model_config = ConfigDict(allow_inf_nan=False)

    ticker: str
    fetched_at: str
    market_cap: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    dividend_yield: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    beta: float | None = None
    sector: str | None = None
    industry: str | None = None


def optional_float(value) -> float | None:
    """Convert optional metadata into a finite numerical value."""

    if value is None:
        return None

    try:
        result = float(value)

    except (TypeError, ValueError, OverflowError):
        return None

    return result if math.isfinite(result) else None


def optional_text(value) -> str | None:
    """Normalise optional textual metadata."""

    if not isinstance(value, str):
        return None

    value = value.strip()

    return value or None


def normalise_history(
    history: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Normalise and validate single-instrument market history."""

    if history is None or history.empty:
        raise ValueError("No market history returned.")

    df = history.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.columns.duplicated().any():
        raise ValueError("Duplicate market data columns.")

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df.loc[:, list(required_columns)].copy()

    index = pd.DatetimeIndex(
        pd.to_datetime(df.index, errors="raise")
    )

    if index.hasnans:
        raise ValueError("Invalid observation dates.")

    if index.tz is not None:
        index = index.tz_localize(None)

    df.index = index.normalize()
    df.index.name = "date"

    df = df.sort_index()

    df = df.loc[
        ~df.index.duplicated(keep="last")
    ].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if not np.isfinite(
        df.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Market history contains missing or non-finite values."
        )

    if (df["Close"] <= 0).any():
        raise ValueError(
            "Market history contains non-positive closing prices."
        )

    return df


def calculate_daily_returns(
    history: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Calculate consecutive-observation returns before date filtering."""

    df = history.copy()

    df["daily_return"] = df["Close"].pct_change(
        fill_method=None
    )

    df = df.dropna(subset=["daily_return"])

    df = df.loc[
        (df.index >= pd.Timestamp(start_date))
        & (df.index < pd.Timestamp(end_date))
    ].copy()

    if df.empty:
        raise ValueError(
            "No observations with calculable returns."
        )

    if not np.isfinite(
        df["daily_return"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Calculated returns contain non-finite values."
        )

    return df


def assess_forecast_readiness(
    equity_dates: list[str],
    benchmark_dates: list[str],
) -> tuple[bool, str]:
    """Assess whether equity and benchmark histories are aligned."""

    equity = sorted(
        set(equity_dates)
    )

    benchmark = sorted(
        set(benchmark_dates)
    )

    if not equity:
        return False, "Equity history is unavailable."

    if not benchmark:
        return False, "Benchmark history is unavailable."

    if equity[-1] != benchmark[-1]:
        return (
            False,
            f"Latest equity observation {equity[-1]} "
            f"does not match benchmark observation "
            f"{benchmark[-1]}.",
        )

    overlap_start = max(
        equity[0],
        benchmark[0],
    )

    overlap_end = min(
        equity[-1],
        benchmark[-1],
    )

    equity_overlap = {
        observation
        for observation in equity
        if overlap_start <= observation <= overlap_end
    }

    benchmark_overlap = {
        observation
        for observation in benchmark
        if overlap_start <= observation <= overlap_end
    }

    if not equity_overlap or not benchmark_overlap:
        return False, "No overlapping market observations."

    missing_benchmark = sorted(
        equity_overlap - benchmark_overlap
    )

    missing_equity = sorted(
        benchmark_overlap - equity_overlap
    )

    if missing_benchmark or missing_equity:
        reasons = []

        if missing_benchmark:
            reasons.append(
                "Missing benchmark dates: "
                + ", ".join(missing_benchmark)
            )

        if missing_equity:
            reasons.append(
                "Missing equity dates: "
                + ", ".join(missing_equity)
            )

        return False, "; ".join(reasons)

    return True, "Equity and benchmark dates are aligned."


class MarketDatabase:
    """Manage local market data persistence using DuckDB."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
    ):
        self.db_path = db_path
        self.closed = False

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.conn = duckdb.connect(db_path)

        try:
            self._initialise_tables()

        except Exception:
            self.close()
            raise

    def _initialise_tables(self) -> None:
        """Create the existing database schema when necessary."""

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                ticker VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                daily_return DOUBLE,
                PRIMARY KEY (ticker, date)
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker VARCHAR,
                fetched_at TIMESTAMP,
                market_cap DOUBLE,
                pe_ratio DOUBLE,
                forward_pe DOUBLE,
                dividend_yield DOUBLE,
                fifty_two_week_high DOUBLE,
                fifty_two_week_low DOUBLE,
                beta DOUBLE,
                sector VARCHAR,
                industry VARCHAR,
                PRIMARY KEY (ticker, fetched_at)
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS macro (
                ticker VARCHAR,
                date DATE,
                close DOUBLE,
                daily_return DOUBLE,
                PRIMARY KEY (ticker, date)
            )
            """
        )

        logger.info(
            "Database initialised at %s",
            self.db_path,
        )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit a complete batch or roll it back."""

        self.conn.execute("BEGIN TRANSACTION")

        try:
            yield

        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

        else:
            self.conn.execute("COMMIT")

    def upsert_ohlcv(
        self,
        records: list[OHLCVRecord],
    ) -> None:
        """Insert or replace validated equity observations."""

        if not records:
            return

        data = [
            (
                record.ticker,
                record.date,
                record.open,
                record.high,
                record.low,
                record.close,
                record.volume,
                record.daily_return,
            )
            for record in records
        ]

        with self.transaction():
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO ohlcv (
                    ticker,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    daily_return
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                data,
            )

        logger.info(
            "Upserted %d OHLCV records",
            len(records),
        )

    def upsert_fundamentals(
        self,
        snapshot: FundamentalSnapshot,
    ) -> None:
        """Store a timestamped fundamental snapshot."""

        self.conn.execute(
            """
            INSERT OR REPLACE INTO fundamentals (
                ticker,
                fetched_at,
                market_cap,
                pe_ratio,
                forward_pe,
                dividend_yield,
                fifty_two_week_high,
                fifty_two_week_low,
                beta,
                sector,
                industry
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.ticker,
                snapshot.fetched_at,
                snapshot.market_cap,
                snapshot.pe_ratio,
                snapshot.forward_pe,
                snapshot.dividend_yield,
                snapshot.fifty_two_week_high,
                snapshot.fifty_two_week_low,
                snapshot.beta,
                snapshot.sector,
                snapshot.industry,
            ),
        )

    def upsert_macro(
        self,
        ticker: str,
        history: pd.DataFrame,
    ) -> None:
        """Store validated macroeconomic observations."""

        if history.empty:
            return

        data = [
            (
                ticker,
                timestamp.date().isoformat(),
                float(row["Close"]),
                float(row["daily_return"]),
            )
            for timestamp, row in history.iterrows()
        ]

        with self.transaction():
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO macro (
                    ticker,
                    date,
                    close,
                    daily_return
                )
                VALUES (?, ?, ?, ?)
                """,
                data,
            )

    def get_ohlcv(
        self,
        ticker: str,
        days: int = 60,
    ) -> pd.DataFrame:
        """Retrieve recent stored equity observations."""

        return self.conn.execute(
            """
            SELECT *
            FROM ohlcv
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (ticker, days),
        ).df()

    def get_latest_fundamentals(
        self,
        ticker: str,
    ) -> dict:
        """Retrieve the latest stored fundamental snapshot."""

        result = self.conn.execute(
            """
            SELECT *
            FROM fundamentals
            WHERE ticker = ?
            ORDER BY fetched_at DESC
            LIMIT 1
            """,
            (ticker,),
        ).df()

        if result.empty:
            return {}

        return result.to_dict("records")[0]

    def close(self) -> None:
        """Close the database connection."""

        if not self.closed:
            self.conn.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()


class MarketDataAgent:
    """Retrieve, validate and persist market observations."""

    def __init__(
        self,
        watchlist: dict[str, str] | None = None,
        macro_tickers: dict[str, str] | None = None,
        db_path: str = DEFAULT_DB_PATH,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ):
        if lookback_days < 1:
            raise ValueError(
                "Lookback period must be positive."
            )

        self.watchlist = dict(
            DEFAULT_WATCHLIST
            if watchlist is None
            else watchlist
        )

        self.macro_tickers = dict(
            MACRO_TICKERS
            if macro_tickers is None
            else macro_tickers
        )

        if BENCHMARK_TICKER not in self.macro_tickers:
            raise ValueError(
                "The ASX 200 benchmark must be configured."
            )

        self.lookback_days = lookback_days
        self.timezone = ZoneInfo(ASX_TIMEZONE)

        self.db = MarketDatabase(db_path)

        self.start_date = ""
        self.end_date = ""

        self._refresh_date_window()

    def _refresh_date_window(self) -> None:
        """Recalculate the ingestion window for the current run."""

        today = datetime.now(
            self.timezone
        ).date()

        start = today - timedelta(
            days=self.lookback_days
        )

        self.start_date = start.isoformat()
        self.end_date = today.isoformat()

    @staticmethod
    def _validate_freshness(
        ticker: str,
        latest_date: date,
        reference_date: date,
    ) -> None:
        """Reject stale, current-day or future observations."""

        age = (
            reference_date - latest_date
        ).days

        if age <= 0:
            raise ValueError(
                f"{ticker}: current-day or future observation."
            )

        if age > MAX_DATA_AGE_DAYS:
            raise ValueError(
                f"{ticker}: stale market data. "
                f"Latest observation: {latest_date}; "
                f"age: {age} calendar days."
            )

    def _download_history(
        self,
        ticker: str,
        required_columns: tuple[str, ...],
        timezone_name: str = ASX_TIMEZONE,
    ) -> pd.DataFrame:
        """Retrieve completed daily observations."""

        local_today = datetime.now(
            ZoneInfo(timezone_name)
        ).date()

        start = date.fromisoformat(
            self.start_date
        )

        end = min(
            date.fromisoformat(self.end_date),
            local_today,
        )

        download_start = start - timedelta(
            days=RETURN_BUFFER_DAYS
        )

        logger.info(
            "Downloading %s | %s to %s (exclusive)",
            ticker,
            download_start,
            end,
        )

        history = yf.download(
            ticker,
            start=download_start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=30,
        )

        history = normalise_history(
            history,
            required_columns,
        )

        # Exclude current-day observations even if the provider
        # returns data beyond the requested exclusive end date.
        history = history.loc[
            history.index < pd.Timestamp(end)
        ].copy()

        history = calculate_daily_returns(
            history,
            start,
            end,
        )

        self._validate_freshness(
            ticker,
            history.index[-1].date(),
            local_today,
        )

        return history

    def _fetch_ohlcv(
        self,
        ticker: str,
    ) -> list[OHLCVRecord]:
        """Retrieve and validate adjusted equity observations."""

        history = self._download_history(
            ticker,
            required_columns=(
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ),
        )

        records = [
            OHLCVRecord(
                ticker=ticker,
                date=timestamp.date().isoformat(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
                daily_return=float(row["daily_return"]),
            )
            for timestamp, row in history.iterrows()
        ]

        logger.info(
            "%s: %d validated observations",
            ticker,
            len(records),
        )

        return records

    def _fetch_fundamentals(
        self,
        ticker: str,
    ) -> FundamentalSnapshot:
        """Retrieve current company fundamentals and metadata."""

        info = yf.Ticker(ticker).info

        if not info:
            raise ValueError(
                "No fundamental metadata returned."
            )

        fetched_at = datetime.now(
            timezone.utc
        ).replace(tzinfo=None)

        return FundamentalSnapshot(
            ticker=ticker,
            fetched_at=fetched_at.isoformat(),
            market_cap=optional_float(
                info.get("marketCap")
            ),
            pe_ratio=optional_float(
                info.get("trailingPE")
            ),
            forward_pe=optional_float(
                info.get("forwardPE")
            ),
            dividend_yield=optional_float(
                info.get("dividendYield")
            ),
            fifty_two_week_high=optional_float(
                info.get("fiftyTwoWeekHigh")
            ),
            fifty_two_week_low=optional_float(
                info.get("fiftyTwoWeekLow")
            ),
            beta=optional_float(
                info.get("beta")
            ),
            sector=optional_text(
                info.get("sector")
            ),
            industry=optional_text(
                info.get("industry")
            ),
        )

    def _fetch_macro(self) -> dict:
        """Retrieve benchmark and supplementary macroeconomic data."""

        results = {
            "success": [],
            "failed": [],
            "latest_dates": {},
            "observation_dates": {},
            "errors": [],
        }

        for ticker, name in self.macro_tickers.items():
            try:
                history = self._download_history(
                    ticker,
                    required_columns=("Close",),
                    timezone_name=MACRO_TIMEZONES.get(
                        ticker,
                        ASX_TIMEZONE,
                    ),
                )

                self.db.upsert_macro(
                    ticker,
                    history,
                )

                observation_dates = [
                    timestamp.date().isoformat()
                    for timestamp in history.index
                ]

                latest_date = observation_dates[-1]

                results["success"].append(
                    ticker
                )

                results["latest_dates"][ticker] = (
                    latest_date
                )

                results["observation_dates"][ticker] = (
                    observation_dates
                )

                logger.info(
                    "%s (%s): %d observations | latest %s",
                    name,
                    ticker,
                    len(history),
                    latest_date,
                )

            except Exception as exc:
                message = (
                    f"Macro ingestion failed for "
                    f"{ticker}: {exc}"
                )

                results["failed"].append(
                    ticker
                )

                results["errors"].append(
                    message
                )

                logger.warning(
                    "%s",
                    message,
                )

        return results

    def run(self) -> dict:
        """Ingest data and assess downstream analytical readiness."""

        self._refresh_date_window()

        logger.info(
            "Market data ingestion started | %s to %s",
            self.start_date,
            self.end_date,
        )

        results = {
            "success": [],
            "failed": [],
            "total_records": 0,
            "latest_dates": {},
            "macro_success": [],
            "macro_failed": [],
            "macro_latest_dates": {},
            "fundamentals_success": [],
            "fundamentals_failed": [],
            "benchmark_ready": False,
            "forecast_ready": [],
            "forecast_blocked": [],
            "readiness_reasons": {},
            "warnings": [],
            "errors": [],
        }

        macro_results = self._fetch_macro()

        results["macro_success"] = (
            macro_results["success"]
        )

        results["macro_failed"] = (
            macro_results["failed"]
        )

        results["macro_latest_dates"] = (
            macro_results["latest_dates"]
        )

        results["errors"].extend(
            macro_results["errors"]
        )

        benchmark_dates = (
            macro_results["observation_dates"]
            .get(BENCHMARK_TICKER, [])
        )

        # Benchmark availability alone does not establish
        # that any individual security is forecast-ready.
        results["benchmark_ready"] = bool(
            benchmark_dates
        )

        if not benchmark_dates:
            message = (
                "ASX 200 benchmark unavailable. "
                "Equity ingestion will continue, but "
                "forecasting readiness cannot be established."
            )

            results["warnings"].append(
                message
            )

            logger.warning(
                "%s",
                message,
            )

        for ticker in self.watchlist:
            try:
                records = self._fetch_ohlcv(
                    ticker
                )

                if not records:
                    raise ValueError(
                        "No validated equity observations."
                    )

                # Persist valid equity data independently
                # of benchmark availability.
                self.db.upsert_ohlcv(
                    records
                )

                results["success"].append(
                    ticker
                )

                results["latest_dates"][ticker] = (
                    records[-1].date
                )

                results["total_records"] += len(
                    records
                )

            except Exception as exc:
                message = (
                    f"Equity ingestion failed for "
                    f"{ticker}: {exc}"
                )

                results["failed"].append(
                    ticker
                )

                results["errors"].append(
                    message
                )

                logger.warning(
                    "%s",
                    message,
                )

                continue

            # Assess readiness without deleting or fabricating
            # otherwise valid market observations.
            ready, reason = assess_forecast_readiness(
                equity_dates=[
                    record.date
                    for record in records
                ],
                benchmark_dates=benchmark_dates,
            )

            results["readiness_reasons"][ticker] = (
                reason
            )

            if ready:
                results["forecast_ready"].append(
                    ticker
                )

                logger.info(
                    "%s: forecast data aligned",
                    ticker,
                )

            else:
                results["forecast_blocked"].append(
                    ticker
                )

                message = (
                    f"{ticker}: forecast readiness "
                    f"blocked. {reason}"
                )

                results["warnings"].append(
                    message
                )

                logger.warning(
                    "%s",
                    message,
                )

            # Fundamentals remain independent of
            # forecast-readiness status.
            try:
                snapshot = self._fetch_fundamentals(
                    ticker
                )

                if snapshot.sector is None:
                    raise ValueError(
                        "Sector metadata is unavailable."
                    )

                self.db.upsert_fundamentals(
                    snapshot
                )

                results["fundamentals_success"].append(
                    ticker
                )

            except Exception as exc:
                message = (
                    f"Fundamentals ingestion failed for "
                    f"{ticker}: {exc}"
                )

                results["fundamentals_failed"].append(
                    ticker
                )

                results["errors"].append(
                    message
                )

                logger.warning(
                    "%s",
                    message,
                )

        logger.info(
            "Equity ingestion: %d/%d successful",
            len(results["success"]),
            len(self.watchlist),
        )

        logger.info(
            "Macro ingestion: %d/%d successful",
            len(results["macro_success"]),
            len(self.macro_tickers),
        )

        logger.info(
            "Fundamentals ingestion: %d/%d successful",
            len(results["fundamentals_success"]),
            len(results["success"]),
        )

        logger.info(
            "Forecast-ready securities: %d/%d",
            len(results["forecast_ready"]),
            len(self.watchlist),
        )

        logger.info(
            "Total equity records processed: %d",
            results["total_records"],
        )

        return results

    def get_watchlist_summary(
        self,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return the latest stored observation for each security."""

        selected = (
            list(self.watchlist)
            if tickers is None
            else sorted(set(tickers))
        )

        rows = []

        for ticker in selected:
            history = self.db.get_ohlcv(
                ticker,
                days=1,
            )

            if history.empty:
                continue

            latest = history.iloc[0]

            rows.append(
                {
                    "ticker": ticker,
                    "name": self.watchlist.get(
                        ticker,
                        ticker,
                    ),
                    "date": latest["date"],
                    "close": float(latest["close"]),
                    "daily_return": (
                        f"{latest['daily_return']:+.2%}"
                    ),
                }
            )

        return pd.DataFrame(rows)

    def close(self) -> None:
        """Release database resources."""

        self.db.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()


def main() -> int:
    """Execute standalone ingestion and readiness assessment."""

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve and validate ASX market data."
        )
    )

    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help="Path to the DuckDB database.",
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Historical ingestion window in calendar days.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(name)s | "
            "%(levelname)s | %(message)s"
        ),
    )

    with MarketDataAgent(
        db_path=args.db,
        lookback_days=args.lookback_days,
    ) as agent:

        results = agent.run()

        print("\nMarket Data Ingestion Summary")

        print(
            f"Successful equity tickers: "
            f"{len(results['success'])}"
        )

        print(
            f"Failed equity tickers: "
            f"{len(results['failed'])}"
        )

        print(
            f"Successful macro series: "
            f"{len(results['macro_success'])}"
        )

        print(
            f"Successful fundamental snapshots: "
            f"{len(results['fundamentals_success'])}"
        )

        print(
            f"Equity observations processed: "
            f"{results['total_records']}"
        )

        print("\nForecast Readiness Summary")

        print(
            f"Benchmark available: "
            f"{results['benchmark_ready']}"
        )

        print(
            f"Forecast-ready securities: "
            f"{len(results['forecast_ready'])}"
        )

        print(
            f"Forecast-blocked securities: "
            f"{len(results['forecast_blocked'])}"
        )

        if results["success"]:
            print("\nWatchlist Summary")

            print(
                agent.get_watchlist_summary(
                    results["success"]
                ).to_string(index=False)
            )

        if results["warnings"]:
            print("\nData Quality Warnings")

            for warning in results["warnings"]:
                print(f"- {warning}")

        if results["errors"]:
            print("\nIngestion Errors")

            for error in results["errors"]:
                print(f"- {error}")

        if not results["success"]:
            return 1

        if (
            results["failed"]
            or results["macro_failed"]
            or results["fundamentals_failed"]
            or results["forecast_blocked"]
            or not results["benchmark_ready"]
        ):
            return 2

        return 0


if __name__ == "__main__":
    raise SystemExit(main())