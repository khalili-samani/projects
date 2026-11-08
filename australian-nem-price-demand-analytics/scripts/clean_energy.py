import sys
import pandas as pd

from config import RAW_DIR, PROCESSED_DIR, REGIONS


def load_region_file(year_month: str, region_code: str) -> pd.DataFrame:
    file_path = RAW_DIR / f"PRICE_AND_DEMAND_{year_month}_{region_code}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing raw file: {file_path}")

    df = pd.read_csv(file_path)

    df.columns = [col.strip().lower() for col in df.columns]

    df = df.rename(
        columns={
            "settlementdate": "settlement_datetime",
            "rrp": "rrp_aud_mwh",
            "totaldemand": "total_demand_mw",
            "periodtype": "period_type",
        }
    )

    required_cols = [
        "settlement_datetime",
        "rrp_aud_mwh",
        "total_demand_mw",
        "period_type",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{region_code} missing columns: {missing}")

    df["region_code"] = region_code
    df["settlement_datetime"] = pd.to_datetime(df["settlement_datetime"], errors="coerce")
    df["trading_date"] = df["settlement_datetime"].dt.date
    df["rrp_aud_mwh"] = pd.to_numeric(df["rrp_aud_mwh"], errors="coerce")
    df["total_demand_mw"] = pd.to_numeric(df["total_demand_mw"], errors="coerce")

    keep_cols = [
        "region_code",
        "settlement_datetime",
        "trading_date",
        "rrp_aud_mwh",
        "total_demand_mw",
        "period_type",
    ]

    df = df[keep_cols]

    df = df.dropna(
        subset=[
            "region_code",
            "settlement_datetime",
            "trading_date",
            "rrp_aud_mwh",
            "total_demand_mw",
        ]
    )

    df = df.sort_values(["region_code", "settlement_datetime"])

    return df


def main(year_month: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = []

    for region in REGIONS:
        df_region = load_region_file(year_month, region)
        frames.append(df_region)
        print(f"{region}: {df_region.shape[0]} rows cleaned")

    combined = pd.concat(frames, ignore_index=True)

    output_path = PROCESSED_DIR / f"nem_price_demand_{year_month}_combined.csv"
    combined.to_csv(output_path, index=False)

    print(f"Saved cleaned combined file: {output_path}")
    print(f"Total rows: {len(combined)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python scripts/clean_energy.py YYYYMM")

    main(sys.argv[1])
