import pandas as pd

file_path = "data_raw/jnj_us.csv"

df = pd.read_csv(file_path)

print("Original columns:", df.columns)

df = df.rename(columns={
    "Date": "date",
    "Open": "open_price",
    "High": "high_price",
    "Low": "low_price",
    "Close": "close_price",
    "Volume": "volume"
})

df["company_id"] = 10

df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

numeric_cols = ["open_price", "high_price", "low_price", "close_price", "volume"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

df = df.dropna()

df = df.sort_values(by="date")

df.to_csv("data_cleaned/jnj_us_clean.csv", index=False)

print("Cleaned file saved successfully")
