import pandas as pd

# File path (change this)
file_path = "data_raw/aapl_us.csv"

# Load data
df = pd.read_csv(file_path)

# Show original columns
print("Original columns:", df.columns)

# Rename columns
df = df.rename(columns={
    "Date": "date",
    "Open": "open_price",
    "High": "high_price",
    "Low": "low_price",
    "Close": "close_price",
    "Volume": "volume"
})

# Add company_id (AAPL = 1)
df["company_id"] = 1

# Convert date format
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

# Ensure numeric columns are correct
numeric_cols = ["open_price", "high_price", "low_price", "close_price", "volume"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Drop rows with missing values
df = df.dropna()

# Sort by date
df = df.sort_values(by="date")

# Save cleaned file
df.to_csv("data_cleaned/aapl_us_clean.csv", index=False)

print("Cleaned file saved successfully")
