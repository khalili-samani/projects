import pandas as pd
import matplotlib.pyplot as plt
import os
from sqlalchemy import create_engine

# Replace with your password
password = "your password"

engine = create_engine(f"mysql+pymysql://root:{password}@localhost/market_analysis")

query = "SELECT * FROM vw_company_summary;"
df = pd.read_sql(query, engine)

print(df.head())
print(df.shape)

df = df.sort_values("avg_daily_return_pct", ascending=False)

os.makedirs("outputs/charts", exist_ok=True)

# Create color list based on values
colors = ["#8B0000" if x < 0 else "#006400" for x in df["avg_daily_return_pct"]]

plt.figure(figsize=(10, 6))
plt.bar(df["company_name"], df["avg_daily_return_pct"], color=colors)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average Daily Return (%)")
plt.xlabel("Company")
plt.title("Average Daily Return by Company")
plt.tight_layout()
plt.savefig("outputs/charts/avg_daily_return_by_company.png", dpi=300)
plt.show()
