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

os.makedirs("outputs/charts", exist_ok=True)

sector_colours = {
    "Technology": "#1f4e79",
    "Finance": "#7f6000",
    "Energy": "#38761d",
    "Healthcare": "#741b47"
}

plt.figure(figsize=(10, 6))

for sector, group in df.groupby("sector_name"):
    plt.scatter(
        group["volatility_pct"],
        group["avg_daily_return_pct"],
        s=100,
        label=sector,
        color=sector_colours.get(sector)
    )

for _, row in df.iterrows():
    plt.text(
        row["volatility_pct"],
        row["avg_daily_return_pct"],
        row["company_name"],
        fontsize=9,
    )

plt.xlabel("Volatility (%)")
plt.ylabel("Average Daily Return (%)")
plt.title("Risk vs Return by Company")
plt.legend()

plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/charts/risk_vs_return.png", dpi=300, bbox_inches="tight")
plt.show()
