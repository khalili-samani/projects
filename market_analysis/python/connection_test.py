import pandas as pd
from sqlalchemy import create_engine

# Replace with your password
password = "your password"

engine = create_engine(f"mysql+pymysql://root:{password}@localhost/market_analysis")

query = "SELECT * FROM vw_company_summary;"
df = pd.read_sql(query, engine)

print(df.head())
print(df.shape)
