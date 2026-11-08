"""
Shared database utilities.

Provides a SQLAlchemy engine built from DB_CONFIG so pandas read_sql
works without warnings, alongside raw mysql.connector connections for
bulk inserts (executemany).
"""

from urllib.parse import quote_plus
from sqlalchemy import create_engine
from src.config import DB_CONFIG


def get_engine():
    """
    Return a SQLAlchemy engine for use with pd.read_sql.

    quote_plus encodes special characters in the password (e.g. @, #, !)
    so they don't break the connection URL parser.
    """
    cfg = DB_CONFIG
    user     = quote_plus(cfg["user"])
    password = quote_plus(cfg["password"])  # safe even if empty or has symbols
    host     = cfg["host"]
    port     = cfg["port"]
    database = cfg["database"]

    url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url)