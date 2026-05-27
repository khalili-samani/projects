"""
Database initialisation utilities.
"""

from pathlib import Path
import mysql.connector

from src.config import DB_CONFIG


def initialise_database(
    schema_path="sql/create_tables.sql",
):
    schema_file = Path(schema_path)

    if not schema_file.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_file}"
        )

    # Connect without specifying database so we can CREATE it
    connection_config = {
        "host": DB_CONFIG["host"],
        "port": DB_CONFIG["port"],
        "user": DB_CONFIG["user"],
        "password": DB_CONFIG["password"],
    }

    conn = mysql.connector.connect(**connection_config)
    cursor = conn.cursor()

    sql_script = schema_file.read_text(encoding="utf-8")

    # Execute each statement individually, skipping blanks
    for command in sql_script.split(";"):
        command = command.strip()
        if command:
            cursor.execute(command)

    conn.commit()
    cursor.close()
    conn.close()

    print("[Database] Schema initialised successfully.")