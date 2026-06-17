"""Introspect a SQLite DB into the training schema-string format:  table(col1, col2, ...)"""
import sqlite3
from sqlalchemy import create_engine, inspect, text

def get_db_schema_string(db_path):
    con = sqlite3.connect(db_path); cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    lines = []
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [row[1] for row in cur.fetchall()]
        lines.append(f"  {t}({', '.join(cols)})")
    con.close()
    return "\n".join(lines)

def get_schema_string_any(conn_str):
    """
    Introspect ANY SQL database into the training schema-string format.
    conn_str examples (USE ONLY on your own throwaway test DBs with synthetic data):
      sqlite:////home/6673/projects/aisp-project/demo_dbs/bank.db
      postgresql+psycopg2://user:pass@localhost:5432/styledemo_test
      mysql+pymysql://user:pass@localhost:3306/styledemo_test
    """
    eng = create_engine(conn_str)
    insp = inspect(eng)
    lines = [f"  {t}({', '.join(c['name'] for c in insp.get_columns(t))})"
             for t in insp.get_table_names()]
    eng.dispose()
    return "\n".join(lines)