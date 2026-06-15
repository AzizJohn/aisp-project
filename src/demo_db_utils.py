"""Introspect a SQLite DB into the training schema-string format:  table(col1, col2, ...)"""
import sqlite3

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