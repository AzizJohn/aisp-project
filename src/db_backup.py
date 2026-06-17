"""
Backup / reset YOUR OWN dummy databases so the live destructive demo is repeatable.
  python src/db_backup.py backup "postgresql+psycopg2://u:p@localhost:5432/dummy_shop" dummy_shop.bak
  python src/db_backup.py reset  "postgresql+psycopg2://u:p@localhost:5432/dummy_shop" dummy_shop.bak
(For FK-heavy schemas, native pg_dump/mysqldump is more robust.)
"""
import sys, pickle
import sqlalchemy as sa

def backup_db(conn_str, out_path):
    eng = sa.create_engine(conn_str)
    meta = sa.MetaData(); meta.reflect(bind=eng)
    snap = {}
    with eng.connect() as con:
        for name, tbl in meta.tables.items():
            rows = [dict(r._mapping) for r in con.execute(sa.select(tbl))]
            ddl = str(sa.schema.CreateTable(tbl).compile(eng))
            snap[name] = {"ddl": ddl, "rows": rows, "cols": [c.name for c in tbl.columns]}
    eng.dispose()
    pickle.dump(snap, open(out_path, "wb"))
    print(f"Backed up {len(snap)} tables -> {out_path}")

def reset_db(conn_str, backup_path):
    snap = pickle.load(open(backup_path, "rb"))
    eng = sa.create_engine(conn_str); names = list(snap.keys())
    with eng.begin() as con:
        for name in reversed(names):
            con.execute(sa.text(f"DROP TABLE IF EXISTS {name}"))
        for name in names:
            con.execute(sa.text(snap[name]["ddl"]))
        for name in names:
            d = snap[name]
            if d["rows"]:
                vals = ", ".join(f":{c}" for c in d["cols"])
                con.execute(sa.text(f"INSERT INTO {name} ({', '.join(d['cols'])}) VALUES ({vals})"), d["rows"])
    eng.dispose()
    print(f"Reset {len(names)} tables from {backup_path}")

if __name__ == "__main__":
    cmd, conn, path = sys.argv[1], sys.argv[2], sys.argv[3]
    (backup_db if cmd == "backup" else reset_db)(conn, path)