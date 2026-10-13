import sqlite3
conn = sqlite3.connect("instance/triage_logs.db")
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(tables)
for t in tables:
    cols = [d[1] for d in conn.execute(f"PRAGMA table_info({t})").fetchall()]
    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"{t}: {count} rows, columns={cols}")
