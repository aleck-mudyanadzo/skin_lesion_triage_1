import sqlite3, csv

conn = sqlite3.connect("instance/triage_logs.db")

for table in ["survey_responses", "predictions"]:
    cur = conn.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    with open(f"{table}_export.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    print(f"Exported {len(rows)} rows from {table} to {table}_export.csv")
