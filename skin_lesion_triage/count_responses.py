import sqlite3
conn = sqlite3.connect("instance/triage_logs.db")
count = conn.execute("SELECT COUNT(*) FROM survey_responses").fetchone()[0]
print(f"Total survey responses so far: {count}")
