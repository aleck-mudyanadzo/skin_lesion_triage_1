"""
database.py
SQLite logging of prediction events. The predictions table stores a
one-way hash of the uploaded image (never the image itself, by default)
plus the two result images saved to disk, retrievable later only by a
random, unguessable reference_code - not the sequential row id, which
would let someone enumerate and view other people's results.
"""

import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone

from config import Config


def get_connection():
    Config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn, table, column, coltype):
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_hash TEXT NOT NULL,
            model_used TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            malignant_probability REAL NOT NULL,
            sharpness_score REAL,
            risk_flag TEXT NOT NULL
        )
        """
    )
    _add_column_if_missing(conn, "predictions", "reference_code", "TEXT")
    _add_column_if_missing(conn, "predictions", "original_image_path", "TEXT")
    _add_column_if_missing(conn, "predictions", "heatmap_image_path", "TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_reference_code ON predictions(reference_code)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS survey_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            q1 INTEGER NOT NULL,
            q2 INTEGER NOT NULL,
            q3 INTEGER NOT NULL,
            q4 INTEGER NOT NULL,
            q5 INTEGER NOT NULL,
            q6 INTEGER NOT NULL,
            q7 INTEGER NOT NULL,
            q8 INTEGER NOT NULL,
            q9 INTEGER NOT NULL,
            q10 INTEGER NOT NULL,
            sus_score REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


SUS_ODD_ITEMS = {1, 3, 5, 7, 9}
SUS_EVEN_ITEMS = {2, 4, 6, 8, 10}


def compute_sus_score(answers: dict) -> float:
    total = 0
    for i in range(1, 11):
        rating = answers[i]
        if i in SUS_ODD_ITEMS:
            total += rating - 1
        else:
            total += 5 - rating
    return round(total * 2.5, 1)


def log_survey_response(answers: dict, sus_score: float):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO survey_responses
            (timestamp, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, sus_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            answers[1], answers[2], answers[3], answers[4], answers[5],
            answers[6], answers[7], answers[8], answers[9], answers[10],
            sus_score,
        ),
    )
    conn.commit()
    conn.close()


def get_survey_summary():
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS n, AVG(sus_score) AS mean_score FROM survey_responses"
    ).fetchone()
    conn.close()
    return {
        "respondent_count": row["n"],
        "mean_sus_score": round(row["mean_score"], 1) if row["mean_score"] is not None else None,
    }


def log_prediction(
    image_bytes: bytes,
    model_used: str,
    result: dict,
    sharpness_score: float,
    original_png_bytes: bytes = None,
    heatmap_png_bytes: bytes = None,
) -> dict:
    """Logs a prediction. If the two result images are supplied, saves them
    to disk under a random reference_code, so the person can view them again
    later via that code. Returns {"id": ..., "reference_code": ...} -
    reference_code is None if images were not supplied (nothing retained)."""
    reference_code = None
    original_path = None
    heatmap_path = None

    if original_png_bytes is not None and heatmap_png_bytes is not None:
        reference_code = secrets.token_urlsafe(8)
        Config.RESULT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        original_path = str(Config.RESULT_IMAGE_DIR / f"{reference_code}_original.png")
        heatmap_path = str(Config.RESULT_IMAGE_DIR / f"{reference_code}_heatmap.png")
        with open(original_path, "wb") as f:
            f.write(original_png_bytes)
        with open(heatmap_path, "wb") as f:
            f.write(heatmap_png_bytes)

    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO predictions
            (timestamp, image_hash, model_used, predicted_label,
             malignant_probability, sharpness_score, risk_flag,
             reference_code, original_image_path, heatmap_image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            hash_bytes(image_bytes),
            model_used,
            result["label"],
            result["malignant_probability"],
            sharpness_score,
            result["risk_flag"],
            reference_code,
            original_path,
            heatmap_path,
        ),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"id": row_id, "reference_code": reference_code}


def get_prediction_by_reference_code(reference_code: str):
    """Looks up a prediction by its random reference_code - never by the
    sequential id, since that would let the images be enumerated."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM predictions WHERE reference_code = ?", (reference_code,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_recent_logs(limit: int = 50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_summary_stats():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
    urgent = conn.execute(
        "SELECT COUNT(*) AS c FROM predictions WHERE risk_flag = 'urgent_referral'"
    ).fetchone()["c"]
    conn.close()
    return {"total_predictions": total, "urgent_referrals": urgent}
