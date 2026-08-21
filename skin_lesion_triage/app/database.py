"""
database.py
Lightweight SQLite logging of anonymous usage/prediction events, used later
for the usability/evaluation phase (Objective 4) and for basic auditability.
No personally identifiable information is stored — filenames are hashed.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone

from config import Config


def get_connection():
    Config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the logging table if it doesn't already exist. Safe to call
    on every app startup."""
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
    conn.commit()
    conn.close()


def hash_bytes(data: bytes) -> str:
    """One-way hash of the raw image bytes — lets us dedupe / audit without
    storing anything identifiable."""
    return hashlib.sha256(data).hexdigest()[:16]


def log_prediction(image_bytes: bytes, model_used: str, result: dict, sharpness_score: float):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO predictions
            (timestamp, image_hash, model_used, predicted_label,
             malignant_probability, sharpness_score, risk_flag)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            hash_bytes(image_bytes),
            model_used,
            result["label"],
            result["malignant_probability"],
            sharpness_score,
            result["risk_flag"],
        ),
    )
    conn.commit()
    conn.close()


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
