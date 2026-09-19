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
    """Creates the logging tables if they don't already exist. Safe to call
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
    """One-way hash of the raw image bytes — lets us dedupe / audit without
    storing anything identifiable."""
    return hashlib.sha256(data).hexdigest()[:16]


# System Usability Scale (Brooke, 1996). Odd-numbered statements are worded
# positively, even-numbered ones negatively — this is deliberate in the
# original instrument, alternating the wording keeps respondents reading
# each statement rather than clicking the same column down the page.
SUS_ODD_ITEMS = {1, 3, 5, 7, 9}
SUS_EVEN_ITEMS = {2, 4, 6, 8, 10}


def compute_sus_score(answers: dict) -> float:
    """
    answers: dict mapping question number (1-10) to a Likert rating (1-5).
    Standard SUS scoring: each odd-numbered item contributes (rating - 1),
    each even-numbered item contributes (5 - rating); the ten contributions
    are summed and multiplied by 2.5, giving a score out of 100.
    """
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
