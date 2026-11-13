"""Local SQLite storage for fictional/demo records only."""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "healthcare_demo.db"

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS demo_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            model_output INTEGER NOT NULL,
            model_probability REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

def save_prediction(label, output, probability):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "INSERT INTO demo_predictions(label, model_output, model_probability, created_at) VALUES (?, ?, ?, ?)",
            (label, int(output), float(probability), datetime.now(timezone.utc).isoformat())
        )
        return cur.lastrowid

def read_predictions():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(
            "SELECT * FROM demo_predictions ORDER BY id DESC LIMIT 250"
        ).fetchall()]
