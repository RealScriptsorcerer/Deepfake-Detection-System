from __future__ import annotations
from typing import Any, Dict, List, Optional
import sqlite3
import json
import os
import time


DB_PATH = os.environ.get("DEEPFAKE_DB", "/workspace/deepfake_results.sqlite")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            modality TEXT NOT NULL,
            filename TEXT,
            label TEXT NOT NULL,
            score REAL NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


_CONN = _connect()


def save_result(modality: str, filename: str, label: str, score: float, payload: Dict[str, Any]) -> int:
    cur = _CONN.cursor()
    cur.execute(
        "INSERT INTO results(created_at, modality, filename, label, score, payload) VALUES (?, ?, ?, ?, ?, ?)",
        (time.time(), modality, filename, label, float(score), json.dumps(payload)),
    )
    _CONN.commit()
    return int(cur.lastrowid)


def list_results(limit: int = 50) -> List[Dict[str, Any]]:
    cur = _CONN.cursor()
    cur.execute("SELECT id, created_at, modality, filename, label, score FROM results ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "created_at": r[1],
            "modality": r[2],
            "filename": r[3],
            "label": r[4],
            "score": r[5],
        }
        for r in rows
    ]


def get_result(result_id: int) -> Optional[Dict[str, Any]]:
    cur = _CONN.cursor()
    cur.execute("SELECT id, created_at, modality, filename, label, score, payload FROM results WHERE id = ?", (result_id,))
    row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "created_at": row[1],
        "modality": row[2],
        "filename": row[3],
        "label": row[4],
        "score": row[5],
        "payload": json.loads(row[6] or "{}"),
    }

