from __future__ import annotations
from typing import Any, Dict, List, Optional
import sqlite3
import json
import os
import time


DB_PATH = os.environ.get("DEEPFAKE_DB", "/workspace/deepfake_results.sqlite")
SAVE_PAYLOADS = os.environ.get("DF_SAVE_PAYLOADS", "1") not in ("0", "false", "False")


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
            payload TEXT NOT NULL,
            truth_label TEXT
        )
        """
    )
    # Ensure truth_label column exists for old DBs
    try:
        cur = conn.execute("PRAGMA table_info(results)")
        cols = [r[1] for r in cur.fetchall()]
        if "truth_label" not in cols:
            conn.execute("ALTER TABLE results ADD COLUMN truth_label TEXT")
            conn.commit()
    except Exception:
        pass
    conn.commit()
    return conn


_CONN = _connect()


def save_result(modality: str, filename: str, label: str, score: float, payload: Dict[str, Any]) -> int:
    cur = _CONN.cursor()
    stored_payload = payload if SAVE_PAYLOADS else {"modality": modality, "label": label, "score": float(score)}
    cur.execute(
        "INSERT INTO results(created_at, modality, filename, label, score, payload) VALUES (?, ?, ?, ?, ?, ?)",
        (time.time(), modality, filename, label, float(score), json.dumps(stored_payload)),
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
    cur.execute("SELECT id, created_at, modality, filename, label, score, payload, truth_label FROM results WHERE id = ?", (result_id,))
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
        "truth_label": row[7],
    }


def set_truth_label(result_id: int, truth_label: str) -> bool:
    cur = _CONN.cursor()
    cur.execute("UPDATE results SET truth_label = ? WHERE id = ?", (truth_label, result_id))
    _CONN.commit()
    return cur.rowcount > 0


def compute_metrics() -> Dict[str, Any]:
    cur = _CONN.cursor()
    cur.execute("SELECT label, truth_label FROM results WHERE truth_label IS NOT NULL")
    rows = cur.fetchall()
    tp = fp = tn = fn = 0
    for pred, truth in rows:
        if truth not in ("real", "fake") or pred not in ("real", "fake"):
            continue
        if pred == "fake" and truth == "fake":
            tp += 1
        elif pred == "fake" and truth == "real":
            fp += 1
        elif pred == "real" and truth == "real":
            tn += 1
        elif pred == "real" and truth == "fake":
            fn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total},
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    }

