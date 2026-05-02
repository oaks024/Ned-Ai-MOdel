"""Per-session conversational memory backed by SQLite.

Two tables:
  - ``profile``: one row per session_id holding user attributes used to
    personalize answers (name, interested program, etc.).
  - ``history``: append-only log of user/assistant messages so we can replay
    the recent conversation back to the model.

Memory is intentionally separate from the RAG vector store. Profile facts
must NEVER be treated as authoritative admission data — they only personalize.
"""
import os
import sqlite3
import json
import datetime


class SessionMemory:
    def __init__(self, db_path: str):
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS profile (
                    session_id TEXT PRIMARY KEY,
                    user_name TEXT,
                    interested_program TEXT,
                    education_level TEXT,
                    category TEXT,
                    preferences TEXT,
                    updated_at TEXT
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TEXT
                )"""
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_session ON history(session_id, id)"
            )
            c.commit()

    # --------------- profile ---------------
    def get_profile(self, session_id: str) -> dict:
        with self._conn() as c:
            row = c.execute(
                """SELECT user_name, interested_program, education_level, category, preferences
                   FROM profile WHERE session_id=?""",
                (session_id,),
            ).fetchone()
        if not row:
            return {}
        return {
            "user_name": row[0],
            "interested_program": row[1],
            "education_level": row[2],
            "category": row[3],
            "preferences": json.loads(row[4]) if row[4] else {},
        }

    def update_profile(self, session_id: str, **fields) -> None:
        current = self.get_profile(session_id)
        for k, v in fields.items():
            if v is not None:
                current[k] = v
        prefs = current.get("preferences") or {}
        if not isinstance(prefs, dict):
            prefs = {}
        now = datetime.datetime.utcnow().isoformat()
        with self._conn() as c:
            c.execute(
                """INSERT INTO profile(session_id, user_name, interested_program,
                                       education_level, category, preferences, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       user_name=excluded.user_name,
                       interested_program=excluded.interested_program,
                       education_level=excluded.education_level,
                       category=excluded.category,
                       preferences=excluded.preferences,
                       updated_at=excluded.updated_at""",
                (
                    session_id,
                    current.get("user_name"),
                    current.get("interested_program"),
                    current.get("education_level"),
                    current.get("category"),
                    json.dumps(prefs),
                    now,
                ),
            )
            c.commit()

    # --------------- history ---------------
    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO history(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, datetime.datetime.utcnow().isoformat()),
            )
            c.commit()

    def get_history(self, session_id: str, limit: int = 10) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT role, content FROM history
                   WHERE session_id=? ORDER BY id DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return list(reversed([{"role": r, "content": cnt} for r, cnt in rows]))

    def clear_session(self, session_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM profile WHERE session_id=?", (session_id,))
            c.execute("DELETE FROM history WHERE session_id=?", (session_id,))
            c.commit()
