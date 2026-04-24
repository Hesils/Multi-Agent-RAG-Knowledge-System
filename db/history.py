import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.environ["DB_PATH"])


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript("""
                           CREATE TABLE IF NOT EXISTS conversations (
                                                                        id TEXT PRIMARY KEY,
                                                                        title TEXT NOT NULL,
                                                                        created_at TEXT NOT NULL,
                                                                        updated_at TEXT NOT NULL
                           );

                           CREATE TABLE IF NOT EXISTS messages (
                                                                   id TEXT PRIMARY KEY,
                                                                   conversation_id TEXT NOT NULL,
                                                                   role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                               content TEXT NOT NULL,
                               step_logs TEXT,
                               created_at TEXT NOT NULL,
                               FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                               );
                           """)


def create_conversation(first_message: str) -> str:
    cid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    title = first_message[:60] + ("..." if len(first_message) > 60 else "")
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (cid, title, now, now),
        )
    return cid


def add_message(
        conversation_id: str,
        role: str,
        content: str,
        step_logs: Optional[str] = None,
) -> str:
    mid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, step_logs, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (mid, conversation_id, role, content, step_logs, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
    return mid


def get_conversations() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_messages(conversation_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_conversation(conversation_id: str):
    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def rename_conversation(conversation_id: str, new_title: str):
    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (new_title, conversation_id),
        )