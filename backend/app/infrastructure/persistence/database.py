"""SQLite schema and connection management.

Uses stdlib `sqlite3` rather than SQLModel/ORM on purpose: the data model
is tiny and conversation/usage writes live on the streaming hot path where
we want predictable latency and no session bookkeeping.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations (updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    citations_json TEXT,
    usage_json TEXT,
    latency_ms INTEGER,
    mode TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, created_at);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    source_path TEXT NOT NULL,
    summary TEXT,
    tag TEXT NOT NULL DEFAULT 'other',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_updated
    ON documents (updated_at DESC);

CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    mode TEXT,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms INTEGER,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_created
    ON usage_events (created_at DESC);
"""


class Database:
    """Lightweight SQLite wrapper with per-thread connections and foreign keys."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            _migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self._path,
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
        return conn

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        conn = self._connect()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations for databases created by older versions."""

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(usage_events)")}
    if "request_id" not in columns:
        # One chat request now produces one usage row per model (chat, utility,
        # embeddings); request_id groups them.
        conn.execute("ALTER TABLE usage_events ADD COLUMN request_id TEXT")


def create_database(database_url: str, base_dir: Path) -> Database:
    """Parse a sqlite URL into a Database instance rooted at base_dir if relative."""

    if database_url.startswith("sqlite:///"):
        raw = database_url.removeprefix("sqlite:///")
    elif database_url.startswith("sqlite://"):
        raw = database_url.removeprefix("sqlite://")
    else:
        raw = database_url
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return Database(path)
