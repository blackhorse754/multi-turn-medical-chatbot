# cleaned pipeline_logging.py
import os
import sys
import sqlite3
from typing import Optional, Mapping, Dict, Any, List
from pathlib import Path
import json
import traceback
import time
from datetime import datetime, timezone

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.constants as const
import scripts.utils as u


# ------------------------------------------------------------------
# DB helpers (with auto-migration to add `event_text` + create new tables)
# ------------------------------------------------------------------

COMBINED_DDL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- messages + event_text
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT,
    session_id TEXT,
    role TEXT,
    content TEXT,
    timestamp TEXT,
    output_tokens INTEGER,
    input_tokens INTEGER,
    event_text TEXT
);

-- retrieved_sources
CREATE TABLE IF NOT EXISTS retrieved_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    source TEXT,
    pages TEXT,
    snippets TEXT,
    labels TEXT,
    inserted_at TEXT
);

-- consolidated session records
CREATE TABLE IF NOT EXISTS session_records (
    user_name TEXT,
    session_id TEXT,
    question TEXT,
    rephrased_question TEXT,
    answer TEXT,
    metadata TEXT,
    retrieved_sources TEXT,
    inserted_at TEXT
);

-- timings
CREATE TABLE IF NOT EXISTS timings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT,
    session_id TEXT,
    moderation_time_s FLOAT,
    rewrite_time_s FLOAT,
    classification_time_s FLOAT,
    qa_time_s FLOAT
);

-- session_errors (error breakdown per step)
CREATE TABLE IF NOT EXISTS session_errors (
    user_name TEXT NOT NULL,
    session_id TEXT NOT NULL,

    moderation_error_type        TEXT,
    rewrite_error_type           TEXT,
    classification_error_type    TEXT,
    qa_error_type                TEXT,
    topics_error_type            TEXT,
    unknown_error_type           TEXT,

    moderation_error_message     TEXT,
    rewrite_error_message        TEXT,
    classification_error_message TEXT,
    qa_error_message             TEXT,
    topics_error_message         TEXT,
    unknown_error_message        TEXT,

    moderation_error_traceback     TEXT,
    rewrite_error_traceback        TEXT,
    classification_error_traceback TEXT,
    qa_error_traceback             TEXT,
    topics_error_traceback         TEXT,
    unknown_error_traceback        TEXT
);

-- sessions (session metadata)
CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    user_name         TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    ended_at          TEXT NOT NULL,
    duration_seconds  REAL NOT NULL,
    auto_logout       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_name ON sessions(user_name);
CREATE INDEX IF NOT EXISTS idx_sessions_ended_at ON sessions(ended_at);

-- pipeline logs
CREATE TABLE IF NOT EXISTS pipeline_logs (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    session_id      TEXT NOT NULL,
    user_name       TEXT,
    step_order      INTEGER NOT NULL DEFAULT 0,
    step_name       TEXT NOT NULL,
    step_stage      TEXT,
    status          TEXT NOT NULL,
    details         TEXT,
    error_text      TEXT
);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_session ON pipeline_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_created_at ON pipeline_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_step ON pipeline_logs(step_name);
"""


def _ensure_db_dir(path: str):
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def init_db(db_path: str = const.DB_PATH, timeout: int = 30):
    """
    Ensure the DB exists and core + pipeline tables are present.
    Idempotent and safe to run at startup.
    """
    _ensure_db_dir(db_path)
    try:
        with sqlite3.connect(db_path, timeout=timeout) as conn:
            conn.executescript(COMBINED_DDL)
            conn.commit()
    except Exception:
        print("init_db: failed to ensure combined DB schema")
        traceback.print_exc()
        raise


def insert_session_errors(
    user_name: str,
    session_id: str,
    errors: Mapping[str, Any],
    db_path: str = const.DB_PATH,
) -> None:
    """
    Insert one row into session_errors. Missing keys in `errors` will be NULL.
    If `errors` is empty, just insert user/session columns.
    """
    if errors:
        error_columns = list(errors.keys())
        cols_part = ", ".join(error_columns)
        placeholders = ", ".join(["?"] * len(error_columns))
        insert_sql = f"""
        INSERT INTO session_errors (
            user_name,
            session_id,
            {cols_part}
        ) VALUES (
            ?, ?,
            {placeholders}
        );
        """
        values = [user_name, session_id] + [errors.get(col) for col in error_columns]
    else:
        insert_sql = """
        INSERT INTO session_errors (user_name, session_id) VALUES (?, ?);
        """
        values = [user_name, session_id]

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(insert_sql, values)
            conn.commit()
    except sqlite3.OperationalError as e:
        # If schema missing, ensure DB and retry once
        if "no such table" in str(e).lower():
            init_db(db_path)
            with sqlite3.connect(db_path) as conn:
                conn.execute(insert_sql, values)
                conn.commit()
        else:
            raise


# ------------------------------------------------------------------
# Session-level record persistence (DB + JSON)
# ------------------------------------------------------------------
def save_session_record(
    user_name: str,
    session_id: str,
    sessions: Mapping[str, Any],
    db_path: str = const.DB_PATH,
) -> None:
    """Save a consolidated session record into SQLite."""

    question = sessions.get("question", "")
    rephrased_question = sessions.get("rephrased_question", "")
    answer = sessions.get("answer", "")

    metadata_obj = sessions.get("metadata", {})
    retrieved_sources_obj = sessions.get("retrieved_sources", [])

    metadata_json = (
        metadata_obj if isinstance(metadata_obj, str) else json.dumps(metadata_obj, ensure_ascii=False)
    )
    sources_json = (
        retrieved_sources_obj if isinstance(retrieved_sources_obj, str) else json.dumps(retrieved_sources_obj, ensure_ascii=False)
    )

    now = const.ist_timestamp()

    sql = """
        INSERT INTO session_records
            (user_name, session_id, question, rephrased_question, answer, metadata, retrieved_sources, inserted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(sql, (
                user_name,
                session_id,
                question,
                rephrased_question,
                answer,
                metadata_json,
                sources_json,
                now,
            ))
            conn.commit()
    except Exception as e:
        u.log_exception_to_db(e, context_msg="Failed to save session record",
                              user_name=user_name, session_id=session_id)


def insert_session_metadata(record: Dict[str, Any], db_path: str = const.DB_PATH):
    """
    Insert or update a session row. Expects keys:
      - session_id (str)
      - user_name (str)
      - started_at (str | datetime)
      - ended_at (str | datetime)
      - duration_seconds (float | int)
      - auto_logout (bool | int)
    """
    required = ["session_id", "user_name", "started_at", "ended_at", "duration_seconds", "auto_logout"]
    missing = [k for k in required if k not in record]
    if missing:
        raise ValueError(f"insert_session_metadata: missing keys: {missing}")

    def _iso(v):
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)

    session_id = str(record["session_id"])
    user_name = str(record["user_name"])
    started_at = _iso(record["started_at"])
    ended_at = _iso(record["ended_at"])
    duration_seconds = float(record["duration_seconds"])
    auto_logout = 1 if bool(record["auto_logout"]) else 0

    sql = """
        INSERT INTO sessions (
            session_id, user_name, started_at, ended_at, duration_seconds, auto_logout
        ) VALUES (?, ?, ?, ?, ?, ?);
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(sql, (session_id, user_name, started_at, ended_at, duration_seconds, auto_logout))
        conn.commit()


def insert_timings(user_name, session_id, timings, db_path: str = const.DB_PATH):
    """
    Inserts a new record into the timings table.
    Expects timings dict with keys:
      'moderation_time_s', 'rewrite_time_s', 'classification_time_s', 'qa_time_s'
    """
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO timings (
                user_name, session_id, moderation_time_s, rewrite_time_s, classification_time_s, qa_time_s
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            (user_name, session_id, timings['moderation_time_s'], timings['rewrite_time_s'],
             timings['classification_time_s'], timings['qa_time_s'])
        )
        conn.commit()
    except Exception as e:
        u.log_exception_to_db(e, user_name=user_name, session_id=session_id)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ------------------------------------------------------------------
# Event logging helpers (DB-backed)
# ------------------------------------------------------------------

def save_message_to_db(
    user_name: Optional[str],
    session_id: Optional[str],
    role: str,
    content: str,
    output_tokens: Optional[int] = None,
    input_tokens: Optional[int] = None,
    timestamp: Optional[str] = None,
    db_path: Optional[str] = const.DB_PATH,
    event_text: Optional[str] = None,
):
    """Insert a single message into the DB."""
    if timestamp is None:
        timestamp = time.time()
    timestamp = const.ist_timestamp()
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO messages (
                    user_name, session_id, role, content, timestamp, output_tokens, input_tokens, event_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(user_name) if user_name is not None else None,
                    str(session_id) if session_id is not None else None,
                    str(role) if role is not None else None,
                    str(content) if content is not None else None,
                    timestamp,
                    int(output_tokens) if isinstance(output_tokens, int) else None,
                    int(input_tokens) if isinstance(input_tokens, int) else None,
                    str(event_text) if event_text is not None else None,
                ),
            )
            conn.commit()
    except Exception as e:
        u.log_exception_to_db(e, context_msg='Failed to save message to DB', user_name=user_name, session_id=session_id)
        print("Failed to save message to DB:", e)


def save_retrieved_sources_to_db(groups: Dict[str, Any], session_id: str, db_path: Optional[str] = const.DB_PATH) -> Optional[str]:
    """
    Save retrieved sources grouped dict -> rows in retrieved_sources.
    Returns db_path on success or None on failure.
    """
    try:
        #now = datetime.utcnow().isoformat() + "Z"
        now = const.ist_timestamp()
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            for source, details in groups.items():
                db_source = None if source == "__unknown__" else str(source)
                pages = details.get('pages', [])
                snippets = details.get('snippets', [])
                labels = details.get('labels', [])
                cur.execute(
                    "INSERT INTO retrieved_sources (session_id, source, pages, snippets, labels, inserted_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        str(session_id),
                        db_source,
                        json.dumps(pages, ensure_ascii=False),
                        json.dumps(snippets, ensure_ascii=False),
                        json.dumps(labels, ensure_ascii=False),
                        now,
                    ),
                )
            conn.commit()
        return db_path
    except Exception as e:
        u.log_exception_to_db(e, context_msg='Failed to save retrieved_sources to DB', session_id=session_id)
        return None


def _json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False)
        except Exception:
            return "{}"


def _write_fallback(record: Dict[str, Any]):
    """Write the record to a JSON fallback file when DB is unavailable."""
    try:
        os.makedirs(const.FALLBACK_DIR, exist_ok=True)
        ts = int(time.time())
        fname = f"pipeline_log_fallback_{record.get('session_id','no_session')}_{ts}.json"
        path = os.path.join(const.FALLBACK_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        print(f"pipeline_logging: wrote fallback log to {path}")
    except Exception:
        print("pipeline_logging: failed to write fallback file")
        traceback.print_exc()


def log_step(
    timestamp: str,
    user_name: str,
    session_id: str,
    step_name: str,
    step_stage: Optional[str] = None,
    status: str = "started",
    details: Optional[Dict[str, Any]] = None,
    error_text: Optional[str] = None,
    db_path: str = const.DB_PATH,
    timeout: int = 30,
) -> None:
    """
    Append a pipeline step record.
    """
    details_json = _json_dumps_safe(details or {})

    record_for_fallback = {
        "created_at": timestamp,
        "session_id": session_id,
        "user_name": user_name,
        "step_name": step_name,
        "step_stage": step_stage,
        "status": status,
        "details": details or {},
        "error_text": error_text,
    }

    try:
        with sqlite3.connect(db_path, timeout=timeout) as conn:
            cur = conn.cursor()

            cur.execute("SELECT COALESCE(MAX(step_order), -1) + 1 FROM pipeline_logs WHERE session_id = ?;", (session_id,))
            row = cur.fetchone()
            next_order = int(row[0]) if row and row[0] is not None else 0

            insert_sql = """
            INSERT INTO pipeline_logs
            (created_at, session_id, user_name, step_order, step_name, step_stage, status, details, error_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            cur.execute(
                insert_sql,
                (
                    timestamp,
                    session_id,
                    user_name,
                    next_order,
                    step_name,
                    step_stage,
                    status,
                    details_json,
                    error_text,
                ),
            )
            conn.commit()
    except Exception:
        print("Error persisting pipeline log to DB; falling back to file. Record:")
        traceback.print_exc()
        _write_fallback(record_for_fallback)

