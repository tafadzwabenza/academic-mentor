import json
import sqlite3
from typing import Any, Dict, List, Optional

DB_PATH = "platform_database.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_platform_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                student_name TEXT,
                student_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                current_stage TEXT,
                chat_history TEXT,
                draft_content TEXT,
                notebook_chat_history TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );

            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                extracted_text TEXT,
                preview_summary TEXT,
                title TEXT,
                authors TEXT,
                summary TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            );
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO users (id, name, email)
            VALUES ('student_123', 'Demo Student', 'student@example.com')
            """
        )
        _migrate_platform_schema(conn)
        conn.commit()

    existing = get_user_projects("student_123")
    if not existing:
        create_project("student_123", "Thesis: Boardroom Dynamics")
        create_project("student_123", "Assignment: Drug Impact")


def _migrate_platform_schema(conn: sqlite3.Connection) -> None:
    user_columns = {
        "student_name": "TEXT",
        "student_level": "TEXT",
    }
    project_columns = {
        "current_stage": "TEXT",
        "chat_history": "TEXT",
        "draft_content": "TEXT",
        "notebook_chat_history": "TEXT",
    }
    for column, col_type in user_columns.items():
        if column not in _table_columns(conn, "users"):
            conn.execute(f"ALTER TABLE users ADD COLUMN {column} {col_type}")
    for column, col_type in project_columns.items():
        if column not in _table_columns(conn, "projects"):
            conn.execute(f"ALTER TABLE projects ADD COLUMN {column} {col_type}")
    source_columns = {
        "title": "TEXT",
        "authors": "TEXT",
        "summary": "TEXT",
        "file_path": "TEXT",
    }
    for column, col_type in source_columns.items():
        if column not in _table_columns(conn, "sources"):
            conn.execute(f"ALTER TABLE sources ADD COLUMN {column} {col_type}")
    conn.execute(
        """
        UPDATE sources
        SET title = COALESCE(title, filename),
            authors = COALESCE(authors, 'Unknown'),
            summary = COALESCE(summary, preview_summary),
            file_path = COALESCE(file_path, filename)
        WHERE title IS NULL OR authors IS NULL OR summary IS NULL OR file_path IS NULL
        """
    )


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, name, email, student_name, student_level, created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def update_user_profile(user_id: str, name: str, level: Optional[str]) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE users
            SET student_name = ?, student_level = ?
            WHERE id = ?
            """,
            (name, level, user_id),
        )
        conn.commit()


def save_project_state(
    project_id: int,
    stage: Optional[str],
    messages_list: List[Dict[str, Any]],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE projects
            SET current_stage = ?, chat_history = ?
            WHERE id = ?
            """,
            (stage, json.dumps(messages_list), project_id),
        )
        conn.commit()


def save_notebook_state(
    project_id: int,
    messages_list: List[Dict[str, Any]],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE projects
            SET notebook_chat_history = ?
            WHERE id = ?
            """,
            (json.dumps(messages_list), project_id),
        )
        conn.commit()


def get_project_state(project_id: int) -> Dict[str, Any]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT current_stage, chat_history, draft_content, notebook_chat_history
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        ).fetchone()
    if row is None:
        return {
            "current_stage": None,
            "chat_history": [],
            "draft_content": "",
            "notebook_chat_history": [],
        }
    chat_history_raw = row["chat_history"]
    chat_history = json.loads(chat_history_raw) if chat_history_raw else []
    notebook_history_raw = row["notebook_chat_history"]
    notebook_chat_history = (
        json.loads(notebook_history_raw) if notebook_history_raw else []
    )
    draft_content = row["draft_content"] or ""
    return {
        "current_stage": row["current_stage"],
        "chat_history": chat_history,
        "draft_content": draft_content,
        "notebook_chat_history": notebook_chat_history,
    }


def save_project_draft(project_id: int, draft_text: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE projects
            SET draft_content = ?
            WHERE id = ?
            """,
            (draft_text, project_id),
        )
        conn.commit()


def create_project(user_id: str, title: str) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO projects (user_id, title) VALUES (?, ?)",
            (user_id, title.strip()),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_user_projects(user_id: str) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, title, created_at
            FROM projects
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_project_sources(project_id: int) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, title, authors, summary, file_path
            FROM sources
            WHERE project_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (project_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_project_source_count(project_id: int) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM sources WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    return int(row["count"]) if row else 0


def save_source(
    project_id: int,
    filename: str,
    extracted_text: str,
    preview_summary: str,
) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO sources (
                project_id, filename, extracted_text, preview_summary,
                title, authors, summary, file_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                filename,
                extracted_text,
                preview_summary,
                filename,
                "Unknown",
                preview_summary,
                filename,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
