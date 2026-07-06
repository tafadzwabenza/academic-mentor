import json
import uuid
from typing import Any, Dict, List, Optional

import bcrypt
import psycopg2
import psycopg2.extras
import streamlit as st


def _connect():
    return psycopg2.connect(st.secrets["SUPABASE_URL"])


def _cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.DictCursor)


def init_platform_db() -> None:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    student_name TEXT,
                    student_level TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    current_stage TEXT,
                    chat_history TEXT,
                    draft_content TEXT,
                    notebook_chat_history TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id SERIAL PRIMARY KEY,
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
                )
                """
            )
            cur.execute(
                """
                INSERT INTO users (id, name, email)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                ("student_123", "Demo Student", "student@example.com"),
            )
        conn.commit()

    existing = get_user_projects("student_123")
    if not existing:
        create_project("student_123", "Thesis: Boardroom Dynamics")
        create_project("student_123", "Assignment: Drug Impact")


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    return _get_user_cached(user_id)


@st.cache_data(ttl=600)
def _get_user_cached(user_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                SELECT id, name, email, username, student_name, student_level, created_at
                FROM users
                WHERE id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def register_user(username: str, password: str) -> str:
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password are required.")

    user_id = f"user_{uuid.uuid4().hex[:12]}"
    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    )

    with _connect() as conn:
        try:
            with _cursor(conn) as cur:
                cur.execute(
                    """
                    INSERT INTO users (id, name, username, password_hash, email)
                    VALUES (%s, %s, %s, %s, NULL)
                    """,
                    (user_id, username, username, password_hash.decode("utf-8")),
                )
            conn.commit()
        except psycopg2.IntegrityError as error:
            conn.rollback()
            if "unique" in str(error).lower() or "duplicate" in str(error).lower():
                raise ValueError("Username already taken. Please choose another.") from error
            raise

    return user_id


def authenticate_user(username: str, password: str) -> Optional[str]:
    username = username.strip()
    if not username or not password:
        return None

    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                SELECT id, password_hash
                FROM users
                WHERE username = %s
                """,
                (username,),
            )
            row = cur.fetchone()

    if row is None or not row["password_hash"]:
        return None

    stored_hash = row["password_hash"]
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode("utf-8")

    if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        return row["id"]

    return None


def update_user_profile(user_id: str, name: str, level: Optional[str]) -> None:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                UPDATE users
                SET student_name = %s, student_level = %s
                WHERE id = %s
                """,
                (name, level, user_id),
            )
        conn.commit()
    st.cache_data.clear()


def save_project_state(
    project_id: int,
    stage: Optional[str],
    messages_list: List[Dict[str, Any]],
) -> None:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                UPDATE projects
                SET current_stage = %s, chat_history = %s
                WHERE id = %s
                """,
                (stage, json.dumps(messages_list), project_id),
            )
        conn.commit()


def save_notebook_state(
    project_id: int,
    messages_list: List[Dict[str, Any]],
) -> None:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                UPDATE projects
                SET notebook_chat_history = %s
                WHERE id = %s
                """,
                (json.dumps(messages_list), project_id),
            )
        conn.commit()


def get_project_state(project_id: int) -> Dict[str, Any]:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                SELECT current_stage, chat_history, draft_content, notebook_chat_history
                FROM projects
                WHERE id = %s
                """,
                (project_id,),
            )
            row = cur.fetchone()
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
        with _cursor(conn) as cur:
            cur.execute(
                """
                UPDATE projects
                SET draft_content = %s
                WHERE id = %s
                """,
                (draft_text, project_id),
            )
        conn.commit()


def create_project(user_id: str, title: str) -> int:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                INSERT INTO projects (user_id, title)
                VALUES (%s, %s)
                RETURNING id
                """,
                (user_id, title.strip()),
            )
            row = cur.fetchone()
        conn.commit()
        st.cache_data.clear()
        return int(row["id"])


def get_user_projects(user_id: str) -> List[Dict[str, Any]]:
    return _get_user_projects_cached(user_id)


@st.cache_data(ttl=600)
def _get_user_projects_cached(user_id: str) -> List[Dict[str, Any]]:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                SELECT id, user_id, title, created_at
                FROM projects
                WHERE user_id = %s
                ORDER BY created_at DESC, id DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
    return [dict(row) for row in rows]


def update_project_title(project_id: int, new_title: str) -> None:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                UPDATE projects
                SET title = %s
                WHERE id = %s
                """,
                (new_title.strip(), project_id),
            )
        conn.commit()
    st.cache_data.clear()


def delete_project(project_id: int) -> None:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                "DELETE FROM sources WHERE project_id = %s",
                (project_id,),
            )
            cur.execute(
                "DELETE FROM projects WHERE id = %s",
                (project_id,),
            )
        conn.commit()
    st.cache_data.clear()


def get_project_sources(project_id: int) -> List[Dict[str, Any]]:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                SELECT id, title, authors, summary, file_path
                FROM sources
                WHERE project_id = %s
                ORDER BY created_at DESC, id DESC
                """,
                (project_id,),
            )
            rows = cur.fetchall()
    return [dict(row) for row in rows]


def get_project_source_count(project_id: int) -> int:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                "SELECT COUNT(*) AS count FROM sources WHERE project_id = %s",
                (project_id,),
            )
            row = cur.fetchone()
    return int(row["count"]) if row else 0


def save_source(
    project_id: int,
    filename: str,
    extracted_text: str,
    preview_summary: str,
) -> int:
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                INSERT INTO sources (
                    project_id, filename, extracted_text, preview_summary,
                    title, authors, summary, file_path
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
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
            row = cur.fetchone()
        conn.commit()
        return int(row["id"])
