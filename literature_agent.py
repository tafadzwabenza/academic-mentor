import hashlib
import os
import sqlite3
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

from mentor_prompts import UNIFIED_ASSISTANT_RULE, build_mentor_instruction

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "").strip()
API_URL = "https://api.perplexity.ai/chat/completions"
DB_PATH = "papers_database.db"


def _cache_key(
    query: str,
    search_type: str,
    year_range: Optional[Tuple[int, int]] = None,
) -> str:
    year_min, year_max = year_range if year_range else ("", "")
    normalized = (
        f"{query.strip().lower()}|{search_type.strip().lower()}|{year_min}|{year_max}"
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers_cache (
                cache_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                search_type TEXT NOT NULL,
                results TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def get_cached_papers(
    query: str,
    search_type: str,
    year_range: Optional[Tuple[int, int]] = None,
) -> Optional[str]:
    _init_db()
    key = _cache_key(query, search_type, year_range)

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT results FROM papers_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()

    if row:
        return row[0]
    return None


def save_papers_to_cache(
    query: str,
    search_type: str,
    results: str,
    year_range: Optional[Tuple[int, int]] = None,
) -> None:
    _init_db()
    key = _cache_key(query, search_type, year_range)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO papers_cache (cache_key, query, search_type, results)
            VALUES (?, ?, ?, ?)
            """,
            (key, query.strip(), search_type.strip(), results),
        )
        conn.commit()


def literature_search(
    query: str,
    search_type: str,
    level: str = "Unknown",
    year_range: Optional[Tuple[int, int]] = None,
) -> str:
    cached = get_cached_papers(query, search_type, year_range)
    if cached:
        return cached

    year_clause = ""
    if year_range:
        year_min, year_max = year_range
        year_clause = (
            f" Restrict results to papers published between {year_min} and {year_max}."
        )

    if search_type == "recent":
        search_instruction = (
            "Find and return many academic papers from the last 5 years relevant to this topic."
            f"{year_clause}"
        )
    elif search_type == "comprehensive":
        search_instruction = (
            "Find and return Seminal Theoretical Papers AND many Recent Empirical studies "
            f"relevant to this topic.{year_clause or ' Include both foundational and recent work.'}"
        )
    else:
        raise ValueError("search_type must be 'recent' or 'comprehensive'")

    mentor_base = build_mentor_instruction(level)
    system_instruction = f"""{mentor_base}

You are an in-app research library acting in NotebookLM style — not a search engine redirect.
Your job is to deliver real academic sources inside this conversation.

{search_instruction}

Response format (required):
1. Open with: "Here are the best sources I found for your topic..."
2. For each paper, provide:
   - Title
   - Authors
   - Summary
   - DOI (if available)
3. End with ONE collaborative question such as: "Which of these summaries stands out to you the most for your paper?"

Never tell the student to search Google, Google Scholar, or any external database. You provide the sources here.

{UNIFIED_ASSISTANT_RULE}"""

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "sonar",
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": query},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    results = data["choices"][0]["message"]["content"]

    save_papers_to_cache(query, search_type, results, year_range)
    return results


if __name__ == "__main__":
    print("--- 📚 Literature Agent (with SQLite cache) ---\n")
    test_query = "employee retention and job satisfaction in remote work"
    print(f"Query: {test_query}\n")
    print(literature_search(test_query, search_type="recent"))
    print("\n--- Second call (should hit cache) ---\n")
    print(literature_search(test_query, search_type="recent"))
