import os
import uuid

import requests
from dotenv import load_dotenv

from mentor_prompts import build_mentor_instruction

load_dotenv()

COPYLEAKS_EMAIL = os.getenv("COPYLEAKS_EMAIL", "").strip()
COPYLEAKS_API_KEY = os.getenv("COPYLEAKS_API_KEY", "").strip()
LOGIN_URL = "https://id.copyleaks.com/v3/account/login/api"
DETECTOR_URL = "https://api.copyleaks.com/v2/writer-detector/{scan_id}/check"

MIN_WORDS = 50


def get_copyleaks_token() -> str:
    response = requests.post(
        LOGIN_URL,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json={"email": COPYLEAKS_EMAIL, "key": COPYLEAKS_API_KEY},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def check_integrity(user_input: str, level: str = "Unknown") -> str:
    word_count = len(user_input.split())
    if word_count <= MIN_WORDS:
        return (
            f"⚠️ Text too short for integrity scan ({word_count} words). "
            f"Please submit more than {MIN_WORDS} words for accurate AI detection."
        )

    level_note = build_mentor_instruction(level, specialization="Focus on AI-detection and integrity scan results.") + "\n\n"

    try:
        token = get_copyleaks_token()
        scan_id = str(uuid.uuid4())

        response = requests.post(
            DETECTOR_URL.format(scan_id=scan_id),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"text": user_input},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        summary = data.get("summary", {})
        ai_score = summary.get("ai", 0)
        human_score = summary.get("human", 0)

        ai_pct = ai_score * 100 if ai_score <= 1 else ai_score
        human_pct = human_score * 100 if human_score <= 1 else human_score

        total_words = data.get("scannedDocument", {}).get("totalWords", word_count)

        return (
            f"{level_note}"
            f"--- 🔍 Integrity Scan Results ---\n"
            f"Words scanned: {total_words}\n"
            f"Human score: {human_pct:.1f}%\n"
            f"AI score: {ai_pct:.1f}%\n"
            f"\nInterpretation: "
            f"{'Likely AI-generated content detected.' if ai_pct > human_pct else 'Likely human-written content.'}"
        )
    except requests.HTTPError as e:
        return f"⚠️ Copyleaks API error: {e.response.status_code} — {e.response.text}"
    except Exception as e:
        return f"⚠️ Integrity scan failed: {e}"


if __name__ == "__main__":
    print("--- 🛡️ Integrity Agent ---\n")
    sample_text = " ".join(["word"] * 60)
    print(check_integrity(sample_text))
