from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from literature_agent import literature_search
from mentor_prompts import build_mentor_instruction

load_dotenv()

client = genai.Client()

writing_history: list = []


class WritingResponse(BaseModel):
    content: str = Field(description="Direct structural guidance and topic sentences in natural prose; never write the full essay.")


def _history_from_messages(chat_messages: List[dict]) -> list:
    return [
        {
            "role": "user" if message["role"] == "user" else "model",
            "parts": [{"text": message["content"]}],
        }
        for message in chat_messages
    ]


def writing_scaffolder(
    user_input: str,
    chat_messages: Optional[List[dict]] = None,
    level: str = "Unknown",
) -> WritingResponse:
    global writing_history

    lit_context = literature_search(
        f"What are the standard academic essay and paper structures used for writing about: {user_input}",
        search_type="comprehensive",
        level=level,
    )

    system_instruction = build_mentor_instruction(
        level,
        specialization=(
            "Focus on essay structure. Use the provided literature context to justify section "
            "choices with in-text citations (author, year). Provide outlines and topic sentences only — "
            "never write full paragraphs, body text, or polished prose. Read the full conversation "
            "history and respond as if you are continuing an ongoing discussion."
        ),
    )

    combined_prompt = f"""Student input:
{user_input}

Literature context:
{lit_context}"""

    if chat_messages is not None:
        writing_history = _history_from_messages(chat_messages[:-1])
        writing_history.append({"role": "user", "parts": [{"text": combined_prompt}]})
    else:
        writing_history.append({"role": "user", "parts": [{"text": combined_prompt}]})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=writing_history,
        config={
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": WritingResponse,
            "temperature": 0.3,
        },
    )

    result = WritingResponse.model_validate_json(response.text)

    writing_history.append({"role": "model", "parts": [{"text": result.content}]})

    return result


if __name__ == "__main__":
    print("--- ✍️ Writing Agent with Memory ---")

    print("\nStudent: 'I need to write my introduction but don't know where to start.'")
    result1 = writing_scaffolder(
        "I need to write my introduction but don't know where to start.",
        level="Undergraduate",
    )
    print(result1.content)
