from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from mentor_prompts import build_mentor_instruction

load_dotenv()

client = genai.Client()

citation_history: list = []


class CitationResponse(BaseModel):
    content: str = Field(description="In-text citation corrections and a perfectly formatted bibliography in the requested referencing style.")


def _history_from_messages(chat_messages: List[dict]) -> list:
    return [
        {
            "role": "user" if message["role"] == "user" else "model",
            "parts": [{"text": message["content"]}],
        }
        for message in chat_messages
    ]


def citation_validator(
    user_input: str,
    referencing_style: str = "APA",
    chat_messages: Optional[List[dict]] = None,
    level: str = "Unknown",
) -> CitationResponse:
    global citation_history

    system_instruction = build_mentor_instruction(
        level,
        specialization=(
            f"Focus on citation and reference validation in {referencing_style}. Check every citation "
            "and reference with zero tolerance for formatting errors. Provide in-text corrections and "
            "a properly formatted bibliography. Read the full conversation history and respond as if "
            "you are continuing an ongoing discussion."
        ),
    )

    if chat_messages is not None:
        citation_history = _history_from_messages(chat_messages)
    else:
        citation_history.append({"role": "user", "parts": [{"text": user_input}]})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=citation_history,
        config={
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": CitationResponse,
            "temperature": 0.1,
        },
    )

    result = CitationResponse.model_validate_json(response.text)

    citation_history.append({"role": "model", "parts": [{"text": result.content}]})

    return result


# TEST THE MEMORY
if __name__ == "__main__":
    print("--- 📎 Citation Validator with Memory ---")

    # Interaction 1
    print("\nStudent: 'Please check my references section.'")
    result1 = citation_validator(
        "Smith, J. (2020). Climate change impacts. Journal of Science, 12(3), 45-67.",
        referencing_style="APA",
    )
    print(result1.content)

    # Interaction 2
    print("\nStudent: 'Also check this in-text citation: (Smith 2020).'")
    result2 = citation_validator(
        "Also check this in-text citation: (Smith 2020).",
        referencing_style="APA",
    )
    print(result2.content)
