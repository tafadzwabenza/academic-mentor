from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from mentor_prompts import build_mentor_instruction

load_dotenv()

client = genai.Client()

conversation_history: list = []


class RefinerResponse(BaseModel):
    is_refined: bool = Field(description="True when the student's research focus is clear and actionable enough to move forward.")
    content: str = Field(description="Direct, warm mentor guidance in natural prose; never write the essay for the student.")


def _history_from_messages(chat_messages: List[dict]) -> list:
    return [
        {
            "role": "user" if message["role"] == "user" else "model",
            "parts": [{"text": message["content"]}],
        }
        for message in chat_messages
    ]


def socratic_refiner(
    user_input: str,
    chat_messages: Optional[List[dict]] = None,
    level: str = "Unknown",
) -> RefinerResponse:
    global conversation_history

    if chat_messages is not None:
        conversation_history = _history_from_messages(chat_messages)
    else:
        conversation_history.append({"role": "user", "parts": [{"text": user_input}]})

    system_instruction = build_mentor_instruction(
        level,
        specialization=(
            "Focus on ideation and narrowing research topics. Never write their essay, paper, "
            "or full paragraphs for them. Set is_refined to true only when the student has a "
            "focused, feasible research direction. Read the full conversation history and respond "
            "as if you are continuing an ongoing discussion."
        ),
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=conversation_history,
        config={
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": RefinerResponse,
            "temperature": 0.3,
        },
    )

    result = RefinerResponse.model_validate_json(response.text)

    conversation_history.append({"role": "model", "parts": [{"text": result.content}]})

    return result


if __name__ == "__main__":
    print("--- 🧠 Socratic Agent with Memory ---")

    print("\nStudent: 'I need to narrow down my focus.'")
    result1 = socratic_refiner("I need to narrow down my focus.", level="Undergraduate")
    print(result1.content)

    print("\nStudent: 'Can you give me 3 specific angles on that?'")
    result2 = socratic_refiner("Can you give me 3 specific angles on that?", level="Undergraduate")
    print(result2.content)
