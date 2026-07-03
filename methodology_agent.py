from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from literature_agent import literature_search
from mentor_prompts import build_mentor_instruction

load_dotenv()

client = genai.Client()

methodology_history: list = []


class MethodologyResponse(BaseModel):
    is_resolved: bool = Field(description="True when the student has a clear analysis plan including variables, data types, and an appropriate statistical test.")
    content: str = Field(description="Direct, actionable methodology guidance in natural prose.")


def _history_from_messages(chat_messages: List[dict]) -> list:
    return [
        {
            "role": "user" if message["role"] == "user" else "model",
            "parts": [{"text": message["content"]}],
        }
        for message in chat_messages
    ]


def methodology_mentor(
    user_input: str,
    chat_messages: Optional[List[dict]] = None,
    level: str = "Unknown",
) -> MethodologyResponse:
    global methodology_history

    lit_context = literature_search(
        f"What are the standard research methodologies and statistical approaches "
        f"used in studies related to: {user_input}",
        search_type="comprehensive",
        level=level,
    )

    system_instruction = build_mentor_instruction(
        level,
        specialization=(
            "Focus on data analysis planning. Use the provided literature context to ground "
            "recommendations and include in-text citations (author, year) when referencing sources. "
            "Help the student identify variables, data types, sample size, and appropriate statistical "
            "tests. If the student asks for sources or articles, share relevant summaries and citations "
            "from the literature context below — never tell them to search Google or lecture them on "
            "how to search. If they need a broader literature sweep, note that you can pull more sources "
            "for their topic right here. Set is_resolved to true only when they have a clear, feasible "
            "analysis plan. Read the full conversation history and respond as if you are continuing "
            "an ongoing discussion."
        ),
    )

    combined_prompt = f"""Student input:
{user_input}

Literature context:
{lit_context}"""

    if chat_messages is not None:
        methodology_history = _history_from_messages(chat_messages[:-1])
        methodology_history.append({"role": "user", "parts": [{"text": combined_prompt}]})
    else:
        methodology_history.append({"role": "user", "parts": [{"text": combined_prompt}]})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=methodology_history,
        config={
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": MethodologyResponse,
            "temperature": 0.3,
        },
    )

    result = MethodologyResponse.model_validate_json(response.text)

    methodology_history.append({"role": "model", "parts": [{"text": result.content}]})

    return result


if __name__ == "__main__":
    print("--- 📊 Methodology Agent with Memory ---")

    print("\nStudent: 'I have 200 survey responses but don't know how to analyze them.'")
    result1 = methodology_mentor(
        "I have 200 survey responses but don't know how to analyze them.",
        level="Masters",
    )
    print(result1.content)
