from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from mentor_prompts import build_mentor_instruction

load_dotenv()

client = genai.Client()

visualizer_history: list = []


class VisualizerResponse(BaseModel):
    content: str = Field(description="Brief explanation of the conceptual framework shown in the diagram.")
    mermaid_code: str = Field(description="Valid Mermaid diagram syntax mapping the student's research concepts.")


def _history_from_messages(chat_messages: List[dict]) -> list:
    return [
        {
            "role": "user" if message["role"] == "user" else "model",
            "parts": [{"text": message["content"]}],
        }
        for message in chat_messages
    ]


def conceptual_visualizer(
    user_input: str,
    chat_messages: Optional[List[dict]] = None,
    level: str = "Unknown",
) -> VisualizerResponse:
    global visualizer_history

    system_instruction = build_mentor_instruction(
        level,
        specialization=(
            "Focus on conceptual frameworks. Provide a short explanation in the content field and valid "
            "Mermaid syntax in the mermaid_code field (e.g., flowchart TD or graph LR). Map variables, "
            "relationships, hypotheses, and theoretical links — not essay prose. Read the full conversation "
            "history and respond as if you are continuing an ongoing discussion."
        ),
    )

    if chat_messages is not None:
        visualizer_history = _history_from_messages(chat_messages)
    else:
        visualizer_history.append({"role": "user", "parts": [{"text": user_input}]})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=visualizer_history,
        config={
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": VisualizerResponse,
            "temperature": 0.3,
        },
    )

    result = VisualizerResponse.model_validate_json(response.text)

    visualizer_history.append({"role": "model", "parts": [{"text": result.content}]})

    return result
