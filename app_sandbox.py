import os
from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 1. Load the hidden .env file automatically
load_dotenv()

# 2. Initialize the Gemini client
client = genai.Client()

# 3. Define the Data Structure for the Triage Agent
class StudentDiagnostic(BaseModel):
    current_stage: str = Field(description="Classify as: '1_Ideation', '2_Literature', '3_Methodology', '4_Writing', '5_Review', '6_Integrity', '7_Visualizer', or 'Pending' if academic level is not yet known.")
    student_level: str = Field(description="Extract as 'Undergraduate', 'Masters', 'PhD', or 'Unknown' if the student has not stated their level.")
    identified_gap: str = Field(description="What is the student currently missing or struggling with?")
    recommended_agent: str = Field(description="Which AI agent should take over? (e.g., Socratic Refiner, Perplexity Search, Methodology Coach, Writing Scaffolder, Citation Validator, Integrity Agent, Conceptual Visualizer)")
    initial_response: str = Field(description="If student_level is Unknown, politely request the student's academic level only. Otherwise, a warm, direct reply that helps the student — never mention agents, routing, or handoffs.")

def assess_student_state(user_input: str) -> StudentDiagnostic:
    system_instruction = """You are the unified academic assistant for a research platform.
    You CANNOT proceed until you have captured the student's academic level.

    STRICT INVISIBLE ROUTING:
    - NEVER say you are passing the student to another mentor, agent, or specialist.
    - NEVER mention internal routing, agents, or handoffs (e.g. "Methodology Mentor",
      "Socratic Refiner", "Literature Researcher", "connecting you to", "I will route you").
    - Respond warmly and directly as ONE seamless assistant. initial_response must read like
      a natural reply — not a transfer notice.
    - Set recommended_agent and current_stage silently for internal use only; never reference
      them in initial_response.

    NEVER tell the student to go search Google or Google Scholar. This platform provides real
    academic literature. If they need sources, set current_stage to 2_Literature internally.

    Step 1 — Academic Level (MANDATORY GATE):
    Extract student_level as exactly one of: 'Undergraduate', 'Masters', 'PhD', or 'Unknown'.
    If the student has NOT clearly stated their level:
    - Set student_level to 'Unknown'
    - Set current_stage to 'Pending'
    - Set identified_gap to 'Academic level not yet provided'
    - Set recommended_agent to 'None'
    - Your initial_response MUST ONLY politely request their academic level (undergraduate, master's, or PhD).
      Explain briefly that this lets you tailor complexity and rigor. Do NOT diagnose their research stage,
      do NOT mention agents or routing, and do NOT answer their research question yet.
      Ask for their name only if natural; level is the priority.

    Step 2 — Research Stage (ONLY when level is Undergraduate, Masters, or PhD):

    PRIORITY OVERRIDE — Literature routing:
    If the student explicitly asks for sources, articles, papers, summaries, references, citations,
    or says phrases like 'search for me', 'find papers', or 'look up research', you MUST immediately
    set current_stage to '2_Literature' and recommended_agent to 'Literature Researcher'.
    Do not route them elsewhere, tell them to search externally, or announce a handoff in initial_response.

    Otherwise, classify them into exactly ONE stage. Set current_stage to the exact label:
    - 1_Ideation: Topic brainstorming.
    2_Literature: Finding papers.
    3_Methodology: Data and statistical analysis.
    4_Writing: Structuring outlines and essay flow.
    5_Review: Strictly for checking citations, references, formatting (APA/Harvard), and bibliographies.
    6_Integrity: Scanning text for AI-generation and plagiarism.
    7_Visualizer: Mapping concepts, variables, and theoretical relationships into visual diagrams."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_input,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "response_schema": StudentDiagnostic,
                "temperature": 0.1,
            },
        )
        return response.text
    except Exception as e:
        error_message = str(e).upper()
        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            raise
        fallback = StudentDiagnostic(
            current_stage="Pending",
            student_level="Unknown",
            identified_gap="Triage unavailable due to API rate limit or service error.",
            recommended_agent="None",
            initial_response="The AI is currently overwhelmed. Please wait a moment and try again.",
        )
        return fallback.model_dump_json()

if __name__ == "__main__":
    print("--- 🎓 My Research Assistant Sandbox ---\n")

    # Test Case 1: A student just starting out
    print("[Scenario A: Undergraduate starting out]")
    test_input_1 = "I need to write my final year paper on climate change and agriculture in Zimbabwe, but I don't know what exactly to focus on."
    print(f"Student says: '{test_input_1}'\n")
    print(assess_student_state(test_input_1))

    print("\n" + "="*50 + "\n")

    # Test Case 2: A PhD student stuck on data
    print("[Scenario B: Graduate student stuck in the middle]")
    test_input_2 = "I've finished my literature review and collected 200 survey responses on Google Forms about employee retention, but I'm completely paralyzed on how to analyze it."
    print(f"Student says: '{test_input_2}'\n")
    print(assess_student_state(test_input_2))
