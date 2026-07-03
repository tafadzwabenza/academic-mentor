import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

import database_manager as dbm
from app_sandbox import StudentDiagnostic, assess_student_state
from citation_agent import citation_validator
from integrity_agent import check_integrity
from literature_agent import literature_search
from methodology_agent import methodology_mentor
from notebook_agent import process_file_upload
from socratic_agent import socratic_refiner
from visualizer_agent import conceptual_visualizer
from writing_agent import writing_scaffolder

dbm.init_platform_db()

STAGE_LABELS = {
    "1_Ideation": "Ideation",
    "2_Literature": "Literature",
    "3_Methodology": "Methodology",
    "4_Writing": "Writing",
    "5_Review": "Review",
    "6_Integrity": "Integrity",
    "7_Visualizer": "Visualization",
}

STAGE_PROGRESS = {
    "1_Ideation": 20,
    "2_Literature": 40,
    "3_Methodology": 60,
    "4_Writing": 80,
    "5_Review": 100,
    "6_Integrity": 100,
}

TRIAGE_INCOMPLETE_STAGES = {None, "Pending", "Triage", "Unknown", "Error"}

PAST_IDEATION_STAGES = {
    "2_Literature",
    "3_Methodology",
    "4_Writing",
    "5_Review",
    "6_Integrity",
    "7_Visualizer",
}

RESET_COMMANDS = {"reset", "start over"}
VALID_STUDENT_LEVELS = {"Undergraduate", "Masters", "PhD"}
SOURCE_REQUEST_PHRASES = (
    "source",
    "sources",
    "article",
    "articles",
    "paper",
    "papers",
    "summaries",
    "summary",
    "search for me",
    "find papers",
    "look up research",
    "literature review",
    "google scholar",
    "find references",
    "need references",
    "seminal",
)

ONBOARDING_SUGGESTIONS = [
    "I'm an undergraduate student.",
    "I'm a master's student.",
    "I'm a PhD student.",
]

CHAT_ATTACHMENT_TYPES = [
    "pdf",
    "docx",
    "txt",
    "csv",
    "png",
    "jpg",
    "jpeg",
    "mp4",
    "mp3",
    "wav",
]

CURRENT_YEAR = datetime.now().year


DEFAULT_START_YEAR = 1950
DEFAULT_END_YEAR = max(CURRENT_YEAR, 2026)


def persist_user_profile() -> None:
    dbm.update_user_profile(
        st.session_state.user_id,
        st.session_state.get("student_name", "Student"),
        st.session_state.get("student_level"),
    )


def persist_project_state(project_id: Optional[int], project_state: Dict[str, Any]) -> None:
    if project_id is None:
        return
    dbm.save_project_state(
        project_id,
        project_state.get("current_stage"),
        project_state.get("messages", []),
    )


def load_notebook_state_from_db(project_id: int) -> None:
    db_state = dbm.get_project_state(project_id)
    st.session_state.notebook_messages = db_state.get("notebook_chat_history") or []
    st.session_state.selected_notebook_sources = []


def persist_notebook_state(project_id: Optional[int]) -> None:
    if project_id is None:
        return
    dbm.save_notebook_state(project_id, st.session_state.get("notebook_messages", []))


def route_to_notebook_rag(prompt: str, selected_sources: List[Dict[str, Any]]) -> str:
    titles = ", ".join(source.get("title", "Untitled") for source in selected_sources)
    return f"Analyzing selected papers: [{titles}]..."


def load_project_state_from_db(project_id: int, project_state: Dict[str, Any]) -> None:
    db_state = dbm.get_project_state(project_id)
    project_state["messages"] = db_state["chat_history"] or []
    project_state["current_stage"] = db_state["current_stage"]
    project_state["draft_content"] = db_state.get("draft_content") or ""
    st.session_state.current_stage = project_state["current_stage"]
    if project_state["current_stage"] in PAST_IDEATION_STAGES:
        project_state["topic_established"] = True
    load_notebook_state_from_db(project_id)


def stage_is_decided(stage: Optional[str]) -> bool:
    return stage is not None and stage not in TRIAGE_INCOMPLETE_STAGES


def should_run_triage(project_state: Dict[str, Any], explicit_trigger: bool = False) -> bool:
    stage = project_state.get("current_stage")
    st.session_state.current_stage = stage

    if stage_is_decided(stage):
        return False

    if explicit_trigger:
        return True

    messages = project_state.get("messages", [])
    if not messages:
        return True

    user_messages = [message for message in messages if message["role"] == "user"]
    if len(user_messages) == 1 and messages[-1]["role"] == "user":
        return True

    return False


def default_project_state() -> Dict[str, Any]:
    return {
        "messages": [],
        "current_stage": None,
        "topic_established": False,
        "processed_upload_keys": [],
        "search_depth": "Recent",
        "year_range": (DEFAULT_START_YEAR, DEFAULT_END_YEAR),
        "referencing_style": "APA",
        "draft_content": "",
    }


def ensure_project_state(project_id: Optional[int]) -> Dict[str, Any]:
    if project_id is None:
        return default_project_state()
    if "project_states" not in st.session_state:
        st.session_state.project_states = {}
    key = str(project_id)
    if key not in st.session_state.project_states:
        st.session_state.project_states[key] = default_project_state()
    state = st.session_state.project_states[key]
    if "literature_mode" in state and "search_depth" not in state:
        state["search_depth"] = (
            "Comprehensive" if state["literature_mode"] == "Full" else "Recent"
        )
    if "year_range" not in state:
        state["year_range"] = (DEFAULT_START_YEAR, DEFAULT_END_YEAR)
    return state


def is_source_request(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in SOURCE_REQUEST_PHRASES)


def apply_literature_routing_override(text: str, project_state: Dict[str, Any]) -> None:
    if is_onboarded() and is_source_request(text):
        project_state["current_stage"] = "2_Literature"
        project_state["topic_established"] = True
        st.session_state.current_stage = project_state["current_stage"]


def mark_topic_established(project_state: Dict[str, Any], stage: Optional[str]) -> None:
    if stage in PAST_IDEATION_STAGES:
        project_state["topic_established"] = True


def log_error(context: str, error: Exception) -> None:
    pass


def service_error_message() -> str:
    return (
        "I'm having technical difficulties right now. Please try again in a moment.\n\n"
        "**What you can do:**\n"
        "- Re-submit your question in the chat box below\n"
        "- Refresh the page (`Ctrl+R` / `Cmd+R`) if the error persists"
    )


def is_rate_limit_error(error: BaseException) -> bool:
    if getattr(error, "code", None) == 429:
        return True
    message = str(error).upper()
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def log_retry_attempt(retry_state) -> None:
    pass


def messages_to_transcript(messages: List[dict]) -> str:
    lines = []
    for message in messages:
        speaker = "Student" if message["role"] == "user" else "Mentor"
        lines.append(f"{speaker}: {message['content']}")
    return "\n\n".join(lines)


def maybe_update_student_profile(prompt: str) -> None:
    if is_onboarded():
        return
    lower = prompt.lower()
    level_updated = False
    if any(term in lower for term in ("phd", "doctoral", "doctorate")):
        st.session_state.student_level = "PhD"
        level_updated = True
    elif any(term in lower for term in ("master", "masters", "postgraduate")):
        st.session_state.student_level = "Masters"
        level_updated = True
    elif any(term in lower for term in ("undergraduate", "undergrad", "bachelor")):
        st.session_state.student_level = "Undergraduate"
        level_updated = True

    name_updated = False
    if st.session_state.get("student_name", "Student") == "Student":
        if lower.startswith("my name is "):
            name = prompt.split(" ", 3)[-1].strip().rstrip(".")
            if name:
                st.session_state.student_name = name.split()[0].title()
                name_updated = True

    if level_updated or name_updated:
        persist_user_profile()


def is_onboarded() -> bool:
    return st.session_state.get("student_level") in VALID_STUDENT_LEVELS


def get_dynamic_suggestions(stage: Optional[str], message_count: int) -> List[str]:
    if message_count <= 1:
        return [
            "I need help narrowing a topic",
            "I need to find literature",
            "I have my data and need to write",
        ]
    if stage == "1_Ideation":
        return [
            "Help me map the key variables",
            "What is a strong research question?",
            "🔍 I'm ready to find literature",
        ]
    if stage == "2_Literature":
        return [
            "Find recent empirical studies",
            "Find seminal theoretical papers",
            "✍️ Let's structure my paper",
        ]
    if stage == "3_Methodology":
        return [
            "Suggest a research design",
            "How do I collect this data?",
            "✍️ Let's start writing",
        ]
    if stage == "4_Writing":
        return [
            "Create a 4-page outline",
            "Help me with my introduction",
            "🔎 Check my citations",
        ]
    return [
        "What should I do next?",
        "Summarize our progress",
    ]


def apply_stage_from_suggestion(suggestion: str, project_state: Dict[str, Any]) -> None:
    lower = suggestion.lower()
    if "citations" in lower:
        project_state["current_stage"] = "5_Review"
    elif "literature" in lower:
        project_state["current_stage"] = "2_Literature"
        project_state["topic_established"] = True
    elif "structure" in lower or "writing" in lower:
        project_state["current_stage"] = "4_Writing"
    elif "methodology" in lower or "data" in lower:
        project_state["current_stage"] = "3_Methodology"
    elif "narrowing a topic" in lower or "research question" in lower or "variables" in lower:
        project_state["current_stage"] = "1_Ideation"
    st.session_state.current_stage = project_state.get("current_stage")


def get_active_project_title(project_id: Optional[int]) -> str:
    if project_id is None:
        return "No project selected"
    for project in dbm.get_user_projects(st.session_state.user_id):
        if project["id"] == project_id:
            return project["title"]
    return "Unknown project"


@retry(
    retry=retry_if_exception(is_rate_limit_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2),
    before_sleep=log_retry_attempt,
    reraise=True,
)
def assess_student_state_with_retry(prompt: str) -> str:
    return assess_student_state(prompt)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("mermaid"):
            st.markdown(f"```mermaid\n{message['mermaid']}\n```")


def search_depth_to_type(search_depth: str) -> str:
    return "comprehensive" if search_depth == "Comprehensive" else "recent"


def sync_literature_settings(project_state: Dict[str, Any]) -> None:
    st.session_state.literature_search_depth = project_state["search_depth"]
    st.session_state.publication_year_range = project_state["year_range"]


def route_to_agent(
    stage: str,
    messages: List[dict],
    search_depth: str,
    year_range: Tuple[int, int],
    referencing_style: str,
    level: Optional[str] = None,
) -> Tuple[str, Optional[str], bool]:
    """Return (response_text, optional_mermaid_code, success)."""
    mentor_level = level or st.session_state.get("student_level", "Unknown")
    transcript = messages_to_transcript(messages)
    latest_prompt = messages[-1]["content"] if messages else ""

    try:
        if stage == "1_Ideation":
            result = socratic_refiner(latest_prompt, chat_messages=messages, level=mentor_level)
            return result.content, None, True
        if stage == "2_Literature":
            search_type = search_depth_to_type(search_depth)
            return (
                literature_search(
                    transcript,
                    search_type,
                    level=mentor_level,
                    year_range=year_range,
                ),
                None,
                True,
            )
        if stage == "3_Methodology":
            result = methodology_mentor(latest_prompt, chat_messages=messages, level=mentor_level)
            return result.content, None, True
        if stage == "4_Writing":
            result = writing_scaffolder(latest_prompt, chat_messages=messages, level=mentor_level)
            return result.content, None, True
        if stage == "5_Review":
            result = citation_validator(
                transcript,
                referencing_style=referencing_style,
                chat_messages=messages,
                level=mentor_level,
            )
            return result.content, None, True
        if stage == "6_Integrity":
            return check_integrity(transcript, level=mentor_level), None, True
        if stage == "7_Visualizer":
            result = conceptual_visualizer(latest_prompt, chat_messages=messages, level=mentor_level)
            return result.content, result.mermaid_code, True
        if stage == "Error":
            return "I'm a bit overwhelmed right now. Please wait a moment and try again.", None, True
        return (
            "I'm not sure how to help with that yet — could you rephrase your question?",
            None,
            True,
        )
    except Exception as e:
        log_error(f"route_to_agent ({stage})", e)
        return service_error_message(), None, False


def run_triage(prompt: str, project_state: Dict[str, Any]) -> StudentDiagnostic:
    st.sidebar.info("🧠 Triage Agent is analyzing the student's progress...")
    diagnosis_raw = assess_student_state_with_retry(prompt)
    diagnosis = StudentDiagnostic.model_validate_json(diagnosis_raw)
    student_level = diagnosis.student_level.strip()

    if student_level in VALID_STUDENT_LEVELS:
        st.session_state.student_level = student_level
        persist_user_profile()
        stage = diagnosis.current_stage.strip()
        if stage and stage not in {"Pending", "Unknown", "Error"}:
            project_state["current_stage"] = stage
            st.session_state.current_stage = stage
            mark_topic_established(project_state, stage)

    apply_literature_routing_override(prompt, project_state)
    st.session_state.current_stage = project_state.get("current_stage")
    st.session_state.triage_initial_response = diagnosis.initial_response
    st.sidebar.info(
        f"Stage: {diagnosis.current_stage} · Gap: {diagnosis.identified_gap}"
    )
    return diagnosis


def handle_chat_reset(project_id: Optional[int], project_state: Dict[str, Any]) -> None:
    project_state["messages"] = []
    project_state["current_stage"] = None
    project_state["topic_established"] = False
    st.session_state.current_stage = None
    project_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "This project's chat has been reset. "
                "Tell me where you are in your research and we'll pick up from there."
            ),
        }
    ]
    persist_project_state(project_id, project_state)


def generate_assistant_response(
    project_state: Dict[str, Any],
    explicit_triage: bool = False,
) -> None:
    """Run triage/agents and append the assistant reply to project messages (no UI rendering)."""
    messages = project_state["messages"]
    if not messages or messages[-1]["role"] != "user":
        return

    prompt = messages[-1]["content"]
    search_depth = project_state["search_depth"]
    year_range = tuple(project_state["year_range"])
    referencing_style = project_state["referencing_style"]
    sync_literature_settings(project_state)

    maybe_update_student_profile(prompt)
    apply_literature_routing_override(prompt, project_state)
    triage_input = messages_to_transcript(messages)

    try:
        if should_run_triage(project_state, explicit_trigger=explicit_triage):
            diagnosis = run_triage(triage_input, project_state)
            stage = project_state.get("current_stage")

            if not is_onboarded() or not stage_is_decided(stage):
                messages.append({"role": "assistant", "content": diagnosis.initial_response})
                return

            welcome_message = diagnosis.initial_response
            agent_output, mermaid_code, agent_succeeded = route_to_agent(
                stage,
                messages,
                search_depth,
                year_range,
                referencing_style,
                level=st.session_state.student_level,
            )
            if not agent_succeeded:
                messages.append({"role": "assistant", "content": agent_output})
                return

            full_response = f"{welcome_message}\n\n{agent_output}"
            assistant_message: Dict[str, Any] = {"role": "assistant", "content": full_response}
            if mermaid_code:
                assistant_message["mermaid"] = mermaid_code
            messages.append(assistant_message)
            return

        stage = project_state.get("current_stage")
        if stage is None:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Tell me a bit more about your research topic and where you're stuck."
                    ),
                }
            )
            return

        agent_output, mermaid_code, agent_succeeded = route_to_agent(
            stage,
            messages,
            search_depth,
            year_range,
            referencing_style,
            level=st.session_state.student_level,
        )
        assistant_message = {"role": "assistant", "content": agent_output}
        if mermaid_code:
            assistant_message["mermaid"] = mermaid_code
        messages.append(assistant_message)

    except Exception as e:
        log_error("generate_assistant_response", e)
        messages.append(
            {
                "role": "assistant",
                "content": (
                    "I'm having trouble connecting right now. "
                    "The service may be rate-limited or temporarily unavailable. "
                    "Please try again in a moment."
                ),
            }
        )


def apply_suggestion_choice(
    suggestion: str,
    project_id: Optional[int],
    project_state: Dict[str, Any],
) -> None:
    apply_stage_from_suggestion(suggestion, project_state)

    lower = suggestion.lower()
    level_updated = False
    if any(term in lower for term in ("undergraduate", "undergrad", "bachelor")):
        st.session_state.student_level = "Undergraduate"
        level_updated = True
    elif any(term in lower for term in ("master", "masters", "postgraduate")):
        st.session_state.student_level = "Masters"
        level_updated = True
    elif any(term in lower for term in ("phd", "doctoral", "doctorate")):
        st.session_state.student_level = "PhD"
        level_updated = True

    if level_updated:
        persist_user_profile()

    project_state["messages"].append({"role": "user", "content": suggestion})
    apply_literature_routing_override(suggestion, project_state)

    spinner_label = (
        "Getting to know you..."
        if not is_onboarded()
        else "One moment..." if project_state["current_stage"] is None else "Thinking..."
    )
    with st.spinner(spinner_label):
        generate_assistant_response(project_state, explicit_triage=True)

    if project_id is not None:
        dbm.save_project_state(
            project_id,
            project_state.get("current_stage"),
            project_state.get("messages", []),
        )
    st.rerun()


def render_guided_actions(project_state: Dict[str, Any], project_id: Optional[int]) -> None:
    message_count = len(project_state.get("messages", []))
    stage = project_state.get("current_stage")

    if not is_onboarded():
        suggestions = ONBOARDING_SUGGESTIONS
    else:
        suggestions = get_dynamic_suggestions(stage, message_count)

    st.markdown("**Suggested next steps**")
    cols = st.columns(len(suggestions))
    for index, suggestion in enumerate(suggestions):
        with cols[index]:
            if st.button(
                suggestion,
                key=f"suggestion_{project_id or 'none'}_{stage}_{message_count}_{index}",
                use_container_width=True,
            ):
                apply_suggestion_choice(suggestion, project_id, project_state)


def render_progress_checklist(current_stage: Optional[str], project_id: int) -> None:
    if current_stage == "1_Ideation":
        st.checkbox("Define research question", value=False, key=f"progress_{project_id}_ideation_1")
        st.checkbox("Identify target demographic", value=False, key=f"progress_{project_id}_ideation_2")
        st.checkbox("Map key variables", value=False, key=f"progress_{project_id}_ideation_3")
    elif current_stage == "2_Literature":
        st.checkbox("Define research question", value=True, key=f"progress_{project_id}_lit_1")
        st.checkbox("Collect core sources", value=False, key=f"progress_{project_id}_lit_2")
        st.checkbox("Summarize key themes", value=False, key=f"progress_{project_id}_lit_3")
    elif current_stage == "3_Methodology":
        st.checkbox("Collect core sources", value=True, key=f"progress_{project_id}_method_1")
        st.checkbox("Choose research design", value=False, key=f"progress_{project_id}_method_2")
        st.checkbox("Plan data collection", value=False, key=f"progress_{project_id}_method_3")
    elif current_stage == "4_Writing":
        st.checkbox("Define research question", value=True, key=f"progress_{project_id}_write_1")
        st.checkbox("Complete literature review draft", value=False, key=f"progress_{project_id}_write_2")
        st.checkbox("Run Structural Feedback check", value=False, key=f"progress_{project_id}_write_3")
    elif current_stage == "5_Review":
        st.checkbox("Run Structural Feedback check", value=True, key=f"progress_{project_id}_review_1")
        st.checkbox("Format all citations", value=False, key=f"progress_{project_id}_review_2")
        st.checkbox("Final proofread", value=False, key=f"progress_{project_id}_review_3")
    elif current_stage == "6_Integrity":
        st.checkbox("Format all citations", value=True, key=f"progress_{project_id}_integrity_1")
        st.checkbox("Check Plagiarism & AI", value=False, key=f"progress_{project_id}_integrity_2")
        st.checkbox("Submit final draft", value=False, key=f"progress_{project_id}_integrity_3")
    else:
        st.checkbox("Share your academic level in chat", value=False, key=f"progress_{project_id}_default_1")
        st.checkbox("Create or select a project", value=True, key=f"progress_{project_id}_default_2")
        st.checkbox("Describe your research topic", value=False, key=f"progress_{project_id}_default_3")


def render_literature_settings(
    project_id: Optional[int],
    project_state: Dict[str, Any],
) -> None:
    st.subheader("⚙️ Literature Settings")

    depth_options = ["Recent", "Comprehensive"]
    current_depth = project_state.get("search_depth", "Recent")
    if current_depth not in depth_options:
        current_depth = "Recent"
    depth_index = depth_options.index(current_depth)
    project_state["search_depth"] = st.selectbox(
        "Search Depth",
        options=depth_options,
        index=depth_index,
        help="Recent focuses on newer work; Comprehensive includes seminal papers.",
        key=f"search_depth_{project_id}",
    )

    default_range = project_state.get("year_range", (DEFAULT_START_YEAR, DEFAULT_END_YEAR))
    year_start_col, year_end_col = st.columns(2)
    with year_start_col:
        start_year = st.number_input(
            "Start Year",
            min_value=1800,
            max_value=DEFAULT_END_YEAR,
            value=int(default_range[0]),
            step=1,
            key=f"start_year_{project_id}",
        )
    with year_end_col:
        end_year = st.number_input(
            "End Year",
            min_value=1800,
            max_value=DEFAULT_END_YEAR,
            value=int(default_range[1]),
            step=1,
            key=f"end_year_{project_id}",
        )

    if int(start_year) > int(end_year):
        start_year, end_year = end_year, start_year
    project_state["year_range"] = (int(start_year), int(end_year))

    ref_options = ["APA", "Harvard", "MLA", "Chicago"]
    ref_index = ref_options.index(project_state.get("referencing_style", "APA"))
    project_state["referencing_style"] = st.selectbox(
        "Referencing Style",
        options=ref_options,
        index=ref_index,
        key=f"ref_style_{project_id}",
    )

    sync_literature_settings(project_state)


def handle_chat_attachments(
    uploaded_files: Optional[List[Any]],
    project_id: Optional[int],
    project_state: Dict[str, Any],
) -> None:
    if not uploaded_files or project_id is None:
        return

    new_attachments = False
    processed = project_state["processed_upload_keys"]
    messages = project_state["messages"]

    for uploaded_file in uploaded_files:
        upload_key = f"{project_id}:{uploaded_file.name}:{len(uploaded_file.getvalue())}"
        if upload_key in processed:
            continue
        try:
            with st.spinner(f"Attaching {uploaded_file.name}..."):
                process_file_upload(uploaded_file, project_id)
            processed.append(upload_key)
            messages.append(
                {
                    "role": "assistant",
                    "content": f"📎 Attached {uploaded_file.name} to this project.",
                }
            )
            persist_project_state(project_id, project_state)
            new_attachments = True
        except Exception as e:
            log_error("process_file_upload", e)
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Could not attach `{uploaded_file.name}`: {e}",
                }
            )
            persist_project_state(project_id, project_state)
            processed.append(upload_key)
            new_attachments = True

    if new_attachments:
        st.rerun()


# ── Page config & session state ──────────────────────────────────────────────

st.set_page_config(page_title="My Research Assistant", page_icon="🎓", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.logged_in and not st.session_state.user_id:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🎓 My Research Assistant")
    st.markdown("Sign in to save your research projects and pick up where you left off.")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log in", use_container_width=True, key="login_button"):
            if login_username.strip() and login_password:
                db_user_id = dbm.authenticate_user(login_username.strip(), login_password)
                if db_user_id:
                    st.session_state.logged_in = True
                    st.session_state.user_id = db_user_id
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            else:
                st.warning("Please enter your username and password.")

    with signup_tab:
        signup_name = st.text_input("Your Name")
        signup_level = st.selectbox(
            "Current Education Level",
            ["Undergraduate", "Masters", "PhD"],
        )
        signup_username = st.text_input("Choose a username", key="signup_username")
        signup_password = st.text_input("Choose a password", type="password", key="signup_password")
        signup_confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
        if st.button("Create account", use_container_width=True, key="signup_button"):
            if not signup_username.strip() or not signup_password:
                st.warning("Please enter a username and password.")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    db_user_id = dbm.register_user(signup_username.strip(), signup_password)
                    dbm.update_user_profile(db_user_id, signup_name, signup_level)
                    st.session_state.logged_in = True
                    st.session_state.user_id = db_user_id
                    st.session_state.student_name = signup_name
                    st.session_state.student_level = signup_level
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))

if not st.session_state.logged_in:
    st.stop()

user_record = dbm.get_user(st.session_state.user_id)
if user_record:
    st.session_state.student_name = user_record.get("student_name") or "Student"
    st.session_state.student_level = user_record.get("student_level")
else:
    st.session_state.student_name = "Student"
    st.session_state.student_level = None
    dbm.update_user_profile(st.session_state.user_id, "Student", None)

if "triage_initial_response" not in st.session_state:
    st.session_state.triage_initial_response = None
if "project_states" not in st.session_state:
    st.session_state.project_states = {}
if "last_loaded_project_id" not in st.session_state:
    st.session_state.last_loaded_project_id = None
if "notebook_messages" not in st.session_state:
    st.session_state.notebook_messages = []
if "selected_notebook_sources" not in st.session_state:
    st.session_state.selected_notebook_sources = []

user_projects = dbm.get_user_projects(st.session_state.user_id)
if "active_project_id" not in st.session_state and user_projects:
    st.session_state.active_project_id = user_projects[0]["id"]

active_project_id = st.session_state.get("active_project_id")
project_state = ensure_project_state(active_project_id)
if (
    active_project_id is not None
    and st.session_state.last_loaded_project_id != active_project_id
):
    load_project_state_from_db(active_project_id, project_state)
    st.session_state.last_loaded_project_id = active_project_id
active_project_title = get_active_project_title(active_project_id)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 👤 My Profile")
    st.write(f"**Name:** {st.session_state.get('student_name', 'Student')}")
    st.write(f"**Level:** {st.session_state.get('student_level', 'Not set')}")
    st.divider()

    st.header("📁 Projects")

    if user_projects:
        project_titles = {project["id"]: project["title"] for project in user_projects}
        project_ids = [project["id"] for project in user_projects]
        current_id = st.session_state.get("active_project_id")
        if current_id not in project_ids:
            current_id = project_ids[0]

        selected_project_id = st.selectbox(
            "Active project",
            options=project_ids,
            index=project_ids.index(current_id),
            format_func=lambda project_id: project_titles[project_id],
            label_visibility="collapsed",
        )
        if selected_project_id != st.session_state.get("active_project_id"):
            st.session_state.active_project_id = selected_project_id
            loaded_state = ensure_project_state(selected_project_id)
            load_project_state_from_db(selected_project_id, loaded_state)
            st.session_state.last_loaded_project_id = selected_project_id
        else:
            st.session_state.active_project_id = selected_project_id
    else:
        st.info("Create your first project below.")

    with st.expander("➕ New project"):
        new_project_title = st.text_input("Project title", placeholder="e.g. Thesis: Climate Policy")
        if st.button("Create project", use_container_width=True):
            title = new_project_title.strip()
            if title:
                new_id = dbm.create_project(st.session_state.user_id, title)
                st.session_state.active_project_id = new_id
                ensure_project_state(new_id)
                st.rerun()
            else:
                st.warning("Enter a project title first.")

    st.divider()
    sidebar_project_id = st.session_state.get("active_project_id")
    sidebar_project_state = ensure_project_state(sidebar_project_id)
    render_literature_settings(sidebar_project_id, sidebar_project_state)

    if not is_onboarded():
        st.divider()
        st.info("Share your academic level once in chat — it applies across all projects.")

    st.caption("Type `reset` or `start over` in chat to clear this project's conversation.")

# Refresh project-scoped state after sidebar may have changed selection
active_project_id = st.session_state.get("active_project_id")
project_state = ensure_project_state(active_project_id)
if (
    active_project_id is not None
    and st.session_state.last_loaded_project_id != active_project_id
):
    load_project_state_from_db(active_project_id, project_state)
    st.session_state.last_loaded_project_id = active_project_id
active_project_title = get_active_project_title(active_project_id)

st.title("🎓 My Research Assistant")

# ── Main dashboard ───────────────────────────────────────────────────────────

tab_chat, tab_notebook, tab_writing, tab_progress = st.tabs([
    "💬 Chat",
    "📚 My Notebook",
    "✍️ Writing Board",
    "📊 Progress Dashboard",
])

with tab_chat:
    # Block A — chat history first
    for message in project_state["messages"]:
        render_message(message)

    # Block B — suggested next steps
    render_guided_actions(project_state, active_project_id)

    # Block C — attachments
    with st.popover("📎 Attach Files"):
        uploaded_files = st.file_uploader(
            "Attach files",
            type=CHAT_ATTACHMENT_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"chat_attachments_{active_project_id or 'none'}",
        )
        handle_chat_attachments(uploaded_files, active_project_id, project_state)

    # Block D — chat input (full width, always last)
    prompt = st.chat_input("Describe where you are in your research...")

    if prompt:
        if prompt.strip().lower() in RESET_COMMANDS:
            handle_chat_reset(active_project_id, project_state)
            st.rerun()

        project_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        maybe_update_student_profile(prompt)
        apply_stage_from_suggestion(prompt, project_state)
        apply_literature_routing_override(prompt, project_state)

        spinner_label = (
            "Getting to know you..."
            if not is_onboarded()
            else "One moment..." if project_state["current_stage"] is None else "Thinking..."
        )
        with st.spinner(spinner_label):
            generate_assistant_response(project_state, explicit_triage=False)

        if active_project_id is not None:
            dbm.save_project_state(
                active_project_id,
                project_state.get("current_stage"),
                project_state.get("messages", []),
            )
        st.rerun()

with tab_notebook:
    st.title("📚 My Notebook")
    st.subheader("View your uploaded PDFs and fetched literature here.")

    project_id = active_project_id
    if not project_id:
        st.info("Select or create a project to view your notebook.")
    else:
        source_col, chat_col = st.columns([1, 1])

        with source_col:
            st.markdown("### 📄 Project Sources")
            sources = dbm.get_project_sources(project_id)
            sources_by_id: Dict[int, Dict[str, Any]] = {}

            if not sources:
                st.info(
                    "No sources attached to this project yet. Upload a PDF in the Chat tab "
                    "or ask the Literature Agent to find papers."
                )
                st.session_state.selected_notebook_sources = []
            else:
                sources_by_id = {source["id"]: source for source in sources}
                selected_ids: List[int] = []

                for source in sources:
                    source_id = source["id"]
                    if st.checkbox(
                        source["title"],
                        key=f"notebook_source_{project_id}_{source_id}",
                    ):
                        selected_ids.append(source_id)

                    with st.expander(f"Details: {source['title']}"):
                        st.write(f"**Title:** {source['title']}")
                        st.write(f"**Authors:** {source['authors']}")
                        st.write(f"**Summary / Extract:** {source['summary']}")

                st.session_state.selected_notebook_sources = selected_ids

        with chat_col:
            st.markdown("### 💬 Source Q&A")

            if not st.session_state.selected_notebook_sources:
                _, center_col, _ = st.columns([1, 2, 1])
                with center_col:
                    st.info("Select sources on the left to begin.")
            else:
                selected_sources = [
                    sources_by_id[source_id]
                    for source_id in st.session_state.selected_notebook_sources
                    if source_id in sources_by_id
                ]
                selected_titles = ", ".join(
                    source.get("title", "Untitled") for source in selected_sources
                )
                st.caption(f"Reading: {selected_titles}")

                for message in st.session_state.notebook_messages:
                    render_message(message)

                notebook_prompt = st.chat_input(
                    "Ask a question about these sources...",
                    key="notebook_chat_input",
                )

                if notebook_prompt:
                    st.session_state.notebook_messages.append(
                        {"role": "user", "content": notebook_prompt}
                    )
                    with st.chat_message("user"):
                        st.markdown(notebook_prompt)

                    with st.spinner("Analyzing..."):
                        notebook_response = route_to_notebook_rag(
                            notebook_prompt,
                            selected_sources,
                        )

                    st.session_state.notebook_messages.append(
                        {"role": "assistant", "content": notebook_response}
                    )
                    dbm.save_notebook_state(
                        project_id,
                        st.session_state.notebook_messages,
                    )
                    st.rerun()

with tab_writing:
    editor_col, tools_col = st.columns([2, 1])

    with editor_col:
        st.markdown("### ✍️ Your Draft")
        if active_project_id is None:
            st.info("Select or create a project to start writing.")
        else:
            draft_text = st.text_area(
                "Draft",
                value=project_state.get("draft_content", ""),
                height=600,
                placeholder="Start writing your essay, thesis chapter, or assignment here...",
                label_visibility="collapsed",
                key=f"draft_editor_{active_project_id}",
            )

            if st.button("💾 Save Draft", key=f"save_draft_{active_project_id}"):
                dbm.save_project_draft(active_project_id, draft_text)
                project_state["draft_content"] = draft_text
                st.success("Draft saved!")

    with tools_col:
        st.markdown("### 🧰 AI Assistant Tools")

        if active_project_id is None:
            st.caption("Tools become available once a project is selected.")
        else:
            draft_for_tools = st.session_state.get(
                f"draft_editor_{active_project_id}",
                project_state.get("draft_content", ""),
            )
            student_level = st.session_state.get("student_level", "Unknown")
            referencing_style = project_state.get("referencing_style", "APA")

            structural_clicked = st.button(
                "📝 Structural Feedback",
                key=f"structural_feedback_{active_project_id}",
                use_container_width=True,
            )
            citations_clicked = st.button(
                "🔎 Format Citations",
                key=f"format_citations_{active_project_id}",
                use_container_width=True,
            )
            plagiarism_clicked = st.button(
                "🛡️ Check Plagiarism & AI",
                key=f"check_plagiarism_{active_project_id}",
                use_container_width=True,
            )

            if structural_clicked or citations_clicked or plagiarism_clicked:
                if not draft_for_tools.strip():
                    st.warning("Add some text to your draft before running AI tools.")
                else:
                    with st.spinner("Analyzing..."):
                        if structural_clicked:
                            result = writing_scaffolder(
                                draft_for_tools,
                                level=student_level or "Unknown",
                            )
                            st.info(result.content)
                        elif citations_clicked:
                            result = citation_validator(
                                draft_for_tools,
                                referencing_style=referencing_style,
                                level=student_level or "Unknown",
                            )
                            st.markdown(result.content)
                        elif plagiarism_clicked:
                            result = check_integrity(
                                draft_for_tools,
                                level=student_level or "Unknown",
                            )
                            st.markdown(result)

with tab_progress:
    project_id = active_project_id

    if project_id:
        st.session_state.current_stage = project_state.get("current_stage")
        st.session_state.draft_content = project_state.get("draft_content", "")

        current_stage = st.session_state.current_stage
        progress_percentage = STAGE_PROGRESS.get(current_stage, 10)

        if current_stage and "_" in current_stage:
            clean_stage_name = current_stage.split("_", 1)[-1].upper()
        else:
            clean_stage_name = "GETTING STARTED"

        st.markdown(f"### 📍 Current Phase: **{clean_stage_name}**")
        st.progress(
            progress_percentage / 100,
            text=f"{progress_percentage}% Complete",
        )

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            word_count = len(st.session_state.get("draft_content", "").split())
            st.metric(label="Draft Word Count", value=word_count)
        with m_col2:
            source_count = dbm.get_project_source_count(project_id)
            st.metric(label="Sources Collected", value=source_count)
        with m_col3:
            st.date_input("Target Deadline", key=f"target_deadline_{project_id}")

        st.divider()
        st.markdown("### 📋 Next Action Steps")
        render_progress_checklist(current_stage, project_id)
    else:
        st.info("Select or create a project to view your progress dashboard.")
