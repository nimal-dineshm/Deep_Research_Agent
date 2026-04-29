"""Streamlit UI for the Deep Research Agent.

Pipeline: Clarification → Brief → Supervisor → Research → Report

Each stage is shown as a collapsible dropdown (st.status) that streams
live output during execution and collapses when complete.
A second tab shows the full LangGraph node/edge diagram via Mermaid.js.
"""

import asyncio
import queue
import re
import threading
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env BEFORE importing any LangChain / provider packages so that
# api-key env vars (ANTHROPIC_API_KEY, TAVILY_API_KEY, etc.) are visible.
load_dotenv()

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from deep_research.agents.full_agent import agent

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Stage Definitions ────────────────────────────────────────────────────────

STAGE_CONFIG: Dict[str, Dict[str, str]] = {
    "clarify_with_user": {
        "label": "Clarification Agent",
        "icon": "🤔",
        "desc": "Analysing your request and checking for missing details",
        "stream_prefix": "🤔 **Clarifier:**",
    },
    "write_research_brief": {
        "label": "Brief Agent",
        "icon": "📝",
        "desc": "Translating your request into a detailed research brief",
        "stream_prefix": "📝 **Brief Writer:**",
    },
    "supervisor_subgraph": {
        "label": "Supervisor Agent",
        "icon": "🧭",
        "desc": "Planning and delegating research tasks to sub-agents",
        "stream_prefix": "🧭 **Supervisor:**",
    },
    "final_report_generation": {
        "label": "Report Agent",
        "icon": "✍️",
        "desc": "Synthesising all findings into a final report",
        "stream_prefix": "✍️ **Report Writer:**",
    },
}

# Sub-graph node → display metadata (shown as writes inside supervisor stage)
SUB_STAGE_CONFIG: Dict[str, Dict[str, str]] = {
    "supervisor": {"label": "Supervisor Planning", "icon": "🧭"},
    "supervisor_tools": {"label": "Tool Execution", "icon": "🔧"},
    "llm_call": {"label": "Researcher", "icon": "🔬"},
    "tool_node": {"label": "Research Tools", "icon": "⚡"},
    "compress_research": {"label": "Compression", "icon": "🗜️"},
}

# ─── Session State Init ───────────────────────────────────────────────────────

_DEFAULTS = {
    "chat_history": [],
    "lc_messages": [],
    "awaiting_clarification": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── Provider / Model Catalogue ──────────────────────────────────────────────

PROVIDERS: Dict[str, Any] = {
    "Anthropic": {
        "icon": "🎭",
        "models": [
            "anthropic:claude-opus-4-7",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-haiku-4-5-20251001",
        ],
        "defaults": {
            "research_model": 1,
            "final_report_model": 1,
            "compression_model": 2,
            "summarization_model": 2,
        },
    },
    "Gemini": {
        "icon": "✨",
        "models": [
            "google_genai:gemini-2.5-pro",
            "google_genai:gemini-2.5-flash",
            "google_genai:gemini-2.0-flash",
            "google_genai:gemini-1.5-pro",
            "google_genai:gemini-1.5-flash",
        ],
        "defaults": {
            "research_model": 1,
            "final_report_model": 0,
            "compression_model": 1,
            "summarization_model": 2,
        },
    },
    "OpenAI": {
        "icon": "🤖",
        "models": [
            "openai:gpt-4.1",
            "openai:gpt-4.1-mini",
            "openai:gpt-4o",
            "openai:gpt-4o-mini",
            "openai:o4-mini",
        ],
        "defaults": {
            "research_model": 0,
            "final_report_model": 0,
            "compression_model": 0,
            "summarization_model": 1,
        },
    },
    "Groq": {
        "icon": "⚡",
        "models": [
            "groq:llama-3.3-70b-versatile",
            "groq:deepseek-r1-distill-llama-70b",
            "groq:llama-3.1-8b-instant",
            "groq:mixtral-8x7b-32768",
            "groq:gemma2-9b-it",
        ],
        "defaults": {
            "research_model": 0,
            "final_report_model": 0,
            "compression_model": 0,
            "summarization_model": 2,
        },
    },
}

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    with st.expander("🤖 Models", expanded=True):
        provider_names = list(PROVIDERS.keys())
        provider = st.selectbox(
            "Provider",
            options=provider_names,
            index=0,
            format_func=lambda p: f"{PROVIDERS[p]['icon']} {p}",
            key="sel_provider",
        )
        prov = PROVIDERS[provider]
        models = prov["models"]
        defs = prov["defaults"]

        research_model = st.selectbox(
            "Research Agent",
            options=models,
            index=defs["research_model"],
            help="Used for clarification, brief writing, supervisor logic, and researchers.",
            key=f"sel_research_{provider}",
        )
        final_report_model = st.selectbox(
            "Final Report Agent",
            options=models,
            index=defs["final_report_model"],
            help="Used to write the final synthesis report.",
            key=f"sel_final_{provider}",
        )
        compression_model = st.selectbox(
            "Compression Agent",
            options=models,
            index=defs["compression_model"],
            help="Used to compress each researcher's findings.",
            key=f"sel_compress_{provider}",
        )
        summarization_model = st.selectbox(
            "Summarization Agent",
            options=models,
            index=defs["summarization_model"],
            help="Used to summarize individual Tavily search results.",
            key=f"sel_summ_{provider}",
        )

    with st.expander("🔍 Search & Research", expanded=True):
        allow_clarification = st.checkbox(
            "Allow Clarification Questions",
            value=True,
            help="When enabled, the agent may ask a clarifying question before researching.",
        )
        max_concurrent = st.slider(
            "Max Concurrent Researchers", min_value=1, max_value=10, value=3,
            help="Number of researchers the supervisor can run in parallel per iteration."
        )
        max_supervisor_iter = st.slider(
            "Supervisor Max Iterations", min_value=1, max_value=10, value=6,
            help="How many rounds of research the supervisor can run."
        )
        max_tool_calls = st.slider(
            "Max Tool Calls / Researcher", min_value=1, max_value=20, value=5,
            help="Max search/tool calls each individual researcher can make."
        )

    st.divider()
    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.lc_messages = []
        st.session_state.awaiting_clarification = False
        st.rerun()

run_config: Dict[str, Any] = {
    "configurable": {
        "research_model": research_model,
        "final_report_model": final_report_model,
        "compression_model": compression_model,
        "summarization_model": summarization_model,
        "allow_clarification": allow_clarification,
        "max_concurrent_research_units": max_concurrent,
        "max_researcher_iterations": max_supervisor_iter,
        "max_react_tool_calls": max_tool_calls,
    }
}

# ─── Async / Research Helpers ─────────────────────────────────────────────────


def _graph_runner(
    lc_messages: list,
    cfg: dict,
    ev_queue: queue.Queue,
) -> None:
    """Execute the deep research LangGraph in a daemon thread via asyncio."""

    async def _async_run() -> None:
        try:
            async for event in agent.astream_events(
                {"messages": lc_messages},
                config=cfg,
                version="v2",
            ):
                ev_queue.put({"type": "event", "data": event})
        except Exception as exc:
            ev_queue.put({"type": "error", "data": str(exc)})
        finally:
            ev_queue.put({"type": "done"})

    asyncio.run(_async_run())


def _extract_text(content: Any) -> str:
    """Normalise various LangChain content formats to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


_SOURCES_RE = re.compile(
    r'(?:^|\n)(?:#{1,3}\s*|\*{1,2})sources\b\*{0,2}',
    re.IGNORECASE,
)


def process_citations(text: str) -> str:
    """Convert bare [N] citation markers into clickable HTML links."""
    sources_match = _SOURCES_RE.search(text)
    if not sources_match:
        return text

    split_pos = sources_match.start() if sources_match.start() == 0 else sources_match.start() + 1
    body = text[:split_pos]
    sources_section = text[split_pos:]

    url_map: Dict[str, str] = {}

    for m in re.finditer(
        r'^\[(\d+)\][^\n]*(https?://[^\s\)\]\n]+)',
        sources_section,
        re.MULTILINE,
    ):
        num = m.group(1)
        if num not in url_map:
            url_map[num] = m.group(2).rstrip(".,;)")

    for m in re.finditer(
        r'^\[(\d+)\][^\n]*\(https?://([^\s\)\n]+)\)',
        sources_section,
        re.MULTILINE,
    ):
        num = m.group(1)
        if num not in url_map:
            full_url = "https://" + m.group(2).rstrip(".,;)")
            url_map[num] = full_url

    if not url_map:
        return text

    def _link(m: re.Match) -> str:
        num = m.group(1)
        url = url_map.get(num)
        if url:
            return (
                f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#1a73e8;text-decoration:none;font-size:0.78em;'
                f'vertical-align:super;font-weight:700;">[{num}]</a>'
            )
        return m.group(0)

    processed_body = re.sub(r'\[(\d+)\](?!\()', _link, body)
    return processed_body + sources_section


def render_content(text: str) -> None:
    """Render markdown, converting [N] citations to clickable links when Sources exist."""
    if _SOURCES_RE.search(text):
        st.markdown(process_citations(text), unsafe_allow_html=True)
    else:
        st.markdown(text)


def run_research(user_input: str) -> None:
    """Process a user message through the deep research pipeline."""
    ev_queue: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_graph_runner,
        args=(list(st.session_state.lc_messages), run_config, ev_queue),
        daemon=True,
    )
    thread.start()

    # Immediate visual feedback while the graph initialises (before first event arrives)
    init_placeholder = st.empty()
    init_placeholder.status("⏳ Starting research pipeline...", state="running", expanded=False)

    stage_containers: Dict[str, Dict] = {}
    current_stage: Optional[str] = None
    researcher_count: int = 0
    final_state: Dict = {}

    while True:
        try:
            item = ev_queue.get(timeout=300)
        except queue.Empty:
            st.error("⏱️ Research timed out after 5 minutes. Please try again.")
            break

        if item["type"] == "done":
            break
        if item["type"] == "error":
            st.error(f"❌ Pipeline error: {item['data']}")
            break

        event: Dict = item["data"]
        ev_type: str = event.get("event", "")
        ev_name: str = event.get("name", "")
        metadata: dict = event.get("metadata", {})
        lg_node: str = metadata.get("langgraph_node", "")

        # ── Top-level stage: START ────────────────────────────────────────────
        if ev_type == "on_chain_start" and ev_name in STAGE_CONFIG:
            if ev_name not in stage_containers:
                init_placeholder.empty()  # clear the "Starting..." placeholder
                info = STAGE_CONFIG[ev_name]
                status = st.status(f"{info['icon']} {info['label']}", expanded=True)
                status.caption(info["desc"])
                stage_containers[ev_name] = {
                    "status": status,
                    "text": "",
                    "text_ph": None,
                    "prefix": f"{info['stream_prefix']}\n\n",
                    "prefix_written": False,
                    "accumulated_text": [],
                }
                current_stage = ev_name

        # ── Sub-stage: START (inside supervisor / researcher subgraphs) ───────
        elif ev_type == "on_chain_start" and ev_name in SUB_STAGE_CONFIG:
            parent = "supervisor_subgraph"
            if parent in stage_containers:
                info = SUB_STAGE_CONFIG[ev_name]
                if ev_name == "llm_call":
                    researcher_count += 1
                    stage_containers[parent]["status"].write(
                        f"**{info['icon']} Researcher #{researcher_count}** started"
                    )
                    stage_containers[parent]["text"] = ""
                    stage_containers[parent]["text_ph"] = None
                    stage_containers[parent]["prefix"] = (
                        f"🔬 **Researcher #{researcher_count}:**\n\n"
                    )
                    stage_containers[parent]["prefix_written"] = False
                else:
                    stage_containers[parent]["status"].write(
                        f"**{info['icon']} {info['label']}**"
                    )

        # ── Tool call: show search queries ────────────────────────────────────
        elif ev_type == "on_tool_start":
            parent = "supervisor_subgraph"
            if parent in stage_containers:
                tool_input = event.get("data", {}).get("input", {})
                if isinstance(tool_input, dict):
                    queries: List[str] = tool_input.get("queries", [])
                    for q in queries[:3]:
                        stage_containers[parent]["status"].write(f"🔎 `{q[:120]}`")

        # ── Model streaming: accumulate text into the active stage ────────────
        elif ev_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if not chunk:
                continue
            text = _extract_text(getattr(chunk, "content", ""))
            if not text:
                continue

            target: Optional[str] = None
            if lg_node in STAGE_CONFIG:
                target = lg_node
            elif lg_node in SUB_STAGE_CONFIG:
                target = "supervisor_subgraph"
            elif current_stage:
                target = current_stage

            if target and target in stage_containers:
                sc = stage_containers[target]
                if not sc["prefix_written"]:
                    sc["text"] = sc["prefix"]
                    sc["accumulated_text"].append(sc["prefix"])
                    sc["prefix_written"] = True
                sc["text"] += text
                sc["accumulated_text"].append(text)
                if sc["text_ph"] is None:
                    if target == "final_report_generation":
                        scroll_box = sc["status"].container(height=600, border=False)
                        sc["text_ph"] = scroll_box.empty()
                    else:
                        sc["text_ph"] = sc["status"].empty()
                sc["text_ph"].markdown(sc["text"])

        # ── Supervisor tool-calls: capture ConductResearch / think_tool decisions ─
        elif ev_type == "on_chat_model_end" and lg_node == "supervisor":
            if "supervisor_subgraph" not in stage_containers:
                continue
            sc = stage_containers["supervisor_subgraph"]
            output = event.get("data", {}).get("output")
            tool_calls = getattr(output, "tool_calls", []) if output else []
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                if name == "ConductResearch":
                    topic = args.get("research_topic", "")
                    preview = topic[:400] + ("..." if len(topic) > 400 else "")
                    entry = f"🔬 **Delegating research:** {preview}\n\n"
                elif name == "think_tool":
                    reflection = args.get("reflection", "")
                    preview = reflection[:300] + ("..." if len(reflection) > 300 else "")
                    entry = f"💭 **Planning:** {preview}\n\n"
                elif name == "ResearchComplete":
                    entry = "✅ **Research complete — moving to report generation**\n\n"
                else:
                    continue
                sc["status"].markdown(entry)
                sc["accumulated_text"].append(entry)

        # ── Top-level stage: END ──────────────────────────────────────────────
        elif ev_type == "on_chain_end" and ev_name in STAGE_CONFIG:
            if ev_name not in stage_containers:
                continue
            output = event.get("data", {}).get("output", {})
            info = STAGE_CONFIG[ev_name]
            sc = stage_containers[ev_name]

            if ev_name == "clarify_with_user":
                msgs = output.get("messages", []) if isinstance(output, dict) else []
                if msgs:
                    content = _extract_text(getattr(msgs[-1], "content", ""))
                    if content:
                        display = f"{info['stream_prefix']}\n\n{content}"
                        sc["status"].markdown(display)
                        sc["accumulated_text"].append(display)

            elif ev_name == "write_research_brief" and isinstance(output, dict):
                brief = output.get("research_brief", "")
                if brief:
                    display = f"{info['stream_prefix']}\n\n{brief}"
                    sc["status"].markdown(display)
                    sc["accumulated_text"].append(display)

            elif ev_name == "supervisor_subgraph" and isinstance(output, dict):
                notes: List[str] = output.get("notes", [])
                if notes:
                    sc["status"].write(f"Collected {len(notes)} research note(s).")

            elif ev_name == "final_report_generation" and isinstance(output, dict):
                final_state = output

            keep_expanded = ev_name == "final_report_generation"
            sc["status"].update(
                label=f"✅ {info['icon']} {info['label']}",
                state="complete",
                expanded=keep_expanded,
            )

        # ── Top-level LangGraph end → capture full output state ───────────────
        elif ev_type == "on_chain_end" and ev_name == "LangGraph":
            output = event.get("data", {}).get("output", {})
            if isinstance(output, dict) and not final_state:
                final_state = output

    thread.join(timeout=10)

    # Collect per-stage accumulated text for persistent display
    stages = []
    for stage_name in STAGE_CONFIG:
        if stage_name not in stage_containers:
            continue
        sc = stage_containers[stage_name]
        full_text = "".join(sc.get("accumulated_text", []))
        if full_text:
            info = STAGE_CONFIG[stage_name]
            stages.append({"label": info["label"], "icon": info["icon"], "text": full_text})

    # Determine and render assistant response
    assistant_msg: str = ""

    report = final_state.get("final_report", "")
    if report:
        assistant_msg = report
        st.session_state.awaiting_clarification = False

    if not assistant_msg:
        msgs = final_state.get("messages", [])
        if msgs:
            last = msgs[-1]
            content = _extract_text(getattr(last, "content", ""))
            if content:
                assistant_msg = content
                st.session_state.awaiting_clarification = not bool(
                    final_state.get("final_report")
                )

    if assistant_msg:
        is_final_report = bool(final_state.get("final_report"))
        if not is_final_report:
            with st.chat_message("assistant"):
                render_content(assistant_msg)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_msg, "stages": stages}
        )
        st.session_state.lc_messages.append(AIMessage(content=assistant_msg))


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🔬 Deep Research Agent")
st.caption(
    "Multi-agent pipeline: **Clarification** → **Brief** → "
    "**Supervisor** → **Research** → **Report**"
)

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_research, tab_graph = st.tabs(["💬 Research", "🗺️ Pipeline Graph"])

# ─── Pipeline Graph Tab ───────────────────────────────────────────────────────

with tab_graph:
    st.subheader("Pipeline Graph")
    st.caption(
        "Live view of every node and edge in the compiled LangGraph. "
        "Toggle **Expand subgraphs** to drill into the supervisor and researcher subgraphs."
    )

    xray = st.toggle("Expand subgraphs (xray)", value=True)

    try:
        mermaid_src: str = agent.get_graph(xray=xray).draw_mermaid()

        mermaid_html = f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    body  {{ background: transparent; margin: 0; padding: 8px; }}
    .mermaid {{ font-family: sans-serif; }}
  </style>
</head>
<body>
  <div class="mermaid">
{mermaid_src}
  </div>
  <script>
    mermaid.initialize({{
      startOnLoad: true,
      theme: "default",
      flowchart: {{ curve: "basis", useMaxWidth: true }},
    }});
  </script>
</body>
</html>"""

        st.components.v1.html(mermaid_html, height=620, scrolling=True)

        with st.expander("📄 Raw Mermaid source"):
            st.code(mermaid_src, language="text")

    except Exception as exc:
        st.error(f"Could not render graph: {exc}")

# ─── Research Tab ─────────────────────────────────────────────────────────────

with tab_research:
    if st.session_state.awaiting_clarification:
        st.info(
            "The agent asked a clarifying question. "
            "Type your answer in the chat bar at the bottom of the page.",
            icon="💬",
        )

    for idx, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "assistant" and msg.get("stages"):
            for stage in msg["stages"]:
                with st.expander(f"{stage['icon']} {stage['label']}", expanded=False):
                    st.markdown(stage["text"])
            with st.chat_message("assistant"):
                with st.container(height=600, border=False):
                    render_content(msg["content"])
            is_report = len(msg["content"]) > 300 or "## " in msg["content"]
            if is_report:
                st.download_button(
                    label="⬇️ Download Report",
                    data=msg["content"],
                    file_name="research_report.md",
                    mime="text/markdown",
                    key=f"dl_{idx}",
                    use_container_width=False,
                )
        else:
            with st.chat_message(msg["role"]):
                render_content(msg["content"])

# ─── Chat Input ───────────────────────────────────────────────────────────────

input_hint = (
    "Answer the clarifying question..."
    if st.session_state.awaiting_clarification
    else "What would you like me to research?"
)
if prompt := st.chat_input(input_hint):
    st.session_state.lc_messages.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with tab_research:
        with st.chat_message("user"):
            st.markdown(prompt)
        run_research(prompt)
    st.rerun()