import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from db.history import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversations,
    get_messages,
    init_db,
    rename_conversation,
)
from pipelines.answering_pipeline import AnsweringPipeline

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
init_db()

st.set_page_config(
    page_title="RAG Answer Pipeline",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0d0d0d;
    color: #e8e4dc;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace; }

/* ── New chat button ── */
.new-chat-btn > button {
    background: transparent !important;
    border: 1px solid #c8f55a !important;
    color: #c8f55a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: .08em !important;
    width: 100% !important;
    margin-bottom: 1.2rem !important;
    transition: background .2s !important;
}
.new-chat-btn > button:hover {
    background: #c8f55a22 !important;
}

/* ── Conversation list items ── */
.conv-item > button {
    background: transparent !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    color: #888 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .78rem !important;
    text-align: left !important;
    width: 100% !important;
    padding: .35rem .6rem !important;
    transition: all .15s !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.conv-item > button:hover {
    border-left-color: #c8f55a !important;
    color: #e8e4dc !important;
    background: #ffffff08 !important;
}
.conv-item-active > button {
    border-left-color: #c8f55a !important;
    color: #c8f55a !important;
    background: #c8f55a0f !important;
}

/* ── Chat container ── */
.chat-wrapper {
    max-width: 780px;
    margin: 0 auto;
    padding: 2rem 1rem 8rem;
}

/* ── Messages ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 1.2rem 0;
}
.msg-user .bubble {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px 16px 4px 16px;
    padding: .75rem 1.1rem;
    max-width: 72%;
    font-size: .88rem;
    line-height: 1.6;
    color: #e8e4dc;
}
.msg-assistant {
    display: flex;
    justify-content: flex-start;
    margin: 1.2rem 0;
}
.msg-assistant .bubble {
    background: #141414;
    border: 1px solid #1e2d00;
    border-left: 3px solid #c8f55a;
    border-radius: 4px 16px 16px 16px;
    padding: .75rem 1.1rem;
    max-width: 84%;
    font-size: .88rem;
    line-height: 1.8;
    color: #d4d0c8;
}

/* ── Steps expander ── */
.step-log {
    margin-top: .5rem;
    font-size: .75rem;
    color: #555;
    border-left: 1px solid #222;
    padding-left: .75rem;
}
.step-badge {
    display: inline-block;
    background: #1a2200;
    color: #c8f55a;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: .7rem;
    margin-bottom: .3rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: .06em;
}

/* ── Input bar ── */
.stChatInput > div {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    font-family: 'DM Mono', monospace !important;
}
.stChatInput > div:focus-within {
    border-color: #c8f55a44 !important;
    box-shadow: 0 0 0 2px #c8f55a18 !important;
}

/* ── Header ── */
.page-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: .12em;
    color: #c8f55a;
    text-transform: uppercase;
    margin-bottom: 2rem;
    padding-bottom: .6rem;
    border-bottom: 1px solid #1a1a1a;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    margin-top: 5rem;
    color: #333;
}
.empty-state .icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}
.empty-state p {
    font-size: .82rem;
    line-height: 1.8;
}
.empty-state span {
    color: #c8f55a;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }

/* ── Sidebar section label ── */
.sidebar-label {
    font-family: 'Syne', sans-serif;
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .14em;
    color: #444;
    text-transform: uppercase;
    margin: 1rem 0 .4rem .2rem;
}

/* ── Delete button ── */
.del-btn > button {
    background: transparent !important;
    border: none !important;
    color: #333 !important;
    font-size: .75rem !important;
    padding: .2rem .4rem !important;
    width: 100% !important;
}
.del-btn > button:hover { color: #ff4444 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_conversation(conv_id: str):
    st.session_state.active_conversation_id = conv_id
    raw = get_messages(conv_id)
    st.session_state.messages = raw
    st.session_state.pipeline_steps = {
        m["id"]: json.loads(m["step_logs"])
        for m in raw
        if m["role"] == "assistant" and m["step_logs"]
    }


def new_conversation():
    st.session_state.active_conversation_id = None
    st.session_state.messages = []
    st.session_state.pipeline_steps = {}


def run_pipeline(query: str, history: list[dict]) -> tuple[str, dict]:
    steps = {}
    try:
        if history:
            if len(history) > 20:
                recent_history = history[-20:-1]
            else:
                recent_history = history[:-1]
            formatted_history = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in recent_history[:-1]  # on exclut le message courant déjà ajouté
            )
            full_input = (
                f"<conversation_history>\n{formatted_history}\n</conversation_history>\n\n"
                f"<current_question>\n{query}\n</current_question>"
            )
        else:
            full_input = query

        pipeline = AnsweringPipeline()
        response = pipeline.answer(full_input)

    except Exception as e:
        response = f"❌ Erreur pipeline : {e}"
        steps = {}

    return response, steps


STEP_LABELS = {
    "query_rewrite":   "① Query rewrite",
    "bm25_retrieval":  "② BM25 retrieval",
    "chunk_filtering": "③ Chunk filtering",
    "answer_draft":    "④ Answer draft",
    "critic_pass":     "⑤ Critic pass",
}


def render_step_logs(steps: dict):
    if not steps:
        return
    with st.expander("Pipeline steps", expanded=False):
        for key, label in STEP_LABELS.items():
            if key in steps:
                st.markdown(
                    f'<span class="step-badge">{label}</span>'
                    f'<div class="step-log">{steps[key]}</div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="page-header">◈ RAG Pipeline</div>', unsafe_allow_html=True)

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("+ New conversation", key="new_conv"):
        new_conversation()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    conversations = get_conversations()

    if conversations:
        st.markdown('<div class="sidebar-label">History</div>', unsafe_allow_html=True)
        for conv in conversations:
            is_active = conv["id"] == st.session_state.active_conversation_id
            col_title, col_del = st.columns([5, 1])
            with col_title:
                css_class = "conv-item-active" if is_active else "conv-item"
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                if st.button(
                        conv["title"],
                        key=f"conv_{conv['id']}",
                        help=conv["updated_at"][:16].replace("T", " "),
                ):
                    load_conversation(conv["id"])
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with col_del:
                st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                if st.button("✕", key=f"del_{conv['id']}"):
                    delete_conversation(conv["id"])
                    if st.session_state.active_conversation_id == conv["id"]:
                        new_conversation()
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:#333;font-size:.78rem;margin-top:1rem;">'
            "No conversations yet.</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">◈</div>
        <p>Ask anything.<br/>
        The pipeline will <span>rewrite your query</span>, retrieve relevant chunks,<br/>
        filter them, then iterate with an <span>Answer / Critic</span> loop.</p>
    </div>
    """, unsafe_allow_html=True)

# Render history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user"><div class="bubble">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="msg-assistant"><div class="bubble">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
        steps = st.session_state.pipeline_steps.get(msg["id"], {})
        render_step_logs(steps)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask your question…"):

    # ── Créer la conversation si nécessaire ────────────────────────────────
    if not st.session_state.active_conversation_id:
        conv_id = create_conversation(prompt)
        st.session_state.active_conversation_id = conv_id
    else:
        conv_id = st.session_state.active_conversation_id

    # ── Persister + afficher le message user ───────────────────────────────
    user_mid = add_message(conv_id, "user", prompt)
    st.session_state.messages.append(
        {"id": user_mid, "role": "user", "content": prompt, "step_logs": None}
    )

    # ── Affichage immédiat ─────────────────────────────────────────────────
    st.markdown(
        f'<div class="msg-user"><div class="bubble">{prompt}</div></div>',
        unsafe_allow_html=True,
    )

    # ── Run pipeline ────────────────────────────────────────────────────────
    with st.spinner("Running pipeline…"):
        response, steps = run_pipeline(prompt, st.session_state.messages)

    # ── Persister + afficher la réponse ────────────────────────────────────
    steps_json = json.dumps(steps, ensure_ascii=False) if steps else None
    asst_mid = add_message(conv_id, "assistant", response, steps_json)
    st.session_state.messages.append(
        {"id": asst_mid, "role": "assistant", "content": response, "step_logs": steps_json}
    )
    st.session_state.pipeline_steps[asst_mid] = steps

    st.rerun()