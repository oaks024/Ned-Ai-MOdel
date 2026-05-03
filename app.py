"""Streamlit chatbot for the NED admission assistant.

Run with:
    streamlit run app.py
"""
import uuid
import streamlit as st

from src.config import Config
from src.rag_chain import RAGChain

st.set_page_config(
    page_title="NED Admission Assistant",
    page_icon="🎓",
    layout="wide",
)

# --------------- styling ---------------
st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; }
    .ned-header {
        background: linear-gradient(135deg, #003366 0%, #0066cc 100%);
        color: #ffffff;
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(0, 51, 102, 0.18);
    }
    .ned-header h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
    }
    .ned-header p {
        color: #d6e4f0;
        margin: 0.35rem 0 0;
        font-size: 0.95rem;
    }
    .profile-chip {
        background: #f1f6fb;
        border-left: 4px solid #0066cc;
        padding: 0.55rem 0.9rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #1a2740;
        margin-bottom: 1rem;
    }
    .starter-label {
        font-size: 0.85rem;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
        margin-top: 0.5rem;
    }
    div.stButton > button[kind="secondary"] {
        text-align: left;
        white-space: normal;
        height: auto;
        padding: 0.7rem 0.9rem;
        line-height: 1.35;
    }
</style>
""",
    unsafe_allow_html=True,
)

STARTER_QUESTIONS = [
    "What is the admission process at NED?",
    "What programs are offered for undergraduate admission?",
    "What documents are required for admission?",
    "What is the fee structure?",
    "Is there an entry test? How does it work?",
    "How can I apply online and where do I find the prospectus?",
]


@st.cache_resource(show_spinner="Loading models and vector store...")
def load_chain():
    cfg = Config()
    cfg.validate()
    return RAGChain(cfg), cfg


# --------------- session state ---------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

try:
    chain, cfg = load_chain()
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()


# --------------- sidebar: profile + controls ---------------
LEVEL_OPTIONS = ["", "Intermediate / FSc", "A-Levels", "Bachelor", "Master"]
CATEGORY_OPTIONS = ["", "Undergraduate", "Postgraduate", "PhD"]


def _safe_index(options, value):
    return options.index(value) if value in options else 0


with st.sidebar:
    st.header("Your profile")
    st.caption(
        "Used only to personalize the chat. Admission facts always come "
        "from the official NED data."
    )
    profile = chain.memory.get_profile(st.session_state.session_id)

    name = st.text_input("Name", value=profile.get("user_name") or "")
    program = st.text_input(
        "Interested program", value=profile.get("interested_program") or ""
    )
    level = st.selectbox(
        "Education level",
        LEVEL_OPTIONS,
        index=_safe_index(LEVEL_OPTIONS, profile.get("education_level") or ""),
    )
    category = st.selectbox(
        "Admission category",
        CATEGORY_OPTIONS,
        index=_safe_index(CATEGORY_OPTIONS, profile.get("category") or ""),
    )

    if st.button("Save profile", use_container_width=True):
        chain.memory.update_profile(
            st.session_state.session_id,
            user_name=name or None,
            interested_program=program or None,
            education_level=level or None,
            category=category or None,
        )
        st.success("Profile saved.")

    st.divider()
    st.subheader("Knowledge base")
    st.caption(f"Indexed chunks: **{chain.store.count()}**")
    if st.button("🔄 Refresh NED data", use_container_width=True):
        with st.spinner("Re-scraping and rebuilding... this can take several minutes."):
            from main import cmd_scrape, cmd_build
            cmd_scrape(cfg)
            cmd_build(cfg, rebuild=True)
        st.cache_resource.clear()
        st.success("Knowledge base refreshed. Reloading...")
        st.rerun()

    st.divider()
    if st.button("Clear chat & memory", use_container_width=True):
        st.session_state.messages = []
        chain.memory.clear_session(st.session_state.session_id)
        st.rerun()


# --------------- main: header + profile chip ---------------
st.markdown(
    """
<div class="ned-header">
    <h1>🎓 NED Admission Assistant</h1>
    <p>Official admission information for NED University of Engineering &amp; Technology — every answer cites a real NED source.</p>
</div>
""",
    unsafe_allow_html=True,
)

profile = chain.memory.get_profile(st.session_state.session_id)
chip_bits = []
if profile.get("user_name"):
    chip_bits.append(f"👤 <b>{profile['user_name']}</b>")
if profile.get("interested_program"):
    chip_bits.append(f"📘 {profile['interested_program']}")
if profile.get("education_level"):
    chip_bits.append(f"🎓 {profile['education_level']}")
if profile.get("category"):
    chip_bits.append(f"🏷️ {profile['category']}")
if chip_bits:
    st.markdown(
        f'<div class="profile-chip">Personalizing for: {" &nbsp;•&nbsp; ".join(chip_bits)}</div>',
        unsafe_allow_html=True,
    )

if chain.store.count() == 0:
    st.warning(
        "The knowledge base is empty. Run `python main.py refresh` from a "
        "terminal, or click **Refresh NED data** in the sidebar."
    )


# --------------- starter questions (only when chat is empty) ---------------
def _set_pending(question: str) -> None:
    st.session_state.pending_question = question


if not st.session_state.messages and chain.store.count() > 0:
    st.markdown('<div class="starter-label">Try one of these to get started</div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(STARTER_QUESTIONS):
        cols[i % 2].button(
            q,
            key=f"starter_{i}",
            use_container_width=True,
            on_click=_set_pending,
            args=(q,),
        )


# --------------- replay history ---------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    label = s.get("title") or s["url"]
                    if s.get("page_number"):
                        label += f" (page {s['page_number']})"
                    st.markdown(f"- [{label}]({s['url']})")


# --------------- handle new question (chat input OR starter button) ---------------
typed = st.chat_input("Ask about NED admissions...")
question = typed or st.session_state.pending_question
if st.session_state.pending_question:
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        spinner_text = (
            f"Searching official NED admission data "
            f"({chain.store.count()} chunks indexed)..."
        )
        with st.spinner(spinner_text):
            try:
                result = chain.answer(question, session_id=st.session_state.session_id)
            except Exception as e:
                result = {"answer": f"Error: {e}", "sources": []}
        st.markdown(result["answer"])
        if result.get("sources"):
            with st.expander("Sources"):
                for s in result["sources"]:
                    label = s.get("title") or s["url"]
                    if s.get("page_number"):
                        label += f" (page {s['page_number']})"
                    st.markdown(f"- [{label}]({s['url']})")
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })
