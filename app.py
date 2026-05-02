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


# --------------- main chat ---------------
st.title("🎓 NED Admission Assistant")
st.caption(
    "Ask about admissions at NED University of Engineering & Technology. "
    "Answers come from official NED sources only."
)

if chain.store.count() == 0:
    st.warning(
        "The knowledge base is empty. Run `python main.py refresh` from a "
        "terminal, or click **Refresh NED data** in the sidebar."
    )

# Replay history.
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

q = st.chat_input("Ask about NED admissions...")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching official NED data..."):
            try:
                result = chain.answer(q, session_id=st.session_state.session_id)
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
