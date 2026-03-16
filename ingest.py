import streamlit as st
from rag_pipeline import RagPipeline

st.set_page_config(
    page_title="ThreadRAG - Email Thread Intelligence",
    layout="wide"
)

st.title("ThreadRAG – AI Email Thread Intelligence System")

# ---------------------------
# Load RAG Pipeline
# ---------------------------

@st.cache_resource
def load_pipeline():
    return RagPipeline()

pipeline = load_pipeline()

# ---------------------------
# Session State
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "debug" not in st.session_state:
    st.session_state.debug = None

# ---------------------------
# Sidebar
# ---------------------------

with st.sidebar:

    st.header("Thread Settings")

    threads = pipeline.get_threads()

    if threads:

        labels = {
            t["thread_id"]: f"{t['thread_id']} | {t.get('subject','No subject')} | {t.get('message_count',0)} messages"
            for t in threads
        }

        selected_thread = st.selectbox(
            "Select Thread",
            list(labels.keys()),
            format_func=lambda key: labels[key]
        )

    else:
        selected_thread = None
        st.warning("No threads found")

    if st.button("Start Session"):

        st.session_state.thread_id = selected_thread
        st.session_state.messages = []
        st.success("Session started")

    if st.button("Reset Session"):

        st.session_state.thread_id = None
        st.session_state.messages = []

    st.caption(f"Current thread: {st.session_state.thread_id or 'None'}")

# ---------------------------
# Chat Interface
# ---------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about the email thread or attachments")

if prompt:

    if not st.session_state.thread_id:

        st.warning("Start a session first")

    else:

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        result = pipeline.ask(
            thread_id=st.session_state.thread_id,
            query=prompt
        )

        answer = result.get("answer", "No answer found")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.session_state.debug = result

        with st.chat_message("assistant"):
            st.markdown(answer)

# ---------------------------
# Debug Panel
# ---------------------------

if st.session_state.debug:

    with st.expander("Debug Panel", expanded=False):

        st.subheader("Rewritten Query")
        st.code(st.session_state.debug.get("rewrite", ""))

        st.subheader("Retrieved Documents")
        st.json(st.session_state.debug.get("retrieved", []))

        st.subheader("Citations")
        st.json(st.session_state.debug.get("citations", []))
