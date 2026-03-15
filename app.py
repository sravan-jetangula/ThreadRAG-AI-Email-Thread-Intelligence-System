import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Email + Attachment RAG", layout="wide")
st.title("Email + Attachment RAG with Thread Memory")

# Load pipeline once
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "debug" not in st.session_state:
    st.session_state.debug = None

# ---------------- Sidebar ----------------

with st.sidebar:
    st.header("Session")

    threads = pipeline.list_threads()

    if threads:
        labels = {
            item["thread_id"]: f"{item['thread_id']} | {item.get('subject','No subject')} | {item.get('message_count',0)} messages"
            for item in threads
        }

        selected_thread = st.selectbox(
            "Thread selector",
            list(labels.keys()),
            format_func=lambda key: labels[key]
        )
    else:
        selected_thread = None
        st.warning("No threads available")

    if st.button("Start session", disabled=selected_thread is None):
        result = pipeline.start_session(selected_thread)
        st.session_state.session_id = result["session_id"]
        st.session_state.messages = []
        st.session_state.debug = None

    if st.button("Switch thread", disabled=selected_thread is None or st.session_state.session_id is None):
        result = pipeline.switch_thread(
            selected_thread,
            session_id=st.session_state.session_id
        )
        st.session_state.session_id = result["session_id"]
        st.session_state.messages = []
        st.session_state.debug = None

    if st.button("Reset session"):
        pipeline.reset_session(st.session_state.session_id)
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.debug = None

    st.caption(f"Current session: {st.session_state.session_id or 'None'}")

# ---------------- Chat ----------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about the selected email thread or its attachments")

if prompt:
    if not st.session_state.session_id:
        st.warning("Start a session before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        result = pipeline.ask(
            session_id=st.session_state.session_id,
            text=prompt
        )

        answer = result["answer"]

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.debug = result

        with st.chat_message("assistant"):
            st.markdown(answer)

# ---------------- Debug Panel ----------------

if st.session_state.debug:
    with st.expander("Debug panel", expanded=True):
        st.subheader("Rewritten query")
        st.code(st.session_state.debug["rewrite"])

        st.subheader("Retrieved documents")
        st.json(st.session_state.debug["retrieved"])

        st.subheader("Citations")
        st.json(st.session_state.debug["citations"])
