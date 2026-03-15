from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Email + Attachment RAG", layout="wide")
st.title("Email + Attachment RAG with Thread Memory")

# ----------------------------
# Session state initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "debug" not in st.session_state:
    st.session_state.debug = None
if "threads" not in st.session_state:
    st.session_state.threads = []

# ----------------------------
# Simulated backend functions
# ----------------------------
def fetch_threads_local() -> list[dict]:
    """Simulate fetching threads."""
    # Replace with real data source or RAG index
    return [
        {"thread_id": "t1", "subject": "Project Update", "message_count": 3},
        {"thread_id": "t2", "subject": "Invoice Details", "message_count": 2},
        {"thread_id": "t3", "subject": "Meeting Notes", "message_count": 5},
    ]

def start_session_local(thread_id: str) -> str:
    """Simulate starting a session."""
    return f"session_{thread_id}"

def switch_thread_local(thread_id: str) -> str:
    """Simulate switching threads."""
    return f"session_{thread_id}"

def reset_session_local() -> None:
    """Simulate resetting session."""
    return None

def ask_local(thread_id: str, prompt: str, search_outside_thread: bool) -> dict:
    """Simulate answering a query with a RAG-style response."""
    # This is where you integrate your actual model / document retrieval
    answer = f"Simulated answer for thread '{thread_id}': {prompt}"
    return {
        "answer": answer,
        "rewrite": f"Rewritten query for: {prompt}",
        "retrieved": [{"doc_id": "doc1", "content": "Sample content"}],
        "citations": [{"source": "doc1"}]
    }

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Session")
    search_outside_thread = st.toggle("Search outside thread", value=False)

    # Load threads once
    if not st.session_state.threads:
        st.session_state.threads = fetch_threads_local()

    labels = {
        item["thread_id"]: f"{item['thread_id']} | {item.get('subject') or 'No subject'} | {item.get('message_count', 0)} messages"
        for item in st.session_state.threads
    }
    selected_thread = st.selectbox("Thread selector", list(labels.keys()), format_func=lambda key: labels[key])

    if st.button("Start session", disabled=selected_thread is None):
        st.session_state.session_id = start_session_local(selected_thread)
        st.session_state.messages = []
        st.session_state.debug = None

    if st.button("Switch thread", disabled=selected_thread is None or st.session_state.session_id is None):
        st.session_state.session_id = switch_thread_local(selected_thread)
        st.session_state.messages = []
        st.session_state.debug = None

    if st.button("Reset session"):
        reset_session_local()
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.debug = None

    st.caption(f"Current session: {st.session_state.session_id or 'None'}")

# ----------------------------
# Chat messages display
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# Chat input
# ----------------------------
prompt = st.chat_input("Ask about the selected email thread or its attachments")

if prompt:
    if not st.session_state.session_id:
        st.warning("Start a session before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        result = ask_local(selected_thread, prompt, search_outside_thread)
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        st.session_state.debug = result

        with st.chat_message("assistant"):
            st.markdown(result["answer"])

# ----------------------------
# Debug panel
# ----------------------------
if st.session_state.debug:
    with st.expander("Debug panel", expanded=True):
        st.subheader("Rewritten query")
        st.code(st.session_state.debug["rewrite"])
        st.subheader("Retrieved documents")
        st.json(st.session_state.debug["retrieved"])
        st.subheader("Citations")
        st.json(st.session_state.debug["citations"])
