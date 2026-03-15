from __future__ import annotations

import subprocess
import time
import requests
import streamlit as st


# -----------------------------
# API Configuration
# -----------------------------
API_URL = "http://localhost:8000"


# -----------------------------
# Start FastAPI Automatically
# -----------------------------
def start_api():
    if "api_started" not in st.session_state:
        try:
            subprocess.Popen(
                ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(3)
            st.session_state.api_started = True
        except Exception as e:
            st.error(f"Failed to start API: {e}")


start_api()


# -----------------------------
# API Calls
# -----------------------------
def fetch_threads(api_url: str):
    response = requests.get(f"{api_url}/threads", timeout=10)
    response.raise_for_status()
    return response.json()


def start_session(api_url: str, thread_id: str):
    response = requests.post(
        f"{api_url}/start_session",
        json={"thread_id": thread_id},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def switch_thread(api_url: str, session_id: str, thread_id: str):
    response = requests.post(
        f"{api_url}/switch_thread",
        json={"session_id": session_id, "thread_id": thread_id},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def ask(api_url: str, session_id: str, text: str, search_outside_thread: bool):
    response = requests.post(
        f"{api_url}/ask",
        json={
            "session_id": session_id,
            "text": text,
            "search_outside_thread": search_outside_thread,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def reset_session(api_url: str, session_id: str | None):
    response = requests.post(
        f"{api_url}/reset_session",
        json={"session_id": session_id},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Email + Attachment RAG", layout="wide")
st.title("Email + Attachment RAG with Thread Memory")


# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "debug" not in st.session_state:
    st.session_state.debug = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("Session")

    api_url = st.text_input("API URL", value=API_URL)

    search_outside_thread = st.toggle(
        "Search outside thread",
        value=False
    )

    try:

        threads = fetch_threads(api_url)

        labels = {
            item["thread_id"]: f"{item['thread_id']} | {item.get('subject') or 'No subject'} | {item.get('message_count',0)} messages"
            for item in threads
        }

        selected_thread = st.selectbox(
            "Thread selector",
            list(labels.keys()),
            format_func=lambda key: labels[key],
        )

    except Exception as exc:

        threads = []
        selected_thread = None

        st.error(f"Unable to load threads: {exc}")

    if st.button("Start session", disabled=selected_thread is None):

        result = start_session(api_url, selected_thread)

        st.session_state.session_id = result["session_id"]
        st.session_state.messages = []
        st.session_state.debug = None

    if st.button(
        "Switch thread",
        disabled=selected_thread is None
        or st.session_state.session_id is None,
    ):

        result = switch_thread(
            api_url,
            st.session_state.session_id,
            selected_thread,
        )

        st.session_state.session_id = result["session_id"]
        st.session_state.messages = []
        st.session_state.debug = None

    if st.button("Reset session"):

        reset_session(api_url, st.session_state.session_id)

        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.debug = None

    st.caption(
        f"Current session: {st.session_state.session_id or 'None'}"
    )


# -----------------------------
# Chat Messages
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


prompt = st.chat_input(
    "Ask about the selected email thread or its attachments"
)


if prompt:

    if not st.session_state.session_id:

        st.warning("Start a session before asking questions.")

    else:

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):

            st.markdown(prompt)

        result = ask(
            api_url,
            st.session_state.session_id,
            prompt,
            search_outside_thread,
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )

        st.session_state.debug = result

        with st.chat_message("assistant"):

            st.markdown(result["answer"])


# -----------------------------
# Debug Panel
# -----------------------------
if st.session_state.debug:

    with st.expander("Debug panel", expanded=True):

        st.subheader("Rewritten query")
        st.code(st.session_state.debug["rewrite"])

        st.subheader("Retrieved documents")
        st.json(st.session_state.debug["retrieved"])

        st.subheader("Citations")
        st.json(st.session_state.debug["citations"])
