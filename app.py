from __future__ import annotations

import requests
import streamlit as st

# Default API URL (change if you deploy backend)
DEFAULT_API_URL = ""

st.set_page_config(
    page_title="ThreadRAG - Email Thread Intelligence",
    layout="wide"
)

st.title("ThreadRAG – AI Email Thread Intelligence System")

# -----------------------------
# Session State Initialization
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "debug" not in st.session_state:
    st.session_state.debug = None

# -----------------------------
# API Functions
# -----------------------------

def safe_request(method, url, **kwargs):
    try:
        response = requests.request(method, url, timeout=60, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {e}")
        return None


def fetch_threads(api_url):
    return safe_request("GET", f"{api_url}/threads")


def start_session(api_url, thread_id):
    return safe_request(
        "POST",
        f"{api_url}/start_session",
        json={"thread_id": thread_id},
    )


def switch_thread(api_url, session_id, thread_id):
    return safe_request(
        "POST",
        f"{api_url}/switch_thread",
        json={"session_id": session_id, "thread_id": thread_id},
    )


def ask(api_url, session_id, text, search_outside_thread):
    return safe_request(
        "POST",
        f"{api_url}/ask",
        json={
            "session_id": session_id,
            "text": text,
            "search_outside_thread": search_outside_thread,
        },
    )


def reset_session(api_url, session_id):
    return safe_request(
        "POST",
        f"{api_url}/reset_session",
        json={"session_id": session_id},
    )


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("Session Settings")

    api_url = st.text_input(
        "Backend API URL",
        value=DEFAULT_API_URL,
        placeholder="https://your-api-url.com"
    )

    search_outside_thread = st.toggle(
        "Search outside thread",
        value=False
    )

    threads = []

    if api_url:

        threads = fetch_threads(api_url)

        if threads:

            labels = {
                item["thread_id"]: f"{item['thread_id']} | {item.get('subject','No subject')} | {item.get('message_count',0)} messages"
                for item in threads
            }

            selected_thread = st.selectbox(
                "Select Email Thread",
                list(labels.keys()),
                format_func=lambda key: labels[key],
            )

        else:
            selected_thread = None

    else:
        selected_thread = None
        st.info("Enter your backend API URL")

    if st.button("Start Session", disabled=selected_thread is None):

        result = start_session(api_url, selected_thread)

        if result:
            st.session_state.session_id = result["session_id"]
            st.session_state.messages = []
            st.success("Session started")

    if st.button(
        "Switch Thread",
        disabled=selected_thread is None or st.session_state.session_id is None,
    ):

        result = switch_thread(
            api_url,
            st.session_state.session_id,
            selected_thread,
        )

        if result:
            st.session_state.messages = []
            st.success("Thread switched")

    if st.button("Reset Session"):

        if st.session_state.session_id:

            reset_session(api_url, st.session_state.session_id)

        st.session_state.session_id = None
        st.session_state.messages = []

    st.caption(f"Current Session: {st.session_state.session_id or 'None'}")


# -----------------------------
# Chat Interface
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask about the email thread or attachments")

if prompt:

    if not st.session_state.session_id:

        st.warning("Start a session first.")

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

        if result:

            answer = result.get("answer", "No response")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            st.session_state.debug = result

            with st.chat_message("assistant"):
                st.markdown(answer)


# -----------------------------
# Debug Panel
# -----------------------------

if st.session_state.debug:

    with st.expander("Debug Panel", expanded=False):

        st.subheader("Rewritten Query")
        st.code(st.session_state.debug.get("rewrite", ""))

        st.subheader("Retrieved Documents")
        st.json(st.session_state.debug.get("retrieved", []))

        st.subheader("Citations")
        st.json(st.session_state.debug.get("citations", []))
