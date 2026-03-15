from __future__ import annotations

import streamlit as st

from rag_pipeline import (
    get_threads,
    start_session,
    switch_thread,
    ask_question,
    reset_session,
)

st.set_page_config(
    page_title="ThreadRAG - Email Thread Intelligence",
    layout="wide",
)

st.title("ThreadRAG - AI Email Thread Intelligence System")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "debug" not in st.session_state:
    st.session_state.debug = None


with st.sidebar:

    st.header("Session Controls")

    try:
        threads = get_threads()

        labels = {
            t["thread_id"]: f"{t['thread_id']} | {t.get('subject','No subject')} | {t.get('message_count',0)} messages"
            for t in threads
        }

        selected_thread = st.selectbox(
            "Thread Selector",
            list(labels.keys()),
            format_func=lambda x: labels[x],
        )

    except Exception as e:

        threads = []
        selected_thread = None
        st.error(f"Unable to load threads: {e}")

    if st.button("Start Session", disabled=selected_thread is None):

        st.session_state.session_id = start_session(selected_thread)
        st.session_state.thread_id = selected_thread
        st.session_state.messages = []
        st.session_state.debug = None

    if st.button(
        "Switch Thread",
        disabled=selected_thread is None
        or st.session_state.session_id is None,
    ):

        st.session_state.session_id = switch_thread(
            st.session_state.session_id,
            selected_thread,
        )

        st.session_state.thread_id = selected_thread
        st.session_state.messages = []
        st.session_state.debug = None

    if st.button("Reset Session"):

        reset_session(st.session_state.session_id)

        st.session_state.session_id = None
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.session_state.debug = None

    st.caption(
        f"Current Session: {st.session_state.session_id or 'None'}"
    )


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input(
    "Ask a question about this email thread or its attachments"
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

        result = ask_question(
            session_id=st.session_state.session_id,
            question=prompt,
        )

        answer = result.get("answer", "No response generated.")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.session_state.debug = result

        with st.chat_message("assistant"):
            st.markdown(answer)


if st.session_state.debug:

    with st.expander("Debug Panel", expanded=False):

        st.subheader("Query Rewrite")
        st.code(st.session_state.debug.get("rewrite"))

        st.subheader("Retrieved Documents")
        st.json(st.session_state.debug.get("retrieved"))

        st.subheader("Citations")
        st.json(st.session_state.debug.get("citations"))
