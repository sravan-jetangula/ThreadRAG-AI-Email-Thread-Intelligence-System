from __future__ import annotations

import streamlit as st
import json
from pathlib import Path

from rag_pipeline import RAGPipeline


INDEX_DIR = "indexes"

st.set_page_config(page_title="ThreadRAG", layout="wide")
st.title("ThreadRAG – AI Email Thread Intelligence System")


@st.cache_resource
def load_pipeline():
    return RAGPipeline(index_dir=INDEX_DIR)


pipeline = load_pipeline()


def load_threads():
    threads_path = Path(INDEX_DIR) / "threads.json"
    if not threads_path.exists():
        return []
    return json.loads(threads_path.read_text())


threads = load_threads()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None


with st.sidebar:

    st.header("Thread Selection")

    if threads:
        labels = {
            item["thread_id"]: f"{item['thread_id']} | {item.get('subject') or 'No subject'} | {item.get('message_count',0)} messages"
            for item in threads
        }

        selected_thread = st.selectbox(
            "Thread selector",
            list(labels.keys()),
            format_func=lambda key: labels[key],
        )

        if st.button("Select thread"):
            st.session_state.thread_id = selected_thread
            st.session_state.messages = []

    else:
        st.warning("No threads available. Run ingestion first.")

    st.caption(f"Current thread: {st.session_state.thread_id or 'None'}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask about the selected email thread")

if prompt:

    if not st.session_state.thread_id:
        st.warning("Select a thread first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    result = pipeline.ask(
        question=prompt,
        thread_id=st.session_state.thread_id,
    )

    answer = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander("Debug panel"):
        st.subheader("Rewritten query")
        st.code(result.get("rewrite"))

        st.subheader("Retrieved documents")
        st.json(result.get("retrieved"))

        st.subheader("Citations")
        st.json(result.get("citations"))
