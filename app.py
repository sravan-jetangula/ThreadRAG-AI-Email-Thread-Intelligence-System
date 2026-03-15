import streamlit as st

st.set_page_config(page_title="Email + Attachment RAG", layout="wide")
st.title("Email + Attachment RAG with Thread Memory")

# ---------------- Demo Email Threads ----------------

threads = {
    "finance-thread": {
        "subject": "Budget discussion",
        "messages": [
            "The Q3 budget proposal was attached in the email.",
            "Please review the financial report before Friday.",
            "The marketing team requested an additional allocation."
        ]
    },
    "project-thread": {
        "subject": "AI Project Updates",
        "messages": [
            "The AI virtual doctor prototype has been deployed.",
            "We integrated LangChain and RAG for document retrieval.",
            "Testing will begin next week."
        ]
    }
}

# ---------------- Session State ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# ---------------- Sidebar ----------------

with st.sidebar:

    st.header("Session")

    labels = {
        key: f"{key} | {value['subject']} | {len(value['messages'])} messages"
        for key, value in threads.items()
    }

    selected_thread = st.selectbox(
        "Thread selector",
        list(labels.keys()),
        format_func=lambda key: labels[key]
    )

    if st.button("Start session"):
        st.session_state.thread_id = selected_thread
        st.session_state.messages = []

    if st.button("Reset session"):
        st.session_state.thread_id = None
        st.session_state.messages = []

    st.caption(f"Current thread: {st.session_state.thread_id or 'None'}")

# ---------------- Chat ----------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about the selected email thread")

if prompt:

    if not st.session_state.thread_id:
        st.warning("Start a session first")

    else:

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        thread_messages = threads[st.session_state.thread_id]["messages"]

        # simple retrieval (keyword match)
        answer = "I couldn't find relevant info."

        for msg in thread_messages:
            if any(word.lower() in msg.lower() for word in prompt.split()):
                answer = msg
                break

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)
