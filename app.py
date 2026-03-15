```python
import streamlit as st
from rag_pipeline import RagPipeline

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="ThreadRAG - Email Thread Intelligence",
    layout="wide"
)

st.title("ThreadRAG – AI Email Thread Intelligence System")

# -------------------------------
# Load RAG Pipeline (cached)
# -------------------------------
@st.cache_resource
def load_pipeline():
    pipeline = RagPipeline()
    return pipeline

pipeline = load_pipeline()

# -------------------------------
# Session State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Chat History
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# Chat Input
# -------------------------------
prompt = st.chat_input("Ask a question about the email dataset")

if prompt:

    # show user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # run RAG pipeline
    try:
        result = pipeline.ask(prompt)
        answer = result.get("answer", "No answer generated.")

    except Exception as e:
        answer = f"Error: {str(e)}"

    # show assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

# -------------------------------
# Debug Panel
# -------------------------------
if "debug" in st.session_state:

    with st.expander("Debug Panel", expanded=False):

        debug = st.session_state.debug

        if "rewrite" in debug:
            st.subheader("Query Rewrite")
            st.code(debug["rewrite"])

        if "retrieved" in debug:
            st.subheader("Retrieved Documents")
            st.json(debug["retrieved"])

        if "citations" in debug:
            st.subheader("Citations")
            st.json(debug["citations"])
```
