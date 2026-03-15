from __future__ import annotations
import os
import streamlit as st

# LangChain + Groq imports
from langchain.embeddings.groq import GroqEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.groq import Groq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ------------------------------------
# ⚙️ Configure API keys (via env vars)
# ------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ Please set your Groq API key in Streamlit secrets as GROQ_API_KEY")
    st.stop()

# ------------------------------------
# 📌 Load / index your documents here
# ------------------------------------
@st.cache_resource
def build_vector_store() -> FAISS:
    """
    Build FAISS vector store with Groq embeddings.
    Replace these sample docs with your email + attachment text.
    """
    raw_docs = [
        Document(page_content="Project update email content..."),
        Document(page_content="Invoice PDF text content..."),
        Document(page_content="Meeting notes and summary content..."),
    ]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(raw_docs)

    embeddings = GroqEmbeddings(api_key=GROQ_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vector_store = build_vector_store()

# ------------------------------------
# 🧠 RAG Answer function
# ------------------------------------
def ask_groq_rag(prompt: str, k: int = 3) -> dict[str, object]:
    """
    Retrieves top-k most relevant text chunks and uses Groq LLM for answer.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    qa = RetrievalQA.from_chain_type(
        llm=Groq(api_key=GROQ_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    result = qa({"query": prompt})
    docs = result.get("source_documents", [])

    retrieved_docs = [
        {"doc_id": f"doc_{i+1}", "content": doc.page_content} 
        for i, doc in enumerate(docs)
    ]
    citations = [{"source": f"doc_{i+1}"} for i in range(len(docs))]

    return {
        "answer": result["result"],
        "rewrite": prompt,
        "retrieved": retrieved_docs,
        "citations": citations,
    }


# ------------------------------------
# 🎯 Streamlit UI & Session Logic
# ------------------------------------
st.set_page_config(page_title="Email + Attachment RAG (Groq)", layout="wide")
st.title("📬 Email + Attachment RAG powered by Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "debug" not in st.session_state:
    st.session_state.debug = None

# Sidebar: Thread selector simulation
with st.sidebar:
    st.header("Session Controls")
    threads = {
        "t1": "Project Update",
        "t2": "Invoice Details",
        "t3": "Meeting Notes",
    }
    selected_thread = st.selectbox("Choose thread", list(threads.keys()))
    
    if st.button("Start session"):
        st.session_state.session_id = f"{selected_thread}_session"
        st.session_state.messages = []
        st.session_state.debug = None

    if st.button("Reset session"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.debug = None

    st.caption(f"Current Session: {st.session_state.session_id or 'None'}")

# Show existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Ask about this thread or its attachments:")

if prompt:
    if not st.session_state.session_id:
        st.warning("Start the session first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = ask_groq_rag(prompt)
        st.session_state.debug = response
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

        with st.chat_message("assistant"):
            st.markdown(response["answer"])

# Debug panel
if st.session_state.debug:
    with st.expander("🔍 Debug Info", expanded=True):
        st.subheader("Rewritten Query")
        st.code(st.session_state.debug["rewrite"])
        st.subheader("Retrieved Docs")
        st.json(st.session_state.debug["retrieved"])
        st.subheader("Citations")
        st.json(st.session_state.debug["citations"])
