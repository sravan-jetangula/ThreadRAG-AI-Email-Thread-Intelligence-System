from __future__ import annotations
<<<<<<< HEAD
import requests
=======
import os
import pandas as pd
>>>>>>> 9a4832465b91164b5e9fb9a3a77c1751ca705da7
import streamlit as st
import sys
import pkg_resources

<<<<<<< HEAD
API_URL = "http://localhost:8000"

def fetch_threads(api_url: str) -> list[dict[str, object]]:
    response = requests.get(f"{api_url}/threads", timeout=10)
    response.raise_for_status()
    return response.json()

def start_session(api_url: str, thread_id: str) -> dict[str, object]:
    response = requests.post(f"{api_url}/start_session", json={"thread_id": thread_id}, timeout=20)
    response.raise_for_status()
    return response.json()

def switch_thread(api_url: str, session_id: str, thread_id: str) -> dict[str, object]:
    response = requests.post(
        f"{api_url}/switch_thread",
        json={"session_id": session_id, "thread_id": thread_id},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()

def ask(api_url: str, session_id: str, text: str, search_outside_thread: bool) -> dict[str, object]:
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

def reset_session(api_url: str, session_id: str | None) -> dict[str, object]:
    response = requests.post(f"{api_url}/reset_session", json={"session_id": session_id}, timeout=20)
    response.raise_for_status()
    return response.json()

=======
installed_packages = [p.key for p in pkg_resources.working_set]
st.write("Installed packages:", installed_packages)

# LangChain + embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.docstore.document import Document

# ----------------------------
# Streamlit Page Config
# ----------------------------
>>>>>>> 9a4832465b91164b5e9fb9a3a77c1751ca705da7
st.set_page_config(page_title="Email + Attachment RAG", layout="wide")
st.title("Email + Attachment RAG with CSV Retrieval")

# ----------------------------
# Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "debug" not in st.session_state:
    st.session_state.debug = None
if "threads" not in st.session_state:
    st.session_state.threads = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ----------------------------
# Load emails CSV
# ----------------------------
@st.cache_data
def load_emails(csv_path="emails.csv"):
    if not os.path.exists(csv_path):
        st.error(f"emails.csv not found in repo!")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Expect CSV to have: thread_id, subject, body
    df.fillna("", inplace=True)
    return df

emails_df = load_emails()

# ----------------------------
# Build Vector Store (FAISS) if not exists
# ----------------------------
@st.cache_resource
def build_vectorstore(df: pd.DataFrame):
    documents = []
    for _, row in df.iterrows():
        content = f"Subject: {row['subject']}\nBody: {row['body']}"
        documents.append(Document(page_content=content, metadata={"thread_id": row["thread_id"]}))
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

if emails_df.shape[0] > 0:
    st.session_state.vector_store = build_vectorstore(emails_df)

# ----------------------------
# Simulated / Dynamic Backend Functions
# ----------------------------
def fetch_threads():
    if emails_df.shape[0] == 0:
        return []
    threads = []
    for tid, group in emails_df.groupby("thread_id"):
        threads.append({
            "thread_id": tid,
            "subject": group.iloc[0]["subject"],
            "message_count": len(group)
        })
    return threads

def start_session(thread_id: str) -> str:
    return f"session_{thread_id}"

def switch_thread(thread_id: str) -> str:
    return f"session_{thread_id}"

def reset_session():
    return None

def ask_rag(thread_id: str, prompt: str, k=3):
    """Use vector store retriever + OpenAI LLM for RAG answer"""
    if st.session_state.vector_store is None:
        return {
            "answer": "Vector store not initialized!",
            "rewrite": prompt,
            "retrieved": [],
            "citations": []
        }
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": {"thread_id": thread_id}}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": prompt})
    docs = result.get("source_documents", [])
    retrieved_docs = [{"doc_id": f"doc{i+1}", "content": doc.page_content} for i, doc in enumerate(docs)]
    citations = [{"source": f"doc{i+1}"} for i in range(len(docs))]
    return {
        "answer": result["result"],
        "rewrite": prompt,
        "retrieved": retrieved_docs,
        "citations": citations
    }

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Session")
    search_outside_thread = st.toggle("Search outside thread", value=False)
    
    if not st.session_state.threads:
        st.session_state.threads = fetch_threads()
    
    labels = {
        item["thread_id"]: f"{item['thread_id']} | {item.get('subject') or 'No subject'} | {item.get('message_count',0)} messages"
        for item in st.session_state.threads
    }
    selected_thread = st.selectbox("Thread selector", list(labels.keys()), format_func=lambda key: labels[key])
    
    if st.button("Start session", disabled=selected_thread is None):
        st.session_state.session_id = start_session(selected_thread)
        st.session_state.messages = []
        st.session_state.debug = None

    if st.button("Switch thread", disabled=selected_thread is None or st.session_state.session_id is None):
        st.session_state.session_id = switch_thread(selected_thread)
        st.session_state.messages = []
        st.session_state.debug = None

    if st.button("Reset session"):
        reset_session()
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.debug = None
    
    st.caption(f"Current session: {st.session_state.session_id or 'None'}")

# ----------------------------
# Display Chat
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# User Input
# ----------------------------
prompt = st.chat_input("Ask about the selected email thread or its attachments")

if prompt:
    if not st.session_state.session_id:
        st.warning("Start a session first!")
    else:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        result = ask_rag(selected_thread, prompt)
        st.session_state.debug = result
        st.session_state.messages.append({"role":"assistant","content":result["answer"]})
        
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
