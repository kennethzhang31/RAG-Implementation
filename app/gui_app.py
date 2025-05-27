import streamlit as st

from src.embed import Embedding
from src.parser import Parser
from src.sqlite_vec_manager import SqliteVecManager
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings

import os
from typing import List, Tuple

st.set_page_config(page_title="PDF RAG App", layout="wide")
st.title("📄 PDF RAG Search & Chat")

DB_PATH = "./db/embeddings.db"
UPLOAD_DIR = "./upload_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.read())
    st.sidebar.success(f"Uploaded {len(uploaded_files)} files.")

# Indexing
if st.sidebar.button("Index PDFs"):
    st.write("🔍 Indexing PDFs...")
    parser = Parser()
    embed = Embedding(chunk_size=1500, chunk_overlap=100)
    sqlite_vec_manager = SqliteVecManager(db_path=DB_PATH)

    texts = parser.extract_text(UPLOAD_DIR)
    files, pages, chunk_ids, chunk_texts, embeddings = [], [], [], [], []
    for filename, doc_level in texts:
        for page_no, page_level in doc_level:
            chunks = embed.split_text(page_level)
            chunk_embeddings = embed.embed_text(chunks)
            for idx, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                files.append(filename)
                pages.append(page_no)
                chunk_ids.append(idx)
                chunk_texts.append(chunk_text)
                embeddings.append(embedding)
    sqlite_vec_manager.index_embeddings(files, pages, chunk_ids, chunk_texts, embeddings)
    st.success(f"✅ Indexed {len(embeddings)} chunks.")
    sqlite_vec_manager.close()

# Query
st.header("🔎 Query PDFs")
query = st.text_input("Enter your question:")
top_k = st.number_input("Number of results", min_value=1, max_value=10, value=3)

if st.button("Retrieve Chunks"):
    if query:
        embed = Embedding(chunk_size=1500, chunk_overlap=100)
        sqlite_vec_manager = SqliteVecManager(db_path=DB_PATH)
        query_embedding = embed.embed_text([query])[0]
        results = sqlite_vec_manager.query(query_embedding, top_k=top_k)
        for file, page, score, chunk_text in results:
            similarity = 1 - score
            st.markdown(f"**[{file}] Page {page}** | Cosine Sim: {similarity:.4f}")
            st.code(chunk_text)
        sqlite_vec_manager.close()
    else:
        st.warning("Please enter a query.")

# Chat Mode
if st.button("Retrieve + Chat (LLM Answer)"):
    if query:
        embed = Embedding(chunk_size=1500, chunk_overlap=100)
        sqlite_vec_manager = SqliteVecManager(db_path=DB_PATH)
        query_embedding = embed.embed_text([query])[0]
        results = sqlite_vec_manager.query(query_embedding, top_k=top_k)

        context = "\n\n".join([f"[{file}] Page {page}: {chunk_text}" for file, page, _, chunk_text in results])
        Settings.llm = Ollama(model="gemma3:4b", request_timeout=360.0)
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant. Use the provided context to answer the user's question."),
            ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {query}")
        ]
        response = Settings.llm.chat(messages)
        st.success("LLM Answer:")
        st.markdown(response.message.content.strip())
        sqlite_vec_manager.close()
    else:
        st.warning("Please enter a query.")