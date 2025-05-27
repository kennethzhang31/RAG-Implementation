# 📚 RAG Pipeline for PDF Search & Chat

This project implements a **Retrieval-Augmented Generation (RAG)** system to index, retrieve, and chat over PDF documents. It features a **CLI** interface for indexing and querying, and a **Streamlit app** for interactive chat.

## 🚀 Features

- ✅ PDF Parsing with `Docling`
- ✅ Text Chunking + Embedding with `llama-index` + `Ollama`
- ✅ Fast Vector Search with `sqlite-vec`
- ✅ MaxSim retrieval for best page-level answers
- ✅ Query and Chat over your PDFs (CLI + Streamlit GUI)

---

## 📂 Project Structure
```
rag_implementation/
│
├── app/
│   ├── src/
│   │   ├── embed.py                 # Embedding pipeline
│   │   ├── parser.py                # PDF text extraction
│   │   ├── sqlite_vec_manager.py    # Vector DB management
│   │
│   ├── docs/                        # Folder for PDF documents
│   ├── db/                          # SQLite-Vec DB (embeddings.db)
│   ├── upload_docs/                 # Upload folder (for Streamlit)
│   │
│   ├── gui_app.py                   # Streamlit frontend app
│   ├── rag.py                       # CLI: index/retrieve/chat
│   ├── requirements.txt             # Python dependencies
```

## ⚙️ Setup

1️⃣ Install dependencies (in your virtualenv/conda):

```bash
pip install -r app/requirements.txt
```

2️⃣ Ensure sqlite-vec is properly installed

3️⃣ Set up Ollama (for embeddings + LLM generation):


## 🛠️ CLI Usage
``` bash
cd app

# Index PDFs
python rag.py index path/to/your/docs

# Retrieve chunks for a query
python rag.py retrieve "What is the company registration date?" --top_k 3

# Retrieve + Chat with LLM
python rag.py chat "What products does the company manufacture?" --top_k 3
```

## 🖥️ Run Streamlit App

```bash
streamlit run app/gui_app.py
```

This will launch an interactive web UI for PDF search + chat!


## 🗃️ Managing the DB

Inside src/sqlite_vec_manager.py:

✅ clear_db() – Delete all embeddings
✅ delete_file(file_name) – Delete all chunks from a specific file

You can also expose these via CLI if needed.
