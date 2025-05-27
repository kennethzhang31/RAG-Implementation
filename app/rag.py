from src.embed import Embedding
from src.parser import Parser
from src.sqlite_vec_manager import SqliteVecManager
import argparse
import sys
from typing import List, Tuple
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from pathlib import Path
import os
from typing import List, Optional, Tuple, Any, Union, BinaryIO

def index_docs(data_dir: str, db_path: Union[str, Path], embed_model: str = "nomic-embed-text"):
    parser = Parser()
    embed = Embedding(chunk_size=1500, chunk_overlap=100, model_name=embed_model)
    sqlite_vec_manager = SqliteVecManager(db_path=db_path)

    print(f"Parsing PDFs in {data_dir}...")
    texts = parser.extract_text(data_dir)

    files, pages, chunk_ids, chunk_texts, embeddings = [], [], [], [], []
    total_pages, total_chunks = 0, 0

    for filename, doc_level in texts:
        for page_no, page_level in doc_level:
            if not page_level.strip():
                continue
            chunks = embed.split_text(page_level)
            if not chunks:
                continue
            chunk_embeddings = embed.embed_text(chunks)
            if len(chunks) != len(chunk_embeddings):
                raise ValueError("Mismatch between chunk count and embedding count!")

            for idx, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                files.append(filename)
                pages.append(page_no)
                chunk_ids.append(idx)
                chunk_texts.append(chunk_text)
                embeddings.append(embedding)

            total_pages += 1
            total_chunks += len(chunks)

    if embeddings:
        sqlite_vec_manager.index_embeddings(files, pages, chunk_ids, chunk_texts, embeddings)
        print(f"Indexed {len(embeddings)} chunks from {len(set(files))} files, {total_pages} pages total.")
    else:
        print("No embeddings were generated. Check your PDF files.")
    
    sqlite_vec_manager.close()

def retrieve_docs(
    question: str, 
    embed_model: str = "nomic-embed-text", 
    top_k: int = 3, 
    chunk_size=1500,
    silent: bool = False
) -> List[Tuple[str, int, float, str]]:
    
    embed = Embedding(chunk_size=chunk_size, chunk_overlap=100, model_name=embed_model)
    sqlite_vec_manager = SqliteVecManager()

    query_embedding = embed.embed_text([question])[0]
    results = sqlite_vec_manager.query(query_embedding, top_k=top_k)

    if not silent:
        print("\nQuery Results:")
        for file, page, score, chunk_text in results:
            similarity = 1 - score  # Convert distance to cosine similarity
            print(f"[{file}] Page {page} | Cosine Sim: {similarity:.4f}")
            print(f"Representative chunk: {chunk_text[:100]}...\n{'-'*60}")

    sqlite_vec_manager.close()
    return results

def chat_docs(results: List[Tuple[str, int, float, str]], query: str, model: str = "gemma3:4b"):
    # Settings.llm = Ollama(model=model, request_timeout=360.0)
    # context = "\n\n".join([f"[{file}] Page {page}: {chunk_text}" for file, page, _, chunk_text in results])
    # prompt = f""" 
    #     You are a helpful assistant. Use the following context to answer the user's question.

    #     Context:
    #     {context}

    #     Question:
    #     {query}

    #     Answer: 
    # """

    # response = Settings.llm.complete(prompt)
    # print("\n LLM Answer:\n")
    # print(response.text.strip())

    Settings.llm = Ollama(model=model, request_timeout=360.0)

    # Prepare context from retrieved chunks
    context = "\n\n".join([
        f"[{file}] Page {page}: {chunk_text}" 
        for file, page, _, chunk_text in results
    ])

    # Build chat messages
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant. Use the provided context to answer the user's question."),
        ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {query}")
    ]

    # Generate response
    response = Settings.llm.chat(messages)
    
    # Output answer
    print("\nLLM Answer:")
    print(response.message.content.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prototype RAG Pipeline for PDF Search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser("index", help="Index PDF documents")
    index_parser.add_argument("data_dir", type=str, help="Path to directory of PDFs")
    index_parser.add_argument("--db_folder", type=str, default="./db", help="Path to the SQLite database folder (default: db)")
    index_parser.add_argument("--db_name", type=str, default="embeddings.db", help="SQLite database name (default: embeddings.db)")

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve relevant chunks for a query")
    retrieve_parser.add_argument("query", type=str, help="Query string for retrieval")
    retrieve_parser.add_argument("--top_k", type=int, default=3, help="Number of top results to return")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Retrieve chunks + (optional) generate answer with LLM")
    chat_parser.add_argument("query", type=str, help="Query string for chat + retrieval")
    chat_parser.add_argument("--top_k", type=int, default=3, help="Number of top results to return")
    args = parser.parse_args()

    if args.command == "index":
        db_path = os.path.join(args.db_folder, args.db_name)
        index_docs(args.data_dir, db_path=db_path)
    elif args.command == "retrieve":
        retrieve_docs(args.query, top_k=args.top_k)
    elif args.command == "chat":
        results = retrieve_docs(args.query, top_k=args.top_k, silent=True)
        chat_docs(results, args.query)
    # else: 
    #     print("[ERROR] Invalid Command")
    #     print("python rag.py [index|retrieve|chat] [data_dir for index | query for retrieve/chat] [--top_k for retrieve/chat]")