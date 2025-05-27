import sqlite3
import sqlite_vec
from typing import List, Tuple, Optional, Dict
import os
import numpy as np
import json
from collections import defaultdict
from pathlib import Path
import argparse

class SqliteVecManager:
    def __init__(self, db_path: str = "./db/embeddings.db", table_name: str = "vec_embeddings"):
        self.db_path = Path(db_path).resolve() 
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.table_name = table_name
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._load_sqlite_vec()
        self.table_created = self._check_table_exists()

    def _check_table_exists(self) -> bool:
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name=?;
        """, (self.table_name,))
        return cursor.fetchone() is not None
        
    def _load_sqlite_vec(self):
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def create_table(self, dim: int):
        if self.table_created:
            return
        
        create_sql = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name} USING vec0(
            file TEXT,
            page INT,
            chunk_id INT,
            chunk_text TEXT,
            embedding FLOAT[{dim}]
        );
        """
        self.conn.execute(create_sql)
        self.conn.commit()
        self.table_created = True
        print(f"Created table '{self.table_name}' with embedding dim: {dim}")

    def index_embeddings(self, 
                         files: List[str], 
                         pages: List[int], 
                         chunk_ids: List[int],
                         chunk_texts: List[str], 
                         embeddings: List[List[float]]):
        """
        Index embeddings and their metadata into the SQLite-Vec database.
        """
        if not (len(files) == len(pages) == len(chunk_ids) == len(chunk_texts) == len(embeddings)):
            raise ValueError("All input lists must have the same length.")

        if not embeddings:
            print("No embeddings to index.")
            return

        dim = len(embeddings[0])

        if not self.table_created:
            self.create_table(dim)

        #for file, page, chunk_id, chunk_text, embedding in zip(files, pages, chunk_ids, chunk_texts, embeddings):
        #    if len(embedding) != dim:
        #        raise ValueError(f"Embedding dimension mismatch: Expected {dim}, got {len(embedding)}")
        #    self.conn.execute(
        #        f"INSERT INTO {self.table_name} (file, page, chunk_id, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
        #       (file, page, chunk_id, chunk_text, embedding)
        #    )
        for file, page, chunk_id, chunk_text, embedding in zip(files, pages, chunk_ids, chunk_texts, embeddings):
            if len(embedding) != dim:
                raise ValueError(f"Embedding dimension mismatch: Expected {dim}, got {len(embedding)}")
            self.conn.execute(
                f"INSERT INTO {self.table_name} (file, page, chunk_id, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
                (file, page, chunk_id, chunk_text, json.dumps(embedding))
            )
        self.conn.commit()
        print(f"Indexed {len(embeddings)} embeddings.")

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, int, int, str, float]]:
        # query_vec = np.array(query_embedding)
        # query_norm = np.linalg.norm(query_vec)
        # cursor = self.conn.execute(f"SELECT file, page, chunk_id, chunk_text, embedding FROM {self.table_name}")
        # page_scores: Dict[Tuple[str, int], Tuple[float, str]] = {}
        # for file, page, chunk_id, chunk_text, chunk_embedding in cursor.fetchall():
        #     chunk_vec = np.array(chunk_embedding)
        #     chunk_norm = np.linalg.norm(chunk_vec)

        #     if chunk_norm == 0 or query_norm == 0:
        #         cos_sim = 0.0
        #     else:
        #         cos_sim = np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm)

        #     key = (file, page)
        #     if key not in page_scores or cos_sim > page_scores[key][0]:
        #         page_scores[key] = (cos_sim, chunk_text)

        # sorted_pages = sorted(page_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

        # return [(file, page, score, chunk_text) for (file, page), (score, chunk_text) in sorted_pages]
        query_json = json.dumps(query_embedding)

        # Fetch all chunk matches (up to a high limit)
        sql = f"""
        SELECT file, page, chunk_id, chunk_text, distance FROM {self.table_name}
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT 100
        """
        cursor = self.conn.execute(sql, (query_json,))
        
        page_scores = defaultdict(lambda: (float('inf'), ""))  # (min_distance, chunk_text)

        for file, page, chunk_id, chunk_text, distance in cursor.fetchall():
            if distance < page_scores[(file, page)][0]:
                page_scores[(file, page)] = (distance, chunk_text)
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1][0])[:top_k]
        return [(file, page, 1 - dist, chunk_text) for (file, page), (dist, chunk_text) in sorted_pages]
    
    def clear_db(self):
        """
            Delete all data from the table
        """
        if not self.table_created:
            print("Table does not exist yet. Nothing to clear.")
            return
        self.conn.execute(f"DELETE FROM {self.table_name};")
        self.conn.commit()
        print(f"Cleared all data from table '{self.table_name}'.")
    
    def delete_file(self, file_name: str):
        """Delete all chunks from a specific file."""
        if not self.table_created:
            print("Table does not exist yet. Nothing to delete.")
            return
        self.conn.execute(f"DELETE FROM {self.table_name} WHERE file = ?;", (file_name,))
        self.conn.commit()
        print(f"Deleted all chunks from file: {file_name}")

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def main():
    parser = argparse.ArgumentParser(description="Manage your SQLite-Vec database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Clear DB command
    clear_parser = subparsers.add_parser("clear", help="Delete all entries from the database")

    # Delete file command
    delete_parser = subparsers.add_parser("delete", help="Delete all entries for a specific file")
    delete_parser.add_argument("file_name", type=str, help="The file name to delete")

    # Query test command (optional for testing)
    query_parser = subparsers.add_parser("query", help="Test a query with a dummy embedding (for testing only)")
    query_parser.add_argument("--top_k", type=int, default=3, help="Top K results to fetch")

    args = parser.parse_args()
    vec_manager = SqliteVecManager()

    if args.command == "clear":
        vec_manager.clear_db()
    elif args.command == "delete":
        vec_manager.delete_file(args.file_name)
    elif args.command == "query":
        # Dummy query (replace with actual embeddings if you want)
        dummy_embedding = [0.1] * 768  # Adjust the dimension as per your model
        results = vec_manager.query(dummy_embedding, top_k=args.top_k)
        for file, page, sim, chunk_text in results:
            print(f"[{file}] Page {page} | Similarity: {sim:.4f}")
            print(f"Chunk: {chunk_text[:100]}\n{'-'*40}")
    else:
        parser.print_help()

    vec_manager.close()

if __name__ == "__main__":
    main()