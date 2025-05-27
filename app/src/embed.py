from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from typing import List
import torch
import asyncio

class Embedding:
    def __init__(self, chunk_size: int, chunk_overlap: int=50, model_name: str="nomic-embed-text"):
        """
        Initialize the embedding model.

        Args:
            model_name (str): The name of the model to use.
            chunk_size (int): The size of the chunks to use.
            chunk_overlap (int): The overlap of the chunks to use.
        """
        self.embed_model = OllamaEmbedding(
            model_name=model_name,
            ollama_additional_kwargs={"mirostat": 0},
            # base_url="http://localhost:11434",
            device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_text(self, text: str) -> List[str]:
        """
        Split the text into chunks.

        Args:
            text (str): The text to split.
        """
        return self.text_splitter.split_text(text)

    def embed_text(self, texts: List[str]) -> List[float]:
        """
        Embed the text.

        Args:
            texts (List[str]): The text to embed.
        """
        return self.embed_model.get_text_embedding_batch(texts)

    async def embed_text_async(self, texts: List[str]) -> List[float]:
        """
        Embed the text.

        Args:
            texts (List[str]): The text to embed.
        """
        return await asyncio.to_thread(self.embed_model.get_text_embedding_batch, texts)
        
if __name__ == "__main__":
    embed = Embedding(chunk_size=1500)
    texts = ["Hello, world!", "This is a test."]
    print(len(embed.embed_text(texts)[1]))