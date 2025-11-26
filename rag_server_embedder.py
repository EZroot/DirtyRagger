import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_server_config import RAGConfig

global SERVER_CONFIG
SERVER_CONFIG = RAGConfig()

class Embedder:
    def __init__(self, model_name: str = SERVER_CONFIG.EMBED_MODEL, device: str = SERVER_CONFIG.DEVICE_EMBEDDER):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Embedder model: {model_name} on device: {self.device}")
        self.model = SentenceTransformer(
            model_name, 
            device=self.device
        )
        self.dim = self.model.get_sentence_embedding_dimension()

    def get_embedding(self, text: str) -> np.ndarray:
        embeddings = self.model.encode([text], convert_to_numpy=True)
        return embeddings[0]

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)