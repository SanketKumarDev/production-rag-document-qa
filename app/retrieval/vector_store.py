from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.models import DocumentChunk, SearchResult


class VectorStore:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.index = None
        self.chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        embeddings = self.model.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores, indices = self.index.search(
            np.asarray(query_embedding, dtype="float32"),
            min(top_k, len(self.chunks)),
        )
        return [
            SearchResult(self.chunks[int(idx)], float(score), "vector")
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def save(self, directory: Path) -> None:
        faiss.write_index(self.index, str(directory / "faiss.index"))

    def load(self, directory: Path, chunks: list[DocumentChunk]) -> None:
        self.index = faiss.read_index(str(directory / "faiss.index"))
        self.chunks = chunks
