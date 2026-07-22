import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from app.core.models import DocumentChunk, SearchResult


class BM25Index:
    def __init__(self):
        self.index = None
        self.chunks: list[DocumentChunk] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.index = BM25Okapi([self.tokenize(c.text) for c in chunks])

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        scores = self.index.get_scores(self.tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            SearchResult(self.chunks[idx], float(score), "bm25")
            for idx, score in ranked
        ]

    def save(self, directory: Path) -> None:
        with (directory / "bm25.pkl").open("wb") as f:
            pickle.dump({"index": self.index, "chunks": self.chunks}, f)

    def load(self, directory: Path) -> None:
        with (directory / "bm25.pkl").open("rb") as f:
            data = pickle.load(f)
        self.index = data["index"]
        self.chunks = data["chunks"]
