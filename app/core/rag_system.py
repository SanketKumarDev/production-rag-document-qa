import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings
from app.core.models import DocumentChunk
from app.generation.ollama_client import OllamaClient
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import VectorStore


class RAGSystem:
    def __init__(self, index_dir: Path | str = "indexes"):
        self.settings = get_settings()
        self.index_dir = Path(index_dir)

        self.retriever: HybridRetriever | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.llm: OllamaClient | None = None

    def load(self) -> None:
        chunks_path = self.index_dir / "chunks.json"

        chunks_data = json.loads(
            chunks_path.read_text(encoding="utf-8")
        )

        chunks = [
            DocumentChunk(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                text=item["text"],
                metadata=item["metadata"],
            )
            for item in chunks_data
        ]

        embedding_model = SentenceTransformer(
            self.settings.embedding_model
        )

        vector_store = VectorStore(embedding_model)
        vector_store.load(
            self.index_dir,
            chunks,
        )

        bm25 = BM25Index()
        bm25.load(self.index_dir)

        self.retriever = HybridRetriever(
            vector_store,
            bm25,
            self.settings.rrf_k,
        )

        self.reranker = CrossEncoderReranker(
            self.settings.reranker_model
        )

        self.llm = OllamaClient(
            self.settings.ollama_base_url,
            self.settings.ollama_model,
        )

    def is_ready(self) -> bool:
        return (
            self.retriever is not None
            and self.reranker is not None
            and self.llm is not None
        )