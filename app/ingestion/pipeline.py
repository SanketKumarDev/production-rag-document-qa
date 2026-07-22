import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.config.settings import Settings
from app.core.models import DocumentChunk
from app.ingestion.cleaner import TextCleaner
from app.ingestion.chunker import TextChunker
from app.ingestion.loader import DocumentLoader
from app.retrieval.bm25 import BM25Index
from app.retrieval.vector_store import VectorStore


class IngestionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)

    def run(self, input_dir: Path, index_dir: Path) -> int:
        index_dir.mkdir(parents=True, exist_ok=True)
        documents = self.loader.load_directory(input_dir)
        chunks: list[DocumentChunk] = []

        for document in documents:
            cleaned = self.cleaner.clean(document.text)
            document = document.__class__(
                document_id=document.document_id,
                source=document.source,
                text=cleaned,
                metadata=document.metadata,
            )
            chunks.extend(self.chunker.chunk(document))

        if not chunks:
            raise ValueError("No supported documents found.")

        model = SentenceTransformer(self.settings.embedding_model)

        vector_store = VectorStore(model)
        vector_store.build(chunks)
        vector_store.save(index_dir)

        bm25 = BM25Index()
        bm25.build(chunks)
        bm25.save(index_dir)

        (index_dir / "chunks.json").write_text(
            json.dumps(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "text": c.text,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        (index_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "documents": len(documents),
                    "chunks": len(chunks),
                    "embedding_model": self.settings.embedding_model,
                    "reranker_model": self.settings.reranker_model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return len(chunks)
