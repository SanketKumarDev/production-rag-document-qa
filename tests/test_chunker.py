from app.core.models import Document
from app.ingestion.chunker import TextChunker


def test_chunker_creates_chunks():
    document = Document(
        document_id="test",
        source="test.txt",
        text=" ".join(["word"] * 1000),
    )
    chunks = TextChunker(chunk_size=100, overlap=20).chunk(document)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "test_00000"
