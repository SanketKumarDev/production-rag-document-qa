import re
from app.core.models import Document, DocumentChunk


class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 120):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        words = document.text.split()
        chunks = []
        start = 0
        index = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document.document_id}_{index:05d}",
                        document_id=document.document_id,
                        text=chunk_text,
                        metadata={
                            **document.metadata,
                            "page": self._extract_page(chunk_text),
                            "chunk_index": index,
                        },
                    )
                )

            if end == len(words):
                break

            start = end - self.overlap
            index += 1

        return chunks

    @staticmethod
    def _extract_page(text: str) -> int | None:
        match = re.search(r"\[Page (\d+)\]", text)
        return int(match.group(1)) if match else None
