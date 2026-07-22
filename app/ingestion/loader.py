from pathlib import Path
import fitz
from app.core.models import Document


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def load(self, path: Path) -> Document:
        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        if suffix == ".pdf":
            text, pages = self._load_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            pages = []

        return Document(
            document_id=path.stem,
            source=path.name,
            text=text,
            metadata={"file_type": suffix, "file_name": path.name, "pages": pages},
        )

    @staticmethod
    def _load_pdf(path: Path) -> tuple[str, list[dict]]:
        sections = []
        pages = []

        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if text:
                    sections.append(f"[Page {page_number}]\n{text}")
                    pages.append({"page": page_number})

        return "\n\n".join(sections), pages

    def load_directory(self, directory: Path) -> list[Document]:
        return [
            self.load(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
