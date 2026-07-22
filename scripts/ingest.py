from pathlib import Path
from app.config.settings import get_settings
from app.ingestion.pipeline import IngestionPipeline


def main():
    settings = get_settings()
    count = IngestionPipeline(settings).run(
        Path("data/documents"),
        Path("indexes"),
    )
    print(f"Indexed {count} chunks.")


if __name__ == "__main__":
    main()
