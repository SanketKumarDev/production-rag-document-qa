from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config.settings import get_settings
from app.core.rag_system import RAGSystem

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag = RAGSystem()
    rag.load()

    app.state.rag = rag

    yield

    app.state.rag = None


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Production-style hybrid RAG API.",
    lifespan=lifespan,
)

app.include_router(router)