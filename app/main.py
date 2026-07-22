from fastapi import FastAPI
from app.api.routes import router
from app.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Production-style hybrid RAG API.",
)

app.include_router(router)
