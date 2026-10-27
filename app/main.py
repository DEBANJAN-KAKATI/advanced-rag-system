import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import upload, chat, documents, status
from app.core.config import settings
from app.retrieval.vectorstore import vector_store
from app.utils.logging import logger


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting Advanced RAG System backend server...")
    # Build or migrate the vector index in the background so the UI comes up immediately.
    threading.Thread(target=vector_store.ensure_index, name="index-builder", daemon=True).start()
    yield


app = FastAPI(
    title="Advanced RAG System API",
    description="Retrieval-Augmented Generation with hybrid search, reranking, streaming answers and citations.",
    version="2.0.0",
    lifespan=lifespan,
)

# The bundled UI is same-origin and needs no CORS; set CORS_ORIGINS for other front-ends.
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(status.router)

if getattr(sys, "frozen", False):
    application_path = Path(sys._MEIPASS)
else:
    application_path = Path(__file__).resolve().parent.parent

web_dir = application_path / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
