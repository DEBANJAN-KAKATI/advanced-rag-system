from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys
import os
from app.api import upload, chat, documents
from app.utils.logging import logger

app = FastAPI(
    title="Advanced RAG System API",
    description="Enterprise-grade Retrieval-Augmented Generation with Gemini, Hybrid Retrieval, Reranking, and Citations.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(documents.router)

# Mount web frontend
if getattr(sys, 'frozen', False):
    application_path = Path(sys._MEIPASS)
else:
    application_path = Path(__file__).resolve().parent.parent

web_dir = application_path / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Advanced RAG System backend server...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
