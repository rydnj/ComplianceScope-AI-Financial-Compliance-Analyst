from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables and pgvector extension on startup."""
    await init_db()
    yield


app = FastAPI(
    title="ComplianceScope",
    description="AI-powered SEC filing compliance analyzer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit will connect from a different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Register routers ---
from app.ingestion.router import router as ingestion_router

app.include_router(ingestion_router, prefix="/api")

from app.rag.router import router as rag_router
from app.analysis.router import router as analysis_router
from app.report.router import router as report_router

app.include_router(rag_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(report_router, prefix="/api")