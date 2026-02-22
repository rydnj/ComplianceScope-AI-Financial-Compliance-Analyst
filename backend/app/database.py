from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.config import settings


engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create pgvector extension and all tables.
    
    We enable the vector extension first since our models depend on the 
    vector column type. Using create_all instead of Alembic for speed —
    in production you'd want migrations for schema evolution.
    
    IMPORTANT: We must import models before calling create_all().
    SQLAlchemy only knows about model classes that inherit from Base AND
    have been imported into Python's memory. Without this import,
    Base.metadata.tables is empty and create_all() does nothing.
    """
    import app.models  # noqa: F401 — triggers model registration with Base.metadata

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """FastAPI dependency that provides a database session per request."""
    async with async_session() as session:
        yield session