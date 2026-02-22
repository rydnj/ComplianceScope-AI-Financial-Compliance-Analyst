import uuid
from datetime import datetime, date

from sqlalchemy import String, Text, Integer, Float, Date, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector

from app.database import Base


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    filings: Mapped[list["Filing"]] = relationship(back_populates="company", cascade="all, delete-orphan")


class Filing(Base):
    __tablename__ = "filings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    filing_type: Mapped[str] = mapped_column(String(10), nullable=False)
    filing_date: Mapped[date] = mapped_column(Date, nullable=False)
    accession_no: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    # Status tracks ingestion progress: pending → processing → completed / failed
    status: Mapped[str] = mapped_column(String(20), default="pending")
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    company: Mapped["Company"] = relationship(back_populates="filings")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="filing", cascade="all, delete-orphan")
    risk_flags: Mapped[list["RiskFlag"]] = relationship(back_populates="filing", cascade="all, delete-orphan")
    queries: Mapped[list["Query"]] = relationship(back_populates="filing", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filing_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("filings.id", ondelete="CASCADE"))
    section: Mapped[str | None] = mapped_column(String(100))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # 1536-dim vector from text-embedding-3-small
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536))

    filing: Mapped["Filing"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_filing", "filing_id"),
    )


class RiskFlag(Base):
    __tablename__ = "risk_flags"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filing_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("filings.id", ondelete="CASCADE"))
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)  # High, Medium, Low
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chunks.id"))
    detection: Mapped[str] = mapped_column(String(20), nullable=False)  # 'keyword' or 'llm'
    confidence: Mapped[float | None] = mapped_column(Float)  # 0-1 for LLM, NULL for keyword
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    filing: Mapped["Filing"] = relationship(back_populates="risk_flags")

    __table_args__ = (
        Index("idx_risk_flags_filing", "filing_id"),
    )


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filing_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("filings.id"))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[dict | None] = mapped_column(JSONB)  # [{chunk_id, section, excerpt}]
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    filing: Mapped["Filing"] = relationship(back_populates="queries")