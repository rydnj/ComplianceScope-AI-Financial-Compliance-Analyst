import uuid
from datetime import datetime, date
from pydantic import BaseModel, Field


# --- Ingestion ---

class IngestRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Company ticker symbol")
    filing_type: str = Field(..., pattern="^(10-K|10-Q)$", description="Filing type: 10-K or 10-Q")


class CompanyResponse(BaseModel):
    id: uuid.UUID
    ticker: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True


class FilingResponse(BaseModel):
    id: uuid.UUID
    company_id: uuid.UUID
    filing_type: str
    filing_date: date
    accession_no: str
    status: str
    total_chunks: int
    created_at: datetime

    class Config:
        from_attributes = True


class IngestResponse(BaseModel):
    filing: FilingResponse
    company: CompanyResponse
    message: str


# --- RAG Query ---

class QueryRequest(BaseModel):
    filing_id: uuid.UUID
    question: str = Field(..., min_length=1, max_length=1000)


class SourceChunk(BaseModel):
    chunk_id: uuid.UUID
    section: str | None
    excerpt: str


class QueryResponse(BaseModel):
    id: uuid.UUID
    question: str
    answer: str
    sources: list[SourceChunk]
    created_at: datetime


# --- Risk Analysis ---

class RiskFlagResponse(BaseModel):
    id: uuid.UUID
    category: str
    severity: str
    title: str
    description: str
    source_text: str
    detection: str
    confidence: float | None
    created_at: datetime

    class Config:
        from_attributes = True


class RiskSummary(BaseModel):
    total: int
    high: int
    medium: int
    low: int
    by_category: dict[str, int]


# --- Report ---

class ReportResponse(BaseModel):
    filing_id: uuid.UUID
    report_markdown: str