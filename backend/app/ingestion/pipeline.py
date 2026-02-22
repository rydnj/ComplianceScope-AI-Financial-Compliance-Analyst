"""Full ingestion pipeline: ticker → fetch → parse → chunk → embed → store.

This orchestrates all the ingestion steps in order. It's called by the
/api/ingest endpoint and handles creating/updating database records,
error handling, and status tracking.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Company, Filing, Chunk
from app.ingestion.edgar import EdgarClient, FilingMetadata
from app.ingestion.parser import extract_sections
from app.ingestion.chunker import chunk_filing
from app.ingestion.embedder import embed_texts

logger = logging.getLogger(__name__)


async def get_or_create_company(
    db: AsyncSession, ticker: str, company_name: str
) -> Company:
    """Find existing company by ticker or create a new one.
    
    We check for existing companies first to avoid duplicates when
    ingesting multiple filings from the same company.
    """
    result = await db.execute(
        select(Company).where(Company.ticker == ticker.upper())
    )
    company = result.scalar_one_or_none()

    if company:
        logger.info(f"Found existing company: {company.name} ({company.ticker})")
        return company

    company = Company(ticker=ticker.upper(), name=company_name)
    db.add(company)
    await db.flush()  # Assigns the UUID without committing the transaction
    logger.info(f"Created new company: {company.name} ({company.ticker})")
    return company


async def check_existing_filing(db: AsyncSession, accession_no: str) -> Filing | None:
    """Check if we've already ingested this specific filing.
    
    Each filing has a unique accession number. If we've already processed it,
    we skip re-ingestion to avoid duplicate data.
    """
    result = await db.execute(
        select(Filing).where(Filing.accession_no == accession_no)
    )
    return result.scalar_one_or_none()


async def run_ingestion(db: AsyncSession, ticker: str, filing_type: str) -> Filing:
    """Execute the full ingestion pipeline for a single filing.
    
    Steps:
      1. Fetch filing from EDGAR (ticker → CIK → filing HTML)
      2. Parse HTML into sections (Risk Factors, MD&A, etc.)
      3. Chunk sections into ~1000 char pieces with overlap
      4. Embed all chunks via OpenAI
      5. Store everything in PostgreSQL/pgvector
      
    Returns:
        The Filing database record with status 'completed' or 'failed'
    """
    edgar = EdgarClient()

    # --- Step 1: Fetch from EDGAR ---
    logger.info(f"Starting ingestion: {ticker} {filing_type}")
    metadata, html = await edgar.get_latest_filing(ticker, filing_type)
    logger.info(f"Fetched {metadata.filing_type} for {metadata.company_name} ({metadata.filing_date})")

    # Check for duplicate
    existing = await check_existing_filing(db, metadata.accession_no)
    if existing:
        logger.info(f"Filing already ingested: {metadata.accession_no} (status: {existing.status})")
        return existing

    # Create company and filing records
    company = await get_or_create_company(db, ticker, metadata.company_name)

    filing = Filing(
        company_id=company.id,
        filing_type=filing_type,
        filing_date=metadata.filing_date,
        accession_no=metadata.accession_no,
        status="processing",
    )
    db.add(filing)
    await db.flush()  # Get the filing's UUID for linking chunks

    try:
        # --- Step 2: Parse HTML into sections ---
        sections = extract_sections(html, filing_type)
        logger.info(f"Parsed {len(sections)} sections from filing")

        # --- Step 3: Chunk the sections ---
        chunks = chunk_filing(sections)
        logger.info(f"Created {len(chunks)} chunks")

        # --- Step 4: Generate embeddings ---
        # Extract just the text content for embedding
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await embed_texts(chunk_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # --- Step 5: Store chunks + embeddings in pgvector ---
        db_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            db_chunk = Chunk(
                filing_id=filing.id,
                section=chunk.section,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                embedding=embedding,
            )
            db_chunks.append(db_chunk)

        db.add_all(db_chunks)

        # Update filing status
        filing.status = "completed"
        filing.total_chunks = len(db_chunks)
        await db.commit()

        logger.info(
            f"Ingestion complete: {filing.total_chunks} chunks stored "
            f"for {metadata.company_name} {filing_type}"
        )
        return filing

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        filing.status = "failed"
        await db.commit()
        raise