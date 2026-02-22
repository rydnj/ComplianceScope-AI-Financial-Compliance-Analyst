"""API endpoints for filing ingestion and company/filing retrieval."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Company, Filing
from app.schemas import IngestRequest, IngestResponse, CompanyResponse, FilingResponse
from app.ingestion.pipeline import run_ingestion

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_filing(request: IngestRequest, db: AsyncSession = Depends(get_db)):
    """Ingest a SEC filing by ticker and filing type.
    
    Fetches the most recent filing of the specified type from EDGAR,
    parses it into sections, chunks the text, generates embeddings,
    and stores everything in the database.
    """
    try:
        filing = await run_ingestion(db, request.ticker, request.filing_type)

        # Fetch the associated company for the response
        result = await db.execute(
            select(Company).where(Company.id == filing.company_id)
        )
        company = result.scalar_one()

        return IngestResponse(
            filing=FilingResponse.model_validate(filing),
            company=CompanyResponse.model_validate(company),
            message=f"Successfully ingested {filing.filing_type} for {company.name} "
                    f"({filing.total_chunks} chunks)",
        )

    except ValueError as e:
        # ValueError from EDGAR client (ticker not found, no filings, etc.)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/companies", response_model=list[CompanyResponse])
async def list_companies(db: AsyncSession = Depends(get_db)):
    """List all companies that have ingested filings."""
    result = await db.execute(select(Company).order_by(Company.ticker))
    companies = result.scalars().all()
    return [CompanyResponse.model_validate(c) for c in companies]


@router.get("/filings/{filing_id}", response_model=FilingResponse)
async def get_filing(filing_id: str, db: AsyncSession = Depends(get_db)):
    """Get details for a specific filing."""
    result = await db.execute(
        select(Filing).where(Filing.id == filing_id)
    )
    filing = result.scalar_one_or_none()
    if not filing:
        raise HTTPException(status_code=404, detail="Filing not found")
    return FilingResponse.model_validate(filing)