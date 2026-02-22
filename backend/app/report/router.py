"""API endpoint for executive report generation."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import ReportResponse
from app.report.generator import generate_report

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/report/{filing_id}", response_model=ReportResponse)
async def create_report(filing_id: str, db: AsyncSession = Depends(get_db)):
    """Generate an executive risk report for a filing.
    
    Requires that risk analysis has already been run on this filing.
    Returns a markdown-formatted executive briefing.
    """
    try:
        report_md = await generate_report(db, uuid.UUID(filing_id))
        return ReportResponse(
            filing_id=uuid.UUID(filing_id),
            report_markdown=report_md,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")