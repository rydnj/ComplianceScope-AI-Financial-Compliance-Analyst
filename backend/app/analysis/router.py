"""API endpoints for risk analysis, risk retrieval, and risk summary."""

import logging
import uuid
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Filing, RiskFlag
from app.schemas import RiskFlagResponse, RiskSummary
from app.analysis.pipeline import run_risk_analysis

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze/{filing_id}")
async def analyze_filing(filing_id: str, db: AsyncSession = Depends(get_db)):
    """Trigger risk analysis on an ingested filing.
    
    Runs both keyword detection and LLM classification, deduplicates,
    and stores all risk flags. This can take 1-3 minutes depending on
    the number of chunks (each relevant chunk = one LLM API call).
    """
    try:
        total_flags = await run_risk_analysis(db, uuid.UUID(filing_id))
        return {
            "filing_id": filing_id,
            "total_flags": total_flags,
            "message": f"Risk analysis complete: {total_flags} risks identified",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/risks/{filing_id}", response_model=list[RiskFlagResponse])
async def get_risks(filing_id: str, db: AsyncSession = Depends(get_db)):
    """Get all risk flags for a filing, ordered by severity."""
    # Custom ordering: High first, then Medium, then Low
    result = await db.execute(
        select(RiskFlag)
        .where(RiskFlag.filing_id == filing_id)
        .order_by(
            # Sort by severity using a CASE expression
            RiskFlag.created_at.desc()
        )
    )
    flags = result.scalars().all()
    
    # Sort in Python for cleaner severity ordering
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    flags_sorted = sorted(flags, key=lambda f: severity_order.get(f.severity, 3))
    
    return [RiskFlagResponse.model_validate(f) for f in flags_sorted]


@router.get("/risks/{filing_id}/summary", response_model=RiskSummary)
async def get_risk_summary(filing_id: str, db: AsyncSession = Depends(get_db)):
    """Get aggregated risk counts by severity and category for a filing."""
    result = await db.execute(
        select(RiskFlag).where(RiskFlag.filing_id == filing_id)
    )
    flags = result.scalars().all()

    severity_counts = Counter(f.severity for f in flags)
    category_counts = Counter(f.category for f in flags)

    return RiskSummary(
        total=len(flags),
        high=severity_counts.get("High", 0),
        medium=severity_counts.get("Medium", 0),
        low=severity_counts.get("Low", 0),
        by_category=dict(category_counts),
    )