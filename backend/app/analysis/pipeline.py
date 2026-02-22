"""Risk analysis pipeline: keyword detection + LLM classification + deduplication.

Orchestrates the two-tier risk detection system:
  1. Run keyword scan (fast, free, high-precision)
  2. Run LLM classification (slower, costs tokens, catches subtle risks)
  3. Deduplicate: if the same chunk + category is flagged by both methods,
     keep the keyword flag (it's more trustworthy) and skip the LLM duplicate
  4. Store all flags in the database
"""

import logging
import uuid

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Filing, RiskFlag
from app.analysis.keywords import scan_all_chunks
from app.analysis.llm_classifier import classify_all_chunks

logger = logging.getLogger(__name__)


async def _get_filing_chunks(db: AsyncSession, filing_id: uuid.UUID) -> list[dict]:
    """Fetch all chunks for a filing from the database."""
    result = await db.execute(
        select(Chunk).where(Chunk.filing_id == filing_id).order_by(Chunk.chunk_index)
    )
    chunks = result.scalars().all()

    return [
        {
            "id": chunk.id,
            "content": chunk.content,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]


def _deduplicate_flags(keyword_flags, llm_flags) -> tuple[list, list]:
    """Remove LLM flags that duplicate keyword flags on the same chunk + category.
    
    If both keyword and LLM detection flag the same chunk for the same risk
    category, we keep the keyword flag because:
      - It has higher precision (exact pattern match)
      - It's free (no token cost)
      - The LLM flag would be redundant
    
    Returns:
        Tuple of (keyword_flags, filtered_llm_flags) — keyword flags unchanged,
        LLM flags with duplicates removed
    """
    # Build a set of (chunk_id, category) pairs from keyword flags
    keyword_pairs = {
        (str(flag.chunk_id), flag.category) for flag in keyword_flags
    }

    # Filter out LLM flags that overlap
    filtered_llm = []
    duplicates_removed = 0
    for flag in llm_flags:
        pair = (str(flag.chunk_id), flag.category)
        if pair in keyword_pairs:
            duplicates_removed += 1
        else:
            filtered_llm.append(flag)

    if duplicates_removed > 0:
        logger.info(f"Deduplication: removed {duplicates_removed} LLM flags that overlapped with keyword flags")

    return keyword_flags, filtered_llm


async def run_risk_analysis(db: AsyncSession, filing_id: uuid.UUID) -> int:
    """Execute the full risk analysis pipeline for a filing.
    
    Returns:
        Total number of risk flags stored
    """
    # Verify filing exists and is completed
    result = await db.execute(select(Filing).where(Filing.id == filing_id))
    filing = result.scalar_one_or_none()
    if not filing:
        raise ValueError(f"Filing {filing_id} not found")
    if filing.status != "completed":
        raise ValueError(f"Filing {filing_id} is not ready for analysis (status: {filing.status})")

    # Clear any previous risk flags for this filing (allows re-analysis)
    await db.execute(delete(RiskFlag).where(RiskFlag.filing_id == filing_id))

    # Get all chunks
    chunks = await _get_filing_chunks(db, filing_id)
    logger.info(f"Running risk analysis on {len(chunks)} chunks for filing {filing_id}")

    # --- Tier 1: Keyword Detection ---
    keyword_flags = scan_all_chunks(chunks)
    logger.info(f"Keyword detection: {len(keyword_flags)} flags")

    # --- Tier 2: LLM Classification ---
    llm_flags = await classify_all_chunks(chunks)
    logger.info(f"LLM classification: {len(llm_flags)} flags")

    # --- Deduplicate ---
    keyword_flags, llm_flags = _deduplicate_flags(keyword_flags, llm_flags)

    # --- Store all flags ---
    db_flags = []

    for flag in keyword_flags:
        db_flags.append(RiskFlag(
            filing_id=filing_id,
            category=flag.category,
            severity=flag.severity,
            title=flag.title,
            description=flag.description,
            source_text=flag.source_text,
            chunk_id=flag.chunk_id,
            detection="keyword",
            confidence=None,  # Keywords don't have confidence scores
        ))

    for flag in llm_flags:
        db_flags.append(RiskFlag(
            filing_id=filing_id,
            category=flag.category,
            severity=flag.severity,
            title=flag.title,
            description=flag.description,
            source_text=flag.source_text,
            chunk_id=flag.chunk_id,
            detection="llm",
            confidence=flag.confidence,
        ))

    db.add_all(db_flags)
    await db.commit()

    logger.info(
        f"Risk analysis complete: {len(db_flags)} total flags "
        f"({len(keyword_flags)} keyword + {len(llm_flags)} LLM)"
    )

    return len(db_flags)