"""API endpoints for RAG query and query history."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Filing, Query
from app.schemas import QueryRequest, QueryResponse, SourceChunk
from app.rag.retriever import retrieve_relevant_chunks
from app.rag.chain import generate_answer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_filing(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    """Ask a natural language question about an ingested filing.
    
    Pipeline:
      1. Verify the filing exists and is completed
      2. Retrieve the top-k most relevant chunks via vector similarity
      3. Feed chunks + question to GPT-4o-mini for a grounded answer
      4. Store the query and sources in the database
      5. Return answer with source citations
    """
    # Verify the filing exists and was successfully ingested
    result = await db.execute(
        select(Filing).where(Filing.id == request.filing_id)
    )
    filing = result.scalar_one_or_none()

    if not filing:
        raise HTTPException(status_code=404, detail="Filing not found")
    if filing.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Filing is not ready for queries (status: {filing.status})"
        )

    try:
        # Step 1: Retrieve relevant chunks
        chunks = await retrieve_relevant_chunks(db, request.filing_id, request.question)

        # Step 2: Generate answer from context
        answer = await generate_answer(request.question, chunks)

        # Step 3: Build source citations
        sources = [
            SourceChunk(
                chunk_id=uuid.UUID(chunk["chunk_id"]),
                section=chunk["section"],
                # Show first 200 chars of each source chunk as an excerpt
                excerpt=chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
            )
            for chunk in chunks
        ]

        # Step 4: Store the query in the database for history
        db_query = Query(
            filing_id=request.filing_id,
            question=request.question,
            answer=answer,
            sources=[
                {
                    "chunk_id": str(s.chunk_id),
                    "section": s.section,
                    "excerpt": s.excerpt,
                }
                for s in sources
            ],
        )
        db.add(db_query)
        await db.commit()
        await db.refresh(db_query)

        return QueryResponse(
            id=db_query.id,
            question=request.question,
            answer=answer,
            sources=sources,
            created_at=db_query.created_at,
        )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/queries/{filing_id}", response_model=list[QueryResponse])
async def get_query_history(filing_id: str, db: AsyncSession = Depends(get_db)):
    """Get all previous queries for a specific filing."""
    result = await db.execute(
        select(Query)
        .where(Query.filing_id == filing_id)
        .order_by(Query.created_at.desc())
    )
    queries = result.scalars().all()

    return [
        QueryResponse(
            id=q.id,
            question=q.question,
            answer=q.answer,
            sources=[
                SourceChunk(
                    chunk_id=uuid.UUID(s["chunk_id"]),
                    section=s["section"],
                    excerpt=s["excerpt"],
                )
                for s in (q.sources or [])
            ],
            created_at=q.created_at,
        )
        for q in queries
    ]