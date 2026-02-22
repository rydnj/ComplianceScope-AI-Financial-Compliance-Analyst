"""Vector similarity search using pgvector.

This module handles the "retrieval" part of RAG — given a user's question,
find the most relevant chunks from a specific filing.

pgvector supports three distance operators:
  - <-> : L2 (Euclidean) distance
  - <=> : Cosine distance (1 - cosine similarity)
  - <#> : Inner product (negative)

We use cosine distance (<=>)  because it measures the angle between vectors,
ignoring magnitude. This means a short chunk and a long chunk about the same
topic will still be "close" — cosine cares about direction (meaning), not
length. This is the standard choice for text embeddings.
"""

import logging
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.ingestion.embedder import embed_single

logger = logging.getLogger(__name__)


async def retrieve_relevant_chunks(
    db: AsyncSession,
    filing_id: uuid.UUID,
    question: str,
    top_k: int | None = None,
) -> list[dict]:
    """Find the most relevant chunks for a question within a specific filing.
    
    Steps:
      1. Embed the question using the same model as the document chunks
      2. Run a cosine similarity search in pgvector, filtered by filing_id
      3. Return the top-k results with content, section, and similarity score
    
    We filter by filing_id in the WHERE clause rather than using separate
    vector collections per filing. This is efficient because pgvector can
    combine the metadata filter with the vector search in a single query,
    and it avoids the overhead of managing multiple collections.
    
    Args:
        db: Database session
        filing_id: Which filing to search within
        question: The user's natural language question
        top_k: Number of chunks to retrieve (default from config: 5)
        
    Returns:
        List of dicts with keys: chunk_id, section, content, similarity
        Ordered by relevance (most similar first)
    """
    if top_k is None:
        top_k = settings.top_k

    # Step 1: Embed the question with the same model used for chunks
    question_embedding = await embed_single(question)

    # Step 2: Query pgvector for the closest chunks
    # The <=> operator computes cosine distance (0 = identical, 2 = opposite)
    # We subtract from 1 to get cosine similarity (1 = identical, -1 = opposite)
    query = text("""
        SELECT 
            id,
            section,
            content,
            chunk_index,
            1 - (embedding <=> :embedding) AS similarity
        FROM chunks
        WHERE filing_id = :filing_id
        ORDER BY embedding <=> :embedding
        LIMIT :top_k
    """)

    result = await db.execute(
        query,
        {
            "embedding": str(question_embedding),  # pgvector accepts string representation
            "filing_id": str(filing_id),
            "top_k": top_k,
        },
    )

    rows = result.fetchall()

    chunks = [
        {
            "chunk_id": str(row.id),
            "section": row.section,
            "content": row.content,
            "chunk_index": row.chunk_index,
            "similarity": round(row.similarity, 4),
        }
        for row in rows
    ]

    logger.info(
        f"Retrieved {len(chunks)} chunks for question: '{question[:80]}...' "
        f"(similarities: {[c['similarity'] for c in chunks]})"
    )

    return chunks