"""OpenAI embedding generation for text chunks.

Embeddings convert text into dense numerical vectors (1536 dimensions for
text-embedding-3-small). The key insight: texts with similar *meaning* end
up close together in vector space, even if they use completely different words.

For example, "the company faces significant litigation risk" and 
"ongoing lawsuits threaten the firm" would have very similar embeddings,
even though they share almost no words. This is what makes vector search
more powerful than keyword search for our RAG pipeline.

Cost: text-embedding-3-small is ~$0.02 per 1M tokens. A full 10-K filing
with ~50K tokens costs about $0.001 to embed — essentially free.
"""

import logging

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize the async OpenAI client once at module level
# It reads OPENAI_API_KEY from the environment automatically,
# but we pass it explicitly for clarity
_client = AsyncOpenAI(api_key=settings.openai_api_key)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts.
    
    OpenAI's embedding API accepts batches, which is much faster than
    sending one text at a time. We can send up to ~8000 tokens per batch
    item and thousands of items per request.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors (each is 1536 floats)
        Order matches the input texts — texts[0]'s embedding is result[0]
    """
    if not texts:
        return []

    # Process in batches of 100 to avoid hitting API limits on request size
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(f"Embedding batch {i // batch_size + 1}: {len(batch)} texts")

        response = await _client.embeddings.create(
            model=settings.embedding_model,  # "text-embedding-3-small"
            input=batch,
        )

        # Response contains an array of embedding objects, each with an .embedding field
        # They come back in the same order as the input
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    logger.info(f"Generated {len(all_embeddings)} embeddings ({settings.embedding_dimensions} dimensions each)")
    return all_embeddings


async def embed_single(text: str) -> list[float]:
    """Generate an embedding for a single text. Used for query embedding.
    
    When a user asks a question, we embed their question with the same model
    and find the closest chunk embeddings via cosine similarity. Using the
    SAME model for both documents and queries is critical — different models
    produce incompatible vector spaces.
    """
    result = await embed_texts([text])
    return result[0]