"""RAG chain: combines retrieved context with LLM to generate grounded answers.

The prompt engineering here is critical for answer quality:
  - System prompt establishes the role and constraints
  - Context is formatted with section labels so the LLM can cite sources
  - We explicitly instruct the model to say "I don't know" rather than guess
    when the context doesn't contain the answer (reducing hallucination)

"Grounding" means the LLM only uses information from the provided context,
not its general training knowledge. This is what makes RAG reliable for
domain-specific questions — the answer is always traceable to source text.
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

logger = logging.getLogger(__name__)


# --- Prompt Template ---
# Stored as a constant rather than inline for maintainability.
# Each part of the prompt serves a specific purpose:

SYSTEM_PROMPT = """You are a financial compliance analyst reviewing SEC filings.
Your job is to answer questions accurately based ONLY on the provided context 
from the filing. Follow these rules strictly:

1. Only use information from the provided context sections below.
2. Cite which section(s) your answer comes from (e.g., "According to the Risk Factors section...").
3. If the context does not contain enough information to answer the question, 
   say "Based on the available filing context, I cannot fully answer this question."
4. Be specific and quote relevant details from the filing when possible.
5. Keep your answers professional and concise."""

USER_PROMPT_TEMPLATE = """Context from SEC filing (retrieved sections):
{context}

Question: {question}

Provide a thorough answer based on the context above, citing specific sections."""


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a labeled context string for the prompt.
    
    Each chunk gets a header with its section name and index so the LLM
    can reference specific sections in its answer. This is how we enable
    source citations in the response.
    """
    formatted_parts = []
    for i, chunk in enumerate(chunks, 1):
        formatted_parts.append(
            f"[Section: {chunk['section']} | Chunk {chunk['chunk_index']}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(formatted_parts)


# Initialize the LLM once at module level
_llm = ChatOpenAI(
    model=settings.llm_model,           # gpt-4o-mini
    temperature=settings.llm_temperature, # 0.0 — deterministic for compliance work
    api_key=settings.openai_api_key,
)

# Build the prompt template using LangChain
_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT_TEMPLATE),
])

# Chain the prompt template with the LLM
# When invoked, this: formats the prompt → sends to OpenAI → returns response
_chain = _prompt | _llm


async def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate a grounded answer using retrieved context and GPT-4o-mini.
    
    Args:
        question: The user's natural language question
        chunks: Retrieved chunks from the vector search (retriever.py)
        
    Returns:
        The LLM's answer string, grounded in the provided context
    """
    if not chunks:
        return "No relevant context was found in the filing to answer this question."

    context = _format_context(chunks)

    logger.info(f"Generating answer with {len(chunks)} context chunks")

    # ainvoke = async invoke. Sends the formatted prompt to OpenAI.
    response = await _chain.ainvoke({
        "context": context,
        "question": question,
    })

    # LangChain returns an AIMessage object; .content has the text
    return response.content