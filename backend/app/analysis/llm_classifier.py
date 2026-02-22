"""LLM-based risk classification (Tier 2).

Sends chunks to GPT-4o-mini with a structured output schema to detect
risks that keyword matching misses. The LLM can catch:
  - Indirect references ("areas requiring improvement" → Material Weakness)
  - Contextual risks (discussing supply chain + China → geopolitical risk)
  - Implied risks that require reading comprehension

We use Pydantic models to enforce structured output — the LLM must return
valid JSON matching our schema, which prevents the biggest source of bugs
in LLM integration (trying to parse free-text responses).

Only flags with confidence >= 0.7 are kept to reduce false positives.
"""

import logging
import uuid
import json
from dataclasses import dataclass

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

logger = logging.getLogger(__name__)


# --- Structured Output Schema ---
# These Pydantic models define the exact JSON shape the LLM must return.
# LangChain's with_structured_output() enforces this via OpenAI's function calling.

VALID_CATEGORIES = [
    "Regulatory Action",
    "Material Weakness",
    "Going Concern",
    "Related Party Transactions",
    "Litigation",
    "Revenue Recognition",
    "Cybersecurity",
]


class RiskItem(BaseModel):
    """A single risk identified by the LLM."""
    category: str = Field(description=f"One of: {', '.join(VALID_CATEGORIES)}")
    severity: str = Field(description="One of: High, Medium, Low")
    title: str = Field(description="Short descriptive title for this risk (max 20 words)")
    description: str = Field(description="Explanation of why this is a risk (2-3 sentences)")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")


class RiskAnalysisResult(BaseModel):
    """Container for all risks found in a chunk."""
    risks: list[RiskItem] = Field(default_factory=list, description="List of identified risks. Empty list if no risks found.")


@dataclass
class LLMRiskFlag:
    """A risk detected by LLM classification."""
    category: str
    severity: str
    title: str
    description: str
    source_text: str
    chunk_id: uuid.UUID
    confidence: float


# --- LLM Setup ---

SYSTEM_PROMPT = """You are a financial compliance risk analyst reviewing SEC filing excerpts.
Your job is to identify specific compliance risks in the provided text.

Risk categories you should look for:
- Regulatory Action: SEC investigations, enforcement actions, regulatory penalties
- Material Weakness: Internal control deficiencies, audit failures
- Going Concern: Doubts about continued operation, liquidity crises
- Related Party Transactions: Insider dealings, self-dealing, conflicts of interest
- Litigation: Lawsuits, legal proceedings, settlement risks
- Revenue Recognition: Accounting irregularities, restatements, revenue manipulation
- Cybersecurity: Data breaches, system vulnerabilities, cyber incidents

Rules:
1. Only flag genuine compliance risks, not general business challenges.
2. A chunk discussing competitive pressure or market conditions is NOT a compliance risk.
3. Set confidence based on how explicitly the risk is stated (direct mention = 0.9+, implied = 0.6-0.8).
4. If no compliance risks are present, return an empty risks list.
5. Be conservative — false negatives are better than false positives in compliance."""

USER_PROMPT = """Analyze the following SEC filing excerpt for compliance risks:

Section: {section}
Text: {content}

Identify any compliance risks present. Return an empty list if none are found."""

_llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0.0,
    api_key=settings.openai_api_key,
)

# with_structured_output tells LangChain to use OpenAI's function calling
# to force the response into our Pydantic schema. This is much more reliable
# than asking the LLM to output JSON and parsing it ourselves.
_structured_llm = _llm.with_structured_output(RiskAnalysisResult)

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
])

_chain = _prompt | _structured_llm


async def classify_chunk(
    chunk_id: uuid.UUID, content: str, section: str
) -> list[LLMRiskFlag]:
    """Classify a single chunk for compliance risks using GPT-4o-mini.
    
    Args:
        chunk_id: Database ID of the chunk
        content: The chunk's text content
        section: Which filing section this came from
        
    Returns:
        List of LLMRiskFlag objects for risks above the confidence threshold
    """
    try:
        result: RiskAnalysisResult = await _chain.ainvoke({
            "section": section,
            "content": content,
        })
    except Exception as e:
        logger.warning(f"LLM classification failed for chunk {chunk_id}: {e}")
        return []

    flags = []
    for risk in result.risks:
        # Only keep risks above our confidence threshold
        if risk.confidence < settings.risk_confidence_threshold:
            logger.debug(
                f"Skipping low-confidence risk: {risk.title} ({risk.confidence})"
            )
            continue

        # Validate category
        if risk.category not in VALID_CATEGORIES:
            logger.warning(f"LLM returned invalid category: {risk.category}")
            continue

        flags.append(LLMRiskFlag(
            category=risk.category,
            severity=risk.severity,
            title=risk.title,
            description=risk.description,
            source_text=content[:300],  # Store the beginning of the chunk as source
            chunk_id=chunk_id,
            confidence=risk.confidence,
        ))

    return flags


async def classify_all_chunks(
    chunks: list[dict], batch_size: int = 5
) -> list[LLMRiskFlag]:
    """Classify all chunks from a filing using the LLM.
    
    We process chunks sequentially rather than in parallel to avoid
    hitting OpenAI rate limits. Each chunk is one API call.
    
    We only classify Risk Factors and Legal Proceedings sections —
    Financial Statements chunks rarely contain the narrative text
    that indicates compliance risks.
    
    Args:
        chunks: List of dicts with keys: id, content, section
        batch_size: Not used for parallelism, reserved for future batching
        
    Returns:
        All LLM risk flags above the confidence threshold
    """
    # Filter to sections most likely to contain compliance risks
    relevant_sections = {"Risk Factors", "Legal Proceedings"}
    relevant_chunks = [c for c in chunks if c["section"] in relevant_sections]

    logger.info(
        f"LLM classification: {len(relevant_chunks)} relevant chunks "
        f"(of {len(chunks)} total)"
    )

    all_flags = []
    for i, chunk in enumerate(relevant_chunks):
        logger.info(f"Classifying chunk {i + 1}/{len(relevant_chunks)}")
        flags = await classify_chunk(
            chunk_id=chunk["id"],
            content=chunk["content"],
            section=chunk["section"],
        )
        all_flags.extend(flags)

    logger.info(f"LLM classification complete: {len(all_flags)} flags from {len(relevant_chunks)} chunks")
    return all_flags