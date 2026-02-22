"""Executive risk report generation.

Takes all risk flags for a filing and synthesizes them into a narrative
executive briefing using GPT-4o-mini. The report groups risks by category,
highlights the most critical findings, and suggests follow-up actions.

This is a single LLM call with a well-structured prompt — the quality
of the report depends heavily on how we format the risk data in the context.
"""

import logging
import uuid
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Filing, Company, RiskFlag

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior financial compliance analyst writing an executive risk briefing.
Your audience is a compliance director or VP who needs a clear, actionable summary.

Structure your report EXACTLY as follows:

# Executive Risk Summary: {company_name} {filing_type} ({filing_date})

## Overview
A 2-3 sentence high-level summary of the risk landscape. State the total number 
of risks identified, the breakdown by severity, and the most critical areas.

## Critical Findings
The top 3-5 most important risks that require immediate attention. For each:
- What the risk is
- Why it matters
- What section of the filing it was found in

## Risks by Category
For each category that has flags, provide a brief paragraph summarizing the 
key risks. Group related flags together rather than listing each one individually.

## Recommended Actions
3-5 specific, actionable follow-up steps the compliance team should take 
based on the findings.

## Methodology Note
One sentence noting that this analysis used a hybrid keyword + LLM detection 
approach, with the number of flags from each method.

Guidelines:
- Be concise and professional
- Focus on what matters most, not exhaustive listing
- Use specific details from the risk flags provided
- Do not invent risks that aren't in the data"""


USER_PROMPT = """Generate an executive risk briefing for the following filing:

Company: {company_name}
Ticker: {ticker}
Filing Type: {filing_type}
Filing Date: {filing_date}

Risk Summary:
- Total risks: {total_risks}
- High severity: {high_count}
- Medium severity: {medium_count}
- Low severity: {low_count}
- Keyword-detected: {keyword_count}
- LLM-detected: {llm_count}

Risk Flags by Category:
{risk_details}

Generate the executive risk briefing report in markdown format."""


_llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0.1,  # Slight creativity for narrative flow, but still factual
    api_key=settings.openai_api_key,
    max_tokens=3000,
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
])

_chain = _prompt | _llm


def _format_risk_details(flags: list[RiskFlag]) -> str:
    """Format risk flags into a structured text block for the prompt.
    
    Groups flags by category and includes severity, title, description,
    and detection method. This gives the LLM enough context to write
    a meaningful narrative without overwhelming it.
    """
    # Group by category
    by_category = defaultdict(list)
    for flag in flags:
        by_category[flag.category].append(flag)

    parts = []
    for category, cat_flags in sorted(by_category.items()):
        parts.append(f"\n### {category} ({len(cat_flags)} flags)")
        for flag in cat_flags:
            detection_label = f"[{flag.detection}]"
            if flag.confidence:
                detection_label += f" (confidence: {flag.confidence:.1f})"
            parts.append(
                f"- [{flag.severity}] {detection_label} {flag.title}: {flag.description}\n"
                f"  Source: {flag.source_text[:200]}..."
            )

    return "\n".join(parts)


async def generate_report(db: AsyncSession, filing_id: uuid.UUID) -> str:
    """Generate an executive risk report for a filing.
    
    Fetches all risk flags from the database, formats them into context,
    and sends a single request to GPT-4o-mini to generate the narrative.
    
    Args:
        db: Database session
        filing_id: Which filing to generate the report for
        
    Returns:
        Markdown string containing the executive report
    """
    # Fetch filing + company info
    result = await db.execute(
        select(Filing).where(Filing.id == filing_id)
    )
    filing = result.scalar_one_or_none()
    if not filing:
        raise ValueError(f"Filing {filing_id} not found")

    result = await db.execute(
        select(Company).where(Company.id == filing.company_id)
    )
    company = result.scalar_one()

    # Fetch all risk flags
    result = await db.execute(
        select(RiskFlag)
        .where(RiskFlag.filing_id == filing_id)
        .order_by(RiskFlag.severity, RiskFlag.category)
    )
    flags = result.scalars().all()

    if not flags:
        return (
            f"# Executive Risk Summary: {company.name} {filing.filing_type}\n\n"
            f"No risk flags were identified for this filing. "
            f"Run the risk analysis endpoint first."
        )

    # Count stats
    high_count = sum(1 for f in flags if f.severity == "High")
    medium_count = sum(1 for f in flags if f.severity == "Medium")
    low_count = sum(1 for f in flags if f.severity == "Low")
    keyword_count = sum(1 for f in flags if f.detection == "keyword")
    llm_count = sum(1 for f in flags if f.detection == "llm")

    # Format risk details for the prompt
    risk_details = _format_risk_details(flags)

    logger.info(
        f"Generating report for {company.ticker} {filing.filing_type}: "
        f"{len(flags)} flags ({high_count}H/{medium_count}M/{low_count}L)"
    )

    # Generate the report
    response = await _chain.ainvoke({
        "company_name": company.name,
        "ticker": company.ticker,
        "filing_type": filing.filing_type,
        "filing_date": filing.filing_date.isoformat(),
        "total_risks": len(flags),
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "keyword_count": keyword_count,
        "llm_count": llm_count,
        "risk_details": risk_details,
    })

    return response.content