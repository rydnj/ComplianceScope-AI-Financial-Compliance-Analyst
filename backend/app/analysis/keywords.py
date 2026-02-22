"""Keyword-based risk detection (Tier 1).

Scans chunk text for predefined patterns that indicate specific compliance risks.
This is fast, free (no API calls), and high-precision — if the exact phrase
"material weakness" appears, it's almost certainly a Material Weakness risk.

The tradeoff: keywords are high-precision but low-recall. They only catch
risks expressed using the exact phrases we've defined. Subtle or indirect
references are caught by the LLM classifier (Tier 2).

Each pattern maps to a risk category and default severity based on how
serious the SEC considers that type of disclosure.
"""

import re
import logging
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KeywordRiskFlag:
    """A risk detected by keyword matching."""
    category: str
    severity: str       # High, Medium, Low
    title: str
    description: str
    source_text: str    # The excerpt from the chunk that matched
    chunk_id: uuid.UUID
    pattern_matched: str  # Which pattern triggered this flag


# Each tuple: (compiled_regex, category, severity, title_template, description_template)
# Title and description templates provide context about what the keyword means
RISK_PATTERNS = [
    # --- High Severity ---
    (
        re.compile(r"material\s+weakness", re.IGNORECASE),
        "Material Weakness",
        "High",
        "Material Weakness in Internal Controls",
        "Filing contains explicit mention of a material weakness, indicating "
        "significant deficiency in the company's internal controls over financial reporting. "
        "This requires immediate auditor and management attention.",
    ),
    (
        re.compile(r"going\s+concern", re.IGNORECASE),
        "Going Concern",
        "High",
        "Going Concern Doubt",
        "Filing references going concern language, suggesting substantial doubt "
        "about the company's ability to continue operating. This is one of the most "
        "serious disclosures in SEC filings.",
    ),
    (
        re.compile(r"SEC\s+investigation|securities\s+and\s+exchange\s+commission\s+investigation", re.IGNORECASE),
        "Regulatory Action",
        "High",
        "SEC Investigation Disclosed",
        "Filing discloses an investigation by the Securities and Exchange Commission, "
        "indicating potential violations of securities laws.",
    ),
    (
        re.compile(r"consent\s+(?:order|decree)", re.IGNORECASE),
        "Regulatory Action",
        "High",
        "Regulatory Consent Order",
        "Filing references a consent order or decree, indicating the company has "
        "agreed to regulatory terms, often following an enforcement action.",
    ),
    (
        # --- Medium Severity ---
        re.compile(r"restatement\s+of\s+(?:financial|previously)", re.IGNORECASE),
        "Revenue Recognition",
        "Medium",
        "Financial Restatement",
        "Filing mentions a restatement of financial results, indicating prior "
        "financial statements contained material errors.",
    ),
    (
        re.compile(r"related\s+party\s+transaction", re.IGNORECASE),
        "Related Party Transactions",
        "Medium",
        "Related Party Transaction Disclosed",
        "Filing discloses transactions between the company and related parties "
        "(officers, directors, major shareholders), which require scrutiny for fairness.",
    ),
    (
        re.compile(r"(?:pending|ongoing)\s+litigation|(?:lawsuit|legal\s+(?:action|proceeding))\s+(?:filed|brought|commenced)", re.IGNORECASE),
        "Litigation",
        "Medium",
        "Active Litigation Disclosed",
        "Filing discloses pending or ongoing legal proceedings that could "
        "materially affect the company's financial position.",
    ),
    (
        re.compile(r"antitrust\s+(?:lawsuit|investigation|proceeding|action)", re.IGNORECASE),
        "Litigation",
        "Medium",
        "Antitrust Action Disclosed",
        "Filing discloses antitrust-related legal proceedings or investigations.",
    ),
    (
        re.compile(r"cybersecurity\s+(?:incident|breach|attack)", re.IGNORECASE),
        "Cybersecurity",
        "Medium",
        "Cybersecurity Incident Disclosed",
        "Filing discloses a cybersecurity incident, breach, or attack that "
        "may affect the company's data, operations, or reputation.",
    ),
    (
        re.compile(r"data\s+breach|unauthorized\s+access\s+to\s+(?:personal|customer|user)\s+data", re.IGNORECASE),
        "Cybersecurity",
        "Medium",
        "Data Breach Disclosed",
        "Filing references a data breach or unauthorized access to sensitive data.",
    ),
    (
        re.compile(r"revenue\s+recognition\s+(?:error|issue|adjustment|change)", re.IGNORECASE),
        "Revenue Recognition",
        "Medium",
        "Revenue Recognition Issue",
        "Filing mentions issues with revenue recognition practices, which could "
        "indicate improper accounting of income.",
    ),
]


def scan_chunk_for_risks(
    chunk_id: uuid.UUID, content: str, section: str
) -> list[KeywordRiskFlag]:
    """Scan a single chunk for keyword-based risk patterns.
    
    Args:
        chunk_id: Database ID of the chunk
        content: The chunk's text content
        section: Which filing section the chunk came from
        
    Returns:
        List of KeywordRiskFlag objects for any patterns matched.
        A single chunk can trigger multiple flags if it matches multiple patterns.
    """
    flags = []

    for pattern, category, severity, title, description in RISK_PATTERNS:
        match = pattern.search(content)
        if match:
            # Extract a window around the match for context
            start = max(0, match.start() - 100)
            end = min(len(content), match.end() + 100)
            source_excerpt = content[start:end].strip()

            flags.append(KeywordRiskFlag(
                category=category,
                severity=severity,
                title=title,
                description=description,
                source_text=source_excerpt,
                chunk_id=chunk_id,
                pattern_matched=match.group(),
            ))

    return flags


def scan_all_chunks(chunks: list[dict]) -> list[KeywordRiskFlag]:
    """Scan all chunks from a filing for keyword risks.
    
    Args:
        chunks: List of dicts with keys: id, content, section
        
    Returns:
        All keyword risk flags found across all chunks
    """
    all_flags = []

    for chunk in chunks:
        chunk_flags = scan_chunk_for_risks(
            chunk_id=chunk["id"],
            content=chunk["content"],
            section=chunk["section"],
        )
        all_flags.extend(chunk_flags)

    logger.info(f"Keyword scan complete: {len(all_flags)} flags from {len(chunks)} chunks")
    return all_flags