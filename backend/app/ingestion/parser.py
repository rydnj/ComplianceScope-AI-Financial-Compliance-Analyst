"""SEC filing HTML parser — extracts section text from 10-K and 10-Q filings.

10-K filings have standardized sections defined by SEC regulation:
  - Item 1A: Risk Factors — what could go wrong (key for compliance analysis)
  - Item 3: Legal Proceedings — active lawsuits and regulatory actions
  - Item 7: MD&A (Management's Discussion & Analysis) — management's narrative
  - Item 8: Financial Statements — the numbers + footnotes

The challenge: every company formats these differently. The parser needs to
handle multiple "Item 1A" references (table of contents, cross-references)
and find the actual section content rather than a brief mention.

Strategy: find ALL matches for a section header, extract the text between
each match and the next section header, then pick the LONGEST block.
The actual section body is always much longer than a TOC entry or footnote.
"""

import re
import logging
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

# Suppress the XHTML warning — many filings are XHTML, BeautifulSoup handles them fine
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)


@dataclass
class FilingSection:
    """A parsed section from a filing."""
    name: str       # e.g., "Risk Factors"
    item_no: str    # e.g., "1A"
    text: str       # Cleaned plain text content


# Section definitions for 10-K filings
# Each tuple: (item_number, section_name, start_pattern, end_patterns)
# end_patterns is a list because the next section could vary
SECTION_PATTERNS_10K = [
    (
        "1A",
        "Risk Factors",
        r"item\s+1a[\.\s\-\—\:\|]*\s*risk\s+factors",
        [r"item\s+1b", r"item\s+1c", r"item\s+2[\.\s\-\—\:\|]"],
    ),
    (
        "3",
        "Legal Proceedings",
        r"item\s+3[\.\s\-\—\:\|]*\s*legal\s+proceedings",
        [r"item\s+3a", r"item\s+4[\.\s\-\—\:\|]"],
    ),
    (
        "7",
        "MD&A",
        r"item\s+7[\.\s\-\—\:\|]*\s*management[\'\u2019]?s?\s+discussion",
        [r"item\s+7a", r"item\s+8[\.\s\-\—\:\|]"],
    ),
    (
        "8",
        "Financial Statements",
        r"item\s+8[\.\s\-\—\:\|]*\s*financial\s+statements",
        [r"item\s+9[\.\s\-\—\:\|]", r"item\s+9a"],
    ),
]


def clean_html_to_text(html: str) -> str:
    """Strip HTML tags, normalize whitespace, return clean text.
    
    BeautifulSoup handles malformed HTML gracefully — important because
    SEC filings often have broken/nested tags from Word-to-HTML conversion.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    # Normalize whitespace: collapse multiple spaces/newlines into single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _find_best_section(text: str, start_pattern: str, end_patterns: list[str]) -> str | None:
    """Find the actual section content by selecting the longest match.
    
    Strategy:
      1. Find ALL positions where the start pattern matches
      2. For each match, find the nearest end pattern after it
      3. Extract the text between start and end
      4. Return the LONGEST extraction — this is the real section body
    
    Why longest? A table of contents entry for "Item 1A. Risk Factors"
    is ~50 chars. The actual Risk Factors section is 20,000+ chars.
    Cross-references are also short. So the longest match is always the
    real content.
    """
    # Find all start positions
    start_matches = list(re.finditer(start_pattern, text, re.IGNORECASE))
    if not start_matches:
        return None

    # Build a combined end pattern — matches any of the possible next sections
    end_pattern = "|".join(end_patterns)

    candidates = []
    for start_match in start_matches:
        start_pos = start_match.start()

        # Search for the end pattern after this start position
        # Skip a bit ahead to avoid matching the start pattern itself
        search_from = start_pos + len(start_match.group())
        end_match = re.search(end_pattern, text[search_from:], re.IGNORECASE)

        if end_match:
            end_pos = search_from + end_match.start()
        else:
            # No end marker found — take up to 200K chars (safety limit)
            end_pos = min(start_pos + 200_000, len(text))

        section_text = text[start_pos:end_pos].strip()
        candidates.append(section_text)

    if not candidates:
        return None

    # Pick the longest candidate — that's the real section content
    best = max(candidates, key=len)
    return best


def _clean_section_text(text: str) -> str:
    """Remove page headers/footers and other boilerplate from section text.
    
    Apple's filings have repeated headers like "Apple Inc. | 2025 Form 10-K | 28"
    scattered throughout. These create noisy chunks that hurt retrieval quality.
    """
    # Remove patterns like "Apple Inc. | 2025 Form 10-K | 28"
    # and similar company name + form + page number lines
    text = re.sub(
        r"[A-Z][A-Za-z\s\.,]+Inc\.?\s*\|?\s*\d{4}\s*Form\s*10-[KkQq]\s*\|?\s*\d*",
        "",
        text,
    )

    # Remove standalone page numbers (just a number on its own)
    text = re.sub(r"\b\d{1,3}\s+(?=[A-Z])", "", text)

    # Collapse any resulting multiple spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def extract_sections(html: str, filing_type: str = "10-K") -> list[FilingSection]:
    """Extract key sections from a filing's HTML.
    
    For each target section:
      1. Find all regex matches for the section header
      2. Extract text between each match and the next section header
      3. Pick the longest extraction (the real section, not TOC/cross-refs)
      4. Clean out boilerplate (page headers, footers)
      5. Skip sections that are too short (< 500 chars = likely a false match)
    """
    text = clean_html_to_text(html)
    sections = []

    if filing_type == "10-K":
        patterns = SECTION_PATTERNS_10K
    else:
        patterns = SECTION_PATTERNS_10K  # 10-Q shares similar structure

    for item_no, section_name, start_pattern, end_patterns in patterns:
        try:
            section_text = _find_best_section(text, start_pattern, end_patterns)

            if section_text is None:
                logger.warning(f"Section '{section_name}' (Item {item_no}) not found")
                continue

            # Clean out page headers and boilerplate
            section_text = _clean_section_text(section_text)

            # Skip if too short — likely a false match or cross-reference
            if len(section_text) < 500:
                logger.warning(
                    f"Section '{section_name}' too short ({len(section_text)} chars), skipping"
                )
                continue

            sections.append(FilingSection(
                name=section_name,
                item_no=item_no,
                text=section_text,
            ))
            logger.info(
                f"Extracted '{section_name}' (Item {item_no}): {len(section_text):,} chars"
            )

        except Exception as e:
            logger.error(f"Error extracting section '{section_name}': {e}")
            continue

    if not sections:
        logger.error("No sections could be extracted from the filing")
        raise ValueError("Failed to extract any sections from the filing HTML")

    return sections