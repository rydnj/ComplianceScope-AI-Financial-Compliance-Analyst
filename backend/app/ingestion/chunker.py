"""Text chunking for SEC filings using LangChain's RecursiveCharacterTextSplitter.

RecursiveCharacterTextSplitter tries to split on natural boundaries in this order:
  1. Double newlines (paragraph breaks)
  2. Single newlines
  3. Spaces (word boundaries)
  4. Individual characters (last resort)

This is better than splitting on a fixed character count because it preserves
sentence and paragraph structure. A chunk of 1000 chars that ends mid-sentence
is harder to embed meaningfully than one that ends at a paragraph break.
"""

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text from a specific section of a filing."""
    content: str
    section: str      # Which filing section this came from, e.g. "Risk Factors"
    chunk_index: int  # Position within the section (0, 1, 2, ...)


def chunk_section(section_name: str, section_text: str) -> list[TextChunk]:
    """Split a single filing section into overlapping chunks.
    
    Args:
        section_name: e.g. "Risk Factors", "MD&A"
        section_text: The full extracted text of that section
        
    Returns:
        List of TextChunk objects with content, section label, and index
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,      # 1000 chars — roughly a paragraph
        chunk_overlap=settings.chunk_overlap, # 200 chars — prevents losing context at boundaries
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Try paragraph breaks first, then words
    )

    texts = splitter.split_text(section_text)

    chunks = [
        TextChunk(content=text, section=section_name, chunk_index=i)
        for i, text in enumerate(texts)
    ]

    logger.info(f"Section '{section_name}': {len(section_text):,} chars → {len(chunks)} chunks")
    return chunks


def chunk_filing(sections: list) -> list[TextChunk]:
    """Chunk all sections of a filing.
    
    Args:
        sections: List of FilingSection objects from parser.py
        
    Returns:
        All chunks across all sections, preserving section labels
    """
    all_chunks = []

    for section in sections:
        section_chunks = chunk_section(section.name, section.text)
        all_chunks.extend(section_chunks)

    logger.info(f"Total: {len(all_chunks)} chunks across {len(sections)} sections")
    return all_chunks