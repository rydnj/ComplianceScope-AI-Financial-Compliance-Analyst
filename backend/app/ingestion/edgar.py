"""SEC EDGAR API client for fetching company filings.

Uses the official free SEC EDGAR APIs (no API key required):
  1. company_tickers.json — maps ticker symbols to CIK numbers
  2. data.sec.gov/submissions/ — filing history for a company by CIK
  3. sec.gov/Archives/edgar/data/ — actual filing documents

EDGAR requires a User-Agent header with a real name/email on every request.
Rate limit: 10 requests/sec — we add 0.1s sleep between requests.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FilingMetadata:
    """Parsed metadata for a single SEC filing."""
    company_name: str
    ticker: str
    cik: str              # SEC's unique company identifier (zero-padded to 10 digits)
    accession_no: str     # Unique filing identifier, e.g. "0000320193-23-000106"
    filing_type: str      # "10-K" or "10-Q"
    filing_date: date
    document_url: str     # Direct URL to the filing HTML document


class EdgarClient:
    """Async client for the official SEC EDGAR APIs.
    
    Flow: ticker → CIK lookup → submissions API → fetch filing HTML
    All endpoints are free, public, and require no authentication.
    The only requirement is a User-Agent header identifying who you are.
    """

    def __init__(self):
        self.headers = {"User-Agent": settings.edgar_user_agent}
        # Cache the ticker→CIK mapping so we only fetch it once
        self._tickers_cache: dict | None = None

    async def _get_tickers_map(self) -> dict:
        """Fetch and cache the SEC's ticker→CIK mapping file.
        
        This is a single JSON file (~2MB) that maps every public company's
        ticker to its CIK number. We cache it because it rarely changes
        and we don't want to re-download it for every request.
        """
        if self._tickers_cache is not None:
            return self._tickers_cache

        url = "https://www.sec.gov/files/company_tickers.json"
        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            self._tickers_cache = resp.json()
            return self._tickers_cache

    async def get_cik(self, ticker: str) -> tuple[str, str]:
        """Look up a company's CIK number and name from its ticker symbol.
        
        CIK (Central Index Key) is the SEC's unique identifier for each
        filing entity. We need it to query the submissions API.
        
        Returns:
            Tuple of (cik_padded, company_name).
            CIK is zero-padded to 10 digits as required by the submissions API.
        """
        tickers = await self._get_tickers_map()

        ticker_upper = ticker.upper()
        for _, entry in tickers.items():
            if entry["ticker"].upper() == ticker_upper:
                # Zero-pad CIK to 10 digits — the submissions API requires this format
                cik_padded = str(entry["cik_str"]).zfill(10)
                company_name = entry["title"]
                return cik_padded, company_name

        raise ValueError(f"Ticker '{ticker}' not found in SEC database")

    async def get_filings_list(
        self, cik: str, filing_type: str, limit: int = 5
    ) -> list[dict]:
        """Get a company's recent filings of a specific type from the Submissions API.
        
        The submissions endpoint returns a columnar data structure — arrays
        where index 0 across all arrays is one filing, index 1 is the next, etc.
        We zip these into a list of dicts for easier handling.
        
        Args:
            cik: Zero-padded 10-digit CIK
            filing_type: "10-K" or "10-Q"
            limit: Max number of filings to return
            
        Returns:
            List of dicts with keys: accessionNumber, filingDate, primaryDocument, form
        """
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        # The "recent" key contains columnar arrays of filing data
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            raise ValueError(f"No filing data found for CIK {cik}")

        # Zip the columnar arrays into a list of per-filing dicts
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_documents = recent.get("primaryDocument", [])

        results = []
        for i in range(len(forms)):
            if forms[i] == filing_type:
                results.append({
                    "accessionNumber": accession_numbers[i],
                    "filingDate": filing_dates[i],
                    "primaryDocument": primary_documents[i],
                    "form": forms[i],
                })
                if len(results) >= limit:
                    break

        if not results:
            raise ValueError(
                f"No {filing_type} filings found for CIK {cik}. "
                f"Available form types: {list(set(forms[:20]))}"
            )

        return results

    def _build_filing_url(self, cik: str, accession_no: str, primary_document: str) -> str:
        """Construct the direct URL to a filing document.
        
        EDGAR filing URLs follow this pattern:
        https://www.sec.gov/Archives/edgar/data/{CIK}/{accession-no-no-dashes}/{primary-document}
        
        The accession number in the URL has dashes removed.
        """
        accession_no_clean = accession_no.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik.lstrip('0')}/{accession_no_clean}/{primary_document}"
        )

    async def fetch_filing_html(self, url: str) -> str:
        """Download the raw HTML content of a filing document.
        
        10-K filings can be large (1-10MB of HTML), so we use a generous timeout.
        """
        async with httpx.AsyncClient(
            headers=self.headers, timeout=60.0, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    async def get_latest_filing(
        self, ticker: str, filing_type: str
    ) -> tuple[FilingMetadata, str]:
        """Main entry point: fetch the most recent filing for a ticker.
        
        This orchestrates the full flow:
          1. Ticker → CIK lookup
          2. CIK → find the most recent filing of the requested type
          3. Download the filing HTML
        
        Returns:
            Tuple of (FilingMetadata, raw_html_string)
        """
        # Step 1: Ticker → CIK
        cik, company_name = await self.get_cik(ticker)
        logger.info(f"Resolved {ticker} → CIK {cik} ({company_name})")
        await asyncio.sleep(0.1)  # Rate limit

        # Step 2: Get filing list, take the most recent one
        filings = await self.get_filings_list(cik, filing_type, limit=1)
        filing = filings[0]
        logger.info(
            f"Found {filing_type} filed on {filing['filingDate']}: "
            f"{filing['accessionNumber']}"
        )
        await asyncio.sleep(0.1)  # Rate limit

        # Step 3: Build URL and download HTML
        doc_url = self._build_filing_url(cik, filing["accessionNumber"], filing["primaryDocument"])
        logger.info(f"Fetching filing HTML from: {doc_url}")
        html = await self.fetch_filing_html(doc_url)
        logger.info(f"Downloaded filing: {len(html):,} characters")

        metadata = FilingMetadata(
            company_name=company_name,
            ticker=ticker.upper(),
            cik=cik,
            accession_no=filing["accessionNumber"],
            filing_type=filing_type,
            filing_date=date.fromisoformat(filing["filingDate"]),
            document_url=doc_url,
        )

        return metadata, html