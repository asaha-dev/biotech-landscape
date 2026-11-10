"""Europe PMC search client: AI papers by author affiliation, with abstracts.

Europe PMC indexes PubMed/MEDLINE plus bioRxiv/medRxiv preprints. Lessons from
running this by hand are built in:

- Stop paging once every hit is in. Asking for a page past the end makes Europe
  PMC answer 503 rather than an empty page.
- Retry 429/5xx and dropped connections with backoff; fail fast on anything else.
- Full ("core") pages are slow, so page size stays at 100.
"""

import logging
import re
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import date
from typing import Any

import httpx
from pydantic import ValidationError

from biotech_landscape.schemas import Paper, PaperSource, paper_key
from biotech_landscape.sources.http import default_client, get_json

__all__ = [
    "DEFAULT_AI_TERMS",
    "EuropePMCClient",
    "build_query",
    "default_client",
    "parse_record",
]

log = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

DEFAULT_AI_TERMS: tuple[str, ...] = (
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "foundation model",
    "large language model",
    "neural network",
    "generative model",
    "transformer",
)

_TAG = re.compile(r"<[^>]+>")


def _quote(phrase: str) -> str:
    return '"' + phrase.replace('"', " ").strip() + '"'


def build_query(
    affiliations: Sequence[str],
    start: date,
    end: date,
    ai_terms: Sequence[str] = DEFAULT_AI_TERMS,
) -> str:
    """Europe PMC query: any affiliation AND any AI term AND first published in range."""
    if not affiliations:
        raise ValueError("at least one affiliation phrase is required")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    aff = " OR ".join(f"AFF:{_quote(a)}" for a in affiliations)
    ai = " OR ".join(_quote(t) for t in ai_terms)
    return (
        f"({aff}) AND ({ai}) AND FIRST_PDATE:[{start.isoformat()} TO {end.isoformat()}]"
    )


def _affiliations(record: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for author in record.get("authorList", {}).get("author", []):
        if author.get("affiliation"):
            found.append(author["affiliation"])
        details = author.get("authorAffiliationDetailsList", {})
        for aff in details.get("authorAffiliation", []):
            if aff.get("affiliation"):
                found.append(aff["affiliation"])
    return list(dict.fromkeys(a.strip() for a in found if a.strip()))


def parse_record(record: dict[str, Any]) -> Paper | None:
    """Map one Europe PMC "core" result to a Paper; None if it can't be used.

    Pure, so field mapping is tested without any HTTP.
    """
    src, rec_id = record.get("source", ""), record.get("id", "")
    published = record.get("firstPublicationDate")
    if not (src and rec_id and published and record.get("title")):
        return None
    source_id = f"{src}/{rec_id}"
    doi = record.get("doi")
    is_preprint = src == "PPR"
    venue = (
        record.get("journalInfo", {}).get("journal", {}).get("title")
        or record.get("bookOrReportDetails", {}).get("publisher")
        or ""
    )
    try:
        return Paper(
            key=paper_key(doi, PaperSource.EUROPEPMC, source_id),
            source=PaperSource.EUROPEPMC,
            source_id=source_id,
            title=_TAG.sub("", record["title"]).strip(),
            abstract=_TAG.sub("", record.get("abstractText", "")).strip(),
            published=date.fromisoformat(published),
            venue=venue,
            cited_by=int(record.get("citedByCount", 0)),
            url=f"https://doi.org/{doi}"
            if doi
            else f"https://europepmc.org/article/{source_id}",
            is_preprint=is_preprint,
            affiliations=_affiliations(record),
        )
    except (ValidationError, ValueError) as e:
        log.warning("skipping Europe PMC record %s: %s", source_id, e)
        return None


class EuropePMCClient:
    """Searches Europe PMC through an injected httpx client.

    `sleep` is injected so tests don't wait out real backoff delays.
    """

    def __init__(
        self,
        http: httpx.Client,
        *,
        base_url: str = BASE_URL,
        page_size: int = 100,
        max_attempts: int = 6,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        self._http = http
        self._base_url = base_url
        self._page_size = page_size
        self._max_attempts = max_attempts
        self._sleep = sleep

    def search(
        self,
        affiliations: Sequence[str],
        start: date,
        end: date,
        *,
        ai_terms: Sequence[str] = DEFAULT_AI_TERMS,
        max_results: int | None = None,
    ) -> Iterator[Paper]:
        """Yield papers matching the query, paging with Europe PMC's cursor."""
        query = build_query(affiliations, start, end, ai_terms)
        cursor, seen = "*", 0
        while True:
            data = self._get(
                {
                    "query": query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": self._page_size,
                    "cursorMark": cursor,
                }
            )
            hits = int(data.get("hitCount", 0))
            results = data.get("resultList", {}).get("result", [])
            for record in results:
                seen += 1
                paper = parse_record(record)
                if paper is not None:
                    yield paper
                if max_results is not None and seen >= max_results:
                    return
            next_cursor = data.get("nextCursorMark")
            # Stop once every hit is in: a request past the end returns 503.
            if not results or seen >= hits or not next_cursor or next_cursor == cursor:
                return
            cursor = next_cursor

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        return get_json(
            self._http,
            self._base_url,
            params,
            name="Europe PMC",
            max_attempts=self._max_attempts,
            sleep=self._sleep,
        )
