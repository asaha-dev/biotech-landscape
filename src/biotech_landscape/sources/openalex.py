"""OpenAlex client: methods-heavy AI papers by raw author affiliation.

OpenAlex also indexes arXiv and ML conference proceedings (NeurIPS, ICML, ICLR),
where most frontier-methods work appears, which Europe PMC largely misses.

Beyond a small shared daily budget OpenAlex needs a free API key
(https://openalex.org/settings/api); without one it answers 401.
"""

import logging
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import date
from typing import Any

import httpx
from pydantic import ValidationError

from biotech_landscape.schemas import Paper, PaperSource, paper_key
from biotech_landscape.sources.http import default_client, get_json

__all__ = [
    "ADVANCED_TERMS",
    "OpenAlexClient",
    "build_filter",
    "default_client",
    "is_ml_venue",
    "parse_work",
    "rebuild_abstract",
]

log = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org/works"
SELECT = (
    "id,doi,title,publication_date,cited_by_count,primary_location,type,"
    "abstract_inverted_index,authorships"
)
CITATION_BATCH = 50  # OpenAlex accepts up to 50 OR'd values per filter

ADVANCED_TERMS: tuple[str, ...] = (
    "foundation model",
    "large language model",
    "language model",
    "diffusion model",
    "flow matching",
    "generative model",
    "transformer",
    "graph neural network",
    "equivariant",
    "reinforcement learning",
    "agentic",
    "AI agent",
    "multi-agent",
    "active learning",
    "bayesian optimization",
    "federated learning",
    "self-supervised",
    "contrastive learning",
    "protein language model",
    "interpretability",
    "sparse autoencoder",
    "virtual cell",
    "multimodal",
)

ML_VENUES: tuple[str, ...] = (
    "neurips",
    "neural information processing",
    "icml",
    "international conference on machine learning",
    "iclr",
    "learning representations",
    "arxiv",
    "machine learning research",
    "aaai",
    "openreview",
)

_UNSAFE = re.compile(r'[",|]')  # these would break OpenAlex filter syntax
_HINTS = {
    401: "OpenAlex needs an API key: set OPENALEX_API_KEY "
    "(free at https://openalex.org/settings/api)",
    403: "OpenAlex refused the API key; check OPENALEX_API_KEY",
}


def _phrase(text: str) -> str:
    return '"' + " ".join(_UNSAFE.sub(" ", text).split()) + '"'


def build_filter(
    affiliations: Sequence[str],
    start: date,
    end: date,
    terms: Sequence[str] = ADVANCED_TERMS,
) -> str:
    """OpenAlex `filter` value: any affiliation AND any term AND published in range."""
    if not affiliations:
        raise ValueError("at least one affiliation phrase is required")
    if not terms:
        raise ValueError("at least one search term is required")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    return ",".join(
        [
            "raw_affiliation_strings.search:"
            + "|".join(_phrase(a) for a in affiliations),
            "title_and_abstract.search:" + " OR ".join(_phrase(t) for t in terms),
            f"from_publication_date:{start.isoformat()}",
            f"to_publication_date:{end.isoformat()}",
        ]
    )


def rebuild_abstract(inverted: Mapping[str, Sequence[int]] | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    if not inverted:
        return ""
    positioned = sorted(
        (pos, word) for word, positions in inverted.items() for pos in positions
    )
    return " ".join(word for _, word in positioned)


def is_ml_venue(venue: str) -> bool:
    v = venue.lower()
    return any(name in v for name in ML_VENUES)


def _short_id(openalex_id: str) -> str:
    return openalex_id.rstrip("/").rsplit("/", 1)[-1]


def parse_work(work: Mapping[str, Any]) -> Paper | None:
    """Map one OpenAlex work to a Paper; None if it can't be used. Pure."""
    raw_id, title, published = (
        work.get("id"),
        work.get("title"),
        work.get("publication_date"),
    )
    if not (raw_id and title and published):
        return None
    source_id = _short_id(raw_id)
    doi = work.get("doi")
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    venue = source.get("display_name") or ""
    affiliations = [
        aff.strip()
        for authorship in work.get("authorships") or []
        for aff in authorship.get("raw_affiliation_strings") or []
        if aff and aff.strip()
    ]
    try:
        return Paper(
            key=paper_key(doi, PaperSource.OPENALEX, source_id),
            source=PaperSource.OPENALEX,
            source_id=source_id,
            title=" ".join(title.split()),
            abstract=rebuild_abstract(work.get("abstract_inverted_index")),
            published=date.fromisoformat(published),
            venue=venue,
            cited_by=int(work.get("cited_by_count") or 0),
            url=doi or raw_id,
            is_preprint=work.get("type") == "preprint"
            or source.get("type") == "repository",
            ml_venue=is_ml_venue(venue),
            affiliations=list(dict.fromkeys(affiliations)),
        )
    except (ValidationError, ValueError) as e:
        log.warning("skipping OpenAlex work %s: %s", source_id, e)
        return None


class OpenAlexClient:
    """Searches OpenAlex through an injected httpx client.

    The API key is passed in rather than read from the environment here, so
    configuration stays in one place (the CLI) and tests never need a real key.
    """

    def __init__(
        self,
        http: httpx.Client,
        *,
        api_key: str | None,
        base_url: str = BASE_URL,
        per_page: int = 200,
        max_attempts: int = 6,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1 <= per_page <= 200:
            raise ValueError("per_page must be between 1 and 200")
        self._http = http
        self._api_key = api_key or None
        self._base_url = base_url
        self._per_page = per_page
        self._max_attempts = max_attempts
        self._sleep = sleep

    def search(
        self,
        affiliations: Sequence[str],
        start: date,
        end: date,
        *,
        terms: Sequence[str] = ADVANCED_TERMS,
        max_results: int | None = None,
    ) -> Iterator[Paper]:
        """Yield matching papers, most cited first, using OpenAlex cursor paging."""
        flt = build_filter(affiliations, start, end, terms)
        cursor: str | None = "*"
        seen = 0
        while cursor:
            data = self._get(
                {
                    "filter": flt,
                    "per-page": self._per_page,
                    "cursor": cursor,
                    "select": SELECT,
                    "sort": "cited_by_count:desc",
                }
            )
            results = data.get("results") or []
            for work in results:
                seen += 1
                paper = parse_work(work)
                if paper is not None:
                    yield paper
                if max_results is not None and seen >= max_results:
                    return
            if not results:
                return
            cursor = (data.get("meta") or {}).get("next_cursor")

    def citation_counts(self, keys: Sequence[str]) -> dict[str, int]:
        """Current citation counts for stored papers, looked up in batches.

        Keys are paper_key values: DOIs, or "openalex:W123" for papers without one.
        Europe PMC-only keys ("europepmc:...") are skipped; OpenAlex can't resolve
        them. Returns {paper_key: cited_by_count} for the works OpenAlex found.
        """
        dois = [k for k in keys if ":" not in k.split("/", 1)[0]]
        ids = [k.split(":", 1)[1] for k in keys if k.startswith("openalex:")]
        counts: dict[str, int] = {}
        for field, values in (("doi", dois), ("openalex", ids)):
            for i in range(0, len(values), CITATION_BATCH):
                batch = values[i : i + CITATION_BATCH]
                data = self._get(
                    {
                        "filter": f"{field}:" + "|".join(batch),
                        "per-page": CITATION_BATCH,
                        "select": "id,doi,cited_by_count",
                    }
                )
                for work in data.get("results") or []:
                    key = paper_key(
                        work.get("doi"),
                        PaperSource.OPENALEX,
                        _short_id(work.get("id", "")),
                    )
                    counts[key] = int(work.get("cited_by_count") or 0)
        return counts

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._api_key:
            params = {**params, "api_key": self._api_key}
        return get_json(
            self._http,
            self._base_url,
            params,
            name="OpenAlex",
            max_attempts=self._max_attempts,
            sleep=self._sleep,
            hints=_HINTS,
        )
