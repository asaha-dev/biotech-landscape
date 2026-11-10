from datetime import date
from typing import Any

import httpx
import pytest

from biotech_landscape.schemas import PaperSource
from biotech_landscape.sources import SourceError
from biotech_landscape.sources.openalex import (
    OpenAlexClient,
    build_filter,
    is_ml_venue,
    parse_work,
    rebuild_abstract,
)

START, END = date(2024, 11, 1), date(2026, 11, 8)


def work(**overrides: Any) -> dict[str, Any]:
    """A trimmed OpenAlex work, shaped like the real API."""
    data: dict[str, Any] = {
        "id": "https://openalex.org/W4400000001",
        "doi": "https://doi.org/10.48550/ARXIV.2601.00001",
        "title": "Reward-guided  diffusion for protein design",
        "publication_date": "2026-01-15",
        "cited_by_count": 42,
        "type": "article",
        "primary_location": {
            "source": {
                "display_name": "Neural Information Processing Systems",
                "type": "conference",
            }
        },
        "abstract_inverted_index": {
            "We": [0],
            "guide": [1],
            "diffusion": [2, 4],
            "with": [3],
        },
        "authorships": [
            {"raw_affiliation_strings": ["Genentech, South San Francisco, CA"]},
            {
                "raw_affiliation_strings": [
                    "Genentech, South San Francisco, CA",
                    "ETH Zurich",
                ]
            },
        ],
    }
    data.update(overrides)
    return data


class FakeAPI:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


def make_client(
    api: FakeAPI, api_key: str | None = "test-key", **kw: Any
) -> OpenAlexClient:
    http = httpx.Client(transport=httpx.MockTransport(api))
    return OpenAlexClient(http, api_key=api_key, sleep=lambda _: None, **kw)


def ok(results: list[dict[str, Any]], next_cursor: str | None) -> httpx.Response:
    return httpx.Response(
        200, json={"meta": {"next_cursor": next_cursor}, "results": results}
    )


# --- pure helpers -------------------------------------------------------------------


def test_build_filter_combines_parts() -> None:
    f = build_filter(
        ["Genentech", "Roche"], START, END, terms=["foundation model", "agentic"]
    )
    assert f == (
        'raw_affiliation_strings.search:"Genentech"|"Roche",'
        'title_and_abstract.search:"foundation model" OR "agentic",'
        "from_publication_date:2024-11-01,to_publication_date:2026-11-08"
    )


def test_build_filter_strips_characters_that_break_syntax() -> None:
    f = build_filter(['Merck, "Sharp" | Dohme'], START, END, terms=["llm"])
    assert 'raw_affiliation_strings.search:"Merck Sharp Dohme"' in f


@pytest.mark.parametrize(
    ("affs", "terms", "start", "end"),
    [([], ["x"], START, END), (["GSK"], [], START, END), (["GSK"], ["x"], END, START)],
)
def test_build_filter_rejects_bad_input(
    affs: list[str], terms: list[str], start: date, end: date
) -> None:
    with pytest.raises(ValueError):
        build_filter(affs, start, end, terms)


def test_rebuild_abstract_orders_words() -> None:
    assert rebuild_abstract({"b": [1], "a": [0], "c": [2, 3]}) == "a b c c"
    assert rebuild_abstract(None) == ""


@pytest.mark.parametrize(
    ("venue", "expected"),
    [
        ("Neural Information Processing Systems", True),
        ("arXiv (Cornell University)", True),
        ("ICLR 2026", True),
        ("Nature Medicine", False),
        ("", False),
    ],
)
def test_is_ml_venue(venue: str, expected: bool) -> None:
    assert is_ml_venue(venue) is expected


def test_parse_work_maps_fields() -> None:
    p = parse_work(work())
    assert p is not None
    assert p.key == "10.48550/arxiv.2601.00001"
    assert p.source is PaperSource.OPENALEX and p.source_id == "W4400000001"
    assert p.title == "Reward-guided diffusion for protein design"  # whitespace tidied
    assert p.abstract == "We guide diffusion with diffusion"
    assert p.published == date(2026, 1, 15)
    assert p.ml_venue and not p.is_preprint
    assert p.cited_by == 42
    assert p.affiliations == ["Genentech, South San Francisco, CA", "ETH Zurich"]


def test_parse_preprint_without_doi() -> None:
    p = parse_work(
        work(
            doi=None,
            type="preprint",
            primary_location={
                "source": {"display_name": "bioRxiv", "type": "repository"}
            },
        )
    )
    assert p is not None
    assert p.is_preprint
    assert p.key == "openalex:W4400000001"
    assert p.url == "https://openalex.org/W4400000001"


@pytest.mark.parametrize(
    "broken",
    [{"title": None}, {"publication_date": None}, {"publication_date": "2026-02-30"}],
)
def test_unusable_works_are_skipped(broken: dict[str, Any]) -> None:
    assert parse_work(work(**broken)) is None


def test_parse_tolerates_missing_location_and_authors() -> None:
    p = parse_work(
        work(primary_location=None, authorships=None, abstract_inverted_index=None)
    )
    assert p is not None
    assert p.venue == "" and p.affiliations == [] and p.abstract == ""


# --- client -------------------------------------------------------------------------


def test_search_follows_cursor_and_sends_key() -> None:
    api = FakeAPI(
        [
            ok([work(id="https://openalex.org/W1", doi=None)], "next-1"),
            ok([work(id="https://openalex.org/W2", doi=None)], None),
        ]
    )
    client = make_client(api, per_page=1)
    keys = [p.key for p in client.search(["Genentech"], START, END)]
    assert keys == ["openalex:W1", "openalex:W2"]
    assert len(api.requests) == 2
    first = api.requests[0].url.params
    assert first["cursor"] == "*" and first["api_key"] == "test-key"
    assert first["sort"] == "cited_by_count:desc"
    assert api.requests[1].url.params["cursor"] == "next-1"


def test_search_without_key_omits_param() -> None:
    api = FakeAPI([ok([], None)])
    assert list(make_client(api, api_key=None).search(["GSK"], START, END)) == []
    assert "api_key" not in api.requests[0].url.params


def test_missing_key_gives_actionable_error() -> None:
    api = FakeAPI([httpx.Response(401, json={"error": "unauthorised"})])
    with pytest.raises(SourceError, match="OPENALEX_API_KEY"):
        list(make_client(api, api_key=None).search(["GSK"], START, END))
    assert len(api.requests) == 1  # no retries on 401


def test_max_results_caps_output() -> None:
    results = [work(id=f"https://openalex.org/W{i}", doi=None) for i in range(5)]
    api = FakeAPI([ok(results, "more")])
    assert len(list(make_client(api).search(["GSK"], START, END, max_results=2))) == 2


def test_citation_counts_batches_dois_and_ids() -> None:
    dois = [f"10.1/{i}" for i in range(60)]
    keys = [*dois, "openalex:W9", "europepmc:PPR/1"]
    api = FakeAPI(
        [
            ok(
                [
                    {
                        "id": "https://openalex.org/W1",
                        "doi": "https://doi.org/10.1/0",
                        "cited_by_count": 7,
                    }
                ],
                None,
            ),
            ok([], None),
            ok(
                [{"id": "https://openalex.org/W9", "doi": None, "cited_by_count": 3}],
                None,
            ),
        ]
    )
    counts = make_client(api).citation_counts(keys)
    assert counts == {"10.1/0": 7, "openalex:W9": 3}
    filters = [r.url.params["filter"] for r in api.requests]
    assert len(filters) == 3  # 50 DOIs, 10 DOIs, 1 OpenAlex id; europepmc key skipped
    assert filters[0].startswith("doi:10.1/0|10.1/1") and filters[0].count("|") == 49
    assert filters[2] == "openalex:W9"
