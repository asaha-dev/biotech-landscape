from datetime import date
from typing import Any

import httpx
import pytest

from biotech_landscape.schemas import PaperSource
from biotech_landscape.sources import SourceError, TransientSourceError
from biotech_landscape.sources.europepmc import (
    EuropePMCClient,
    build_query,
    parse_record,
)

START, END = date(2024, 11, 1), date(2026, 11, 8)


def record(**overrides: Any) -> dict[str, Any]:
    """A trimmed Europe PMC "core" result, shaped like the real API."""
    data: dict[str, Any] = {
        "id": "41000001",
        "source": "MED",
        "doi": "10.1038/s41586-026-0001",
        "title": "A <i>virtual cell</i> foundation model",
        "abstractText": "<h4>Background</h4>We train a model.",
        "firstPublicationDate": "2026-03-04",
        "journalInfo": {"journal": {"title": "Nature"}},
        "citedByCount": 12,
        "authorList": {
            "author": [
                {
                    "fullName": "A. Author",
                    "authorAffiliationDetailsList": {
                        "authorAffiliation": [
                            {"affiliation": "GSK, Stevenage, UK"},
                            {"affiliation": "University of Oxford"},
                        ]
                    },
                },
                {"fullName": "B. Author", "affiliation": "GSK, Stevenage, UK"},
            ]
        },
    }
    data.update(overrides)
    return data


def page(results: list[dict[str, Any]], hits: int, next_cursor: str) -> dict[str, Any]:
    return {
        "hitCount": hits,
        "nextCursorMark": next_cursor,
        "resultList": {"result": results},
    }


class FakeAPI:
    """Serves queued responses and records every request it receives."""

    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(api: FakeAPI, **kwargs: Any) -> tuple[EuropePMCClient, list[float]]:
    sleeps: list[float] = []
    http = httpx.Client(transport=httpx.MockTransport(api))
    return EuropePMCClient(http, sleep=sleeps.append, **kwargs), sleeps


# --- query and parsing (pure) -------------------------------------------------------


def test_build_query_combines_affiliations_terms_and_dates() -> None:
    q = build_query(["GlaxoSmithKline", "GSK"], START, END, ai_terms=["deep learning"])
    assert q == (
        '(AFF:"GlaxoSmithKline" OR AFF:"GSK") AND ("deep learning") '
        "AND FIRST_PDATE:[2024-11-01 TO 2026-11-08]"
    )


def test_build_query_neutralises_embedded_quotes() -> None:
    q = build_query(['Merck "& Co'], START, END, ai_terms=["llm"])
    # The stray quote becomes a space, so it can't close the phrase early.
    assert q.startswith('(AFF:"Merck  & Co")')
    assert q.count('"') % 2 == 0


@pytest.mark.parametrize(
    ("affs", "start", "end"), [([], START, END), (["GSK"], END, START)]
)
def test_build_query_rejects_bad_input(affs: list[str], start: date, end: date) -> None:
    with pytest.raises(ValueError):
        build_query(affs, start, end)


def test_parse_record_maps_fields() -> None:
    p = parse_record(record())
    assert p is not None
    assert p.key == "10.1038/s41586-026-0001"
    assert p.source is PaperSource.EUROPEPMC and p.source_id == "MED/41000001"
    assert p.title == "A virtual cell foundation model"  # HTML stripped
    assert p.abstract == "BackgroundWe train a model."
    assert p.published == date(2026, 3, 4)
    assert p.venue == "Nature" and p.cited_by == 12
    assert p.url == "https://doi.org/10.1038/s41586-026-0001"
    assert p.affiliations == [
        "GSK, Stevenage, UK",
        "University of Oxford",
    ]  # de-duplicated
    assert not p.is_preprint


def test_parse_preprint_without_doi() -> None:
    p = parse_record(record(source="PPR", id="PPR900", doi=None, journalInfo={}))
    assert p is not None
    assert p.is_preprint
    assert p.key == "europepmc:PPR/PPR900"
    assert p.url == "https://europepmc.org/article/PPR/PPR900"
    assert p.venue == ""


@pytest.mark.parametrize(
    "broken",
    [
        {"firstPublicationDate": None},
        {"title": ""},
        {"firstPublicationDate": "2026-13-40"},
        {"citedByCount": -1},
    ],
)
def test_unusable_records_are_skipped(broken: dict[str, Any]) -> None:
    assert parse_record(record(**broken)) is None


# --- paging and retries -------------------------------------------------------------


def test_stops_paging_when_all_hits_are_in() -> None:
    # Regression: a request past the end makes Europe PMC return 503.
    api = FakeAPI(
        [
            httpx.Response(
                200, json=page([record(), record(id="2", doi="10.1/b")], 2, "AoE2")
            )
        ]
    )
    client, _ = make_client(api)
    papers = list(client.search(["GSK"], START, END))
    assert len(papers) == 2
    assert len(api.requests) == 1


def test_follows_cursor_across_pages() -> None:
    api = FakeAPI(
        [
            httpx.Response(200, json=page([record(id="1", doi="10.1/a")], 2, "c1")),
            httpx.Response(200, json=page([record(id="2", doi="10.1/b")], 2, "c2")),
        ]
    )
    client, _ = make_client(api, page_size=1)
    assert [p.key for p in client.search(["GSK"], START, END)] == ["10.1/a", "10.1/b"]
    assert api.requests[0].url.params["cursorMark"] == "*"
    assert api.requests[1].url.params["cursorMark"] == "c1"
    assert api.requests[0].url.params["resultType"] == "core"


def test_empty_result_makes_one_request() -> None:
    api = FakeAPI([httpx.Response(200, json=page([], 0, "*"))])
    client, _ = make_client(api)
    assert list(client.search(["Noetik"], START, END)) == []
    assert len(api.requests) == 1


def test_max_results_caps_output() -> None:
    results = [record(id=str(i), doi=f"10.1/{i}") for i in range(5)]
    api = FakeAPI([httpx.Response(200, json=page(results, 50, "c1"))])
    client, _ = make_client(api)
    assert len(list(client.search(["Roche"], START, END, max_results=3))) == 3


def test_retries_server_errors_then_succeeds() -> None:
    api = FakeAPI(
        [
            httpx.Response(503),
            httpx.ConnectError("name resolution failed"),
            httpx.Response(200, json=page([record()], 1, "c1")),
        ]
    )
    client, sleeps = make_client(api)
    assert len(list(client.search(["GSK"], START, END))) == 1
    assert sleeps == [2, 4]


def test_gives_up_after_max_attempts_as_transient() -> None:
    api = FakeAPI([httpx.Response(503) for _ in range(3)])
    client, sleeps = make_client(api, max_attempts=3)
    with pytest.raises(TransientSourceError, match="after 3 attempts"):
        list(client.search(["GSK"], START, END))
    assert len(sleeps) == 2


def test_client_errors_fail_fast_without_retry() -> None:
    api = FakeAPI([httpx.Response(400, text="bad query")])
    client, sleeps = make_client(api)
    with pytest.raises(SourceError, match="HTTP 400") as exc:
        list(client.search(["GSK"], START, END))
    assert not isinstance(exc.value, TransientSourceError)
    assert sleeps == []
