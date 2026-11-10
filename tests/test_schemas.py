from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from biotech_landscape.schemas import (
    Deal,
    DealType,
    Org,
    Paper,
    PaperSource,
    Status,
    paper_key,
    sort_key,
)


def deal_data(**overrides: Any) -> dict[str, Any]:
    """A valid deal shaped like the YAML will be (Lilly-Insilico, Mar 2026)."""
    data: dict[str, Any] = {
        "id": "lilly-insilico-2026",
        "pharma": "lilly",
        "partner": "insilico",
        "announced": "2026-03",
        "deal_types": ["asset-licence", "model-licence"],
        "layer": "Small molecule",
        "area": "Undisclosed indications",
        "terms": "Exclusive licence to preclinical oral therapeutics plus Pharma.AI",
        "upfront": {"amount": "115"},
        "headline": {"amount": "2750"},
        "status": "verified",
        "sources": [{"label": "CNBC", "url": "https://www.cnbc.com/2026/03/29/x.html"}],
        "last_checked": "2026-11-08",
    }
    data.update(overrides)
    return data


def test_valid_deal_parses() -> None:
    deal = Deal.model_validate(deal_data())
    assert deal.status is Status.VERIFIED
    assert deal.deal_types == [DealType.ASSET_LICENCE, DealType.MODEL_LICENCE]
    assert deal.upfront is not None and deal.upfront.amount == Decimal("115")
    assert deal.upfront.currency == "USD"
    assert deal.last_checked == date(2026, 11, 8)
    assert deal.partner_is_org


def test_verified_deal_needs_a_source() -> None:
    with pytest.raises(ValidationError, match="at least one source"):
        Deal.model_validate(deal_data(sources=[]))


def test_memory_deal_may_have_no_source() -> None:
    deal = Deal.model_validate(deal_data(status="memory", sources=[]))
    assert deal.status is Status.MEMORY


def test_headline_below_upfront_is_rejected() -> None:
    with pytest.raises(ValidationError, match="below the upfront"):
        Deal.model_validate(deal_data(headline={"amount": "100"}))


def test_mixed_currencies_are_rejected() -> None:
    with pytest.raises(ValidationError, match="same currency"):
        Deal.model_validate(deal_data(headline={"amount": "2750", "currency": "EUR"}))


def test_undisclosed_money_is_none_not_zero() -> None:
    deal = Deal.model_validate(deal_data(upfront=None))
    assert deal.upfront is None


def test_misspelt_field_is_rejected() -> None:
    data = deal_data()
    data["upfornt"] = data.pop("upfront")
    with pytest.raises(ValidationError, match="upfornt"):
        Deal.model_validate(data)


@pytest.mark.parametrize("bad", [[], ["asset-licence", "asset-licence"], ["merger"]])
def test_deal_types_must_be_known_unique_and_present(bad: list[str]) -> None:
    with pytest.raises(ValidationError):
        Deal.model_validate(deal_data(deal_types=bad))


def test_free_text_partner_is_allowed() -> None:
    deal = Deal.model_validate(deal_data(partner="TuneLab network (Benchling, CDD)"))
    assert not deal.partner_is_org


@pytest.mark.parametrize("value", ["2026-10", "2022", "2024-01"])
def test_partial_dates_accepted(value: str) -> None:
    assert Deal.model_validate(deal_data(announced=value)).announced == value


@pytest.mark.parametrize("value", ["Oct 2026", "2026-13", "26-10", "2026-1"])
def test_partial_dates_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        Deal.model_validate(deal_data(announced=value))


def test_sort_key_orders_partial_dates() -> None:
    values = ["2026-10", "2022", "2024-01", "2024"]
    assert sorted(values, key=sort_key) == ["2022", "2024", "2024-01", "2026-10"]


@pytest.mark.parametrize(
    ("doi", "expected"),
    [
        ("https://doi.org/10.1038/ABC", "10.1038/abc"),
        ("http://dx.doi.org/10.1038/abc", "10.1038/abc"),
        ("doi:10.1038/Abc ", "10.1038/abc"),
        ("10.1038/abc", "10.1038/abc"),
    ],
)
def test_paper_key_normalises_doi(doi: str, expected: str) -> None:
    assert paper_key(doi, PaperSource.OPENALEX, "W1") == expected


@pytest.mark.parametrize("doi", [None, "", "  "])
def test_paper_key_falls_back_to_source_id(doi: str | None) -> None:
    assert paper_key(doi, PaperSource.EUROPEPMC, "PMC456") == "europepmc:PMC456"


def test_paper_keeps_raw_affiliations() -> None:
    paper = Paper(
        key="10.1/x",
        source=PaperSource.OPENALEX,
        source_id="W1",
        title="A foundation model",
        published=date(2026, 1, 5),
        affiliations=["Genentech, South San Francisco", "ETH Zurich"],
    )
    assert paper.affiliations[0].startswith("Genentech")


@pytest.mark.parametrize("bad_id", ["Eli Lilly", "Lilly", "lilly_", "-lilly"])
def test_org_id_must_be_a_slug(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        Org(id=bad_id, name="Eli Lilly", short="Lilly", group="pharma", archetype="x")
