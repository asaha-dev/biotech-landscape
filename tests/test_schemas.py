from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from biotech_landscape.schemas import (
    ComputeCluster,
    Deal,
    DealType,
    Org,
    Ownership,
    Paper,
    PaperSource,
    RegulatoryEvent,
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
        "upfront": {"millions": "115"},
        "headline": {"millions": "2750"},
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
    assert deal.upfront is not None and deal.upfront.millions == Decimal("115")
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
        Deal.model_validate(deal_data(headline={"millions": "100"}))


def test_mixed_currencies_are_rejected() -> None:
    with pytest.raises(ValidationError, match="same currency"):
        Deal.model_validate(deal_data(headline={"millions": "2750", "currency": "EUR"}))


def test_money_in_units_not_millions_is_rejected() -> None:
    # "$115M" written out in full instead of as 115.
    with pytest.raises(ValidationError):
        Deal.model_validate(deal_data(upfront={"millions": "115000000"}))


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


def test_provenance_defaults_to_manual_and_unreviewed() -> None:
    deal = Deal.model_validate(deal_data())
    assert deal.extracted_by == "manual"
    assert not deal.reviewed


def test_extracted_by_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        Deal.model_validate(deal_data(extracted_by=""))


def compute_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "lilly-lillypod",
        "org": "lilly",
        "system": "LillyPod, B300, on-prem",
        "gpus": 1016,
        "ownership": "owned",
        "status": "verified",
        "sources": [{"label": "Lilly", "url": "https://investor.lilly.com/x"}],
        "last_checked": "2026-11-08",
    }
    data.update(overrides)
    return data


def test_valid_compute_cluster_parses() -> None:
    cluster = ComputeCluster.model_validate(compute_data())
    assert cluster.ownership is Ownership.OWNED
    assert cluster.announced is None


def test_compute_cluster_needs_gpus() -> None:
    with pytest.raises(ValidationError):
        ComputeCluster.model_validate(compute_data(gpus=0))


def test_compute_cluster_rejects_unknown_ownership() -> None:
    with pytest.raises(ValidationError):
        ComputeCluster.model_validate(compute_data(ownership="leased"))


def regulatory_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "fda-ema-ai-principles-2026",
        "announced": "2026-01",
        "regulators": ["fda", "ema"],
        "title": "FDA–EMA: 10 guiding principles of good AI practice",
        "summary": "Joint and voluntary; risk-based and human-centric.",
        "status": "verified",
        "sources": [{"label": "RAPS", "url": "https://www.raps.org/x"}],
        "last_checked": "2026-11-08",
    }
    data.update(overrides)
    return data


def test_valid_regulatory_event_parses() -> None:
    event = RegulatoryEvent.model_validate(regulatory_data())
    assert event.regulators == ["fda", "ema"]
    assert not event.milestone


def test_regulatory_event_needs_a_regulator() -> None:
    with pytest.raises(ValidationError):
        RegulatoryEvent.model_validate(regulatory_data(regulators=[]))


def test_unreviewed_event_cannot_be_a_milestone() -> None:
    with pytest.raises(ValidationError, match="reviewed"):
        RegulatoryEvent.model_validate(regulatory_data(milestone=True))


def test_reviewed_event_can_be_a_milestone() -> None:
    event = RegulatoryEvent.model_validate(
        regulatory_data(milestone=True, reviewed=True)
    )
    assert event.milestone
