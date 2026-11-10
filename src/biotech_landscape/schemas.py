"""Typed records for curated facts and fetched papers, validated at ingestion.

Curated YAML and API responses both pass through these models.

A typo or a malformed record fails with a clear error instead of producing a broken page.
"""

import re
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

SLUG_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"

Slug = Annotated[str, Field(pattern=SLUG_PATTERN)]
# "2026-10" or "2022": deals are often announced to the month or year only, and a
# full date would invent precision we don't have.
PartialDate = Annotated[str, Field(pattern=r"^\d{4}(-(0[1-9]|1[0-2]))?$")]
CurrencyCode = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class Record(BaseModel):
    """Base for every model: reject unknown fields and make instances immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Provenance -------------------------------------------------------------------


class Status(StrEnum):
    VERIFIED = "verified"  # press release, filing or two independent reports
    UNVERIFIED = "unverified"  # single secondary source
    MEMORY = "memory"  # background knowledge, not yet re-checked


class Source(Record):
    label: str
    url: HttpUrl


class Provenance(Record):
    status: Status
    sources: list[Source] = []
    last_checked: date
    # Who or what wrote the record: "manual", or a tool id ("name@version") such as
    # "ctgov@1", "regex-money@1" or "llm-extractor@1". Trust policy lives in code.
    extracted_by: str = Field(default="manual", min_length=1)
    # False until a person has checked it. The scheduled run uses this to decide
    # what can merge automatically and what waits for review.
    reviewed: bool = False

    @model_validator(mode="after")
    def verified_needs_a_source(self) -> Self:
        if self.status is Status.VERIFIED and not self.sources:
            raise ValueError("a verified record needs at least one source")
        return self


# --- Organisations ----------------------------------------------------------------


class OrgGroup(StrEnum):
    PHARMA = "pharma"
    BIOTECH = "biotech"
    RESEARCH = "research"


class Org(Record):
    id: Slug
    name: str
    short: str
    group: OrgGroup
    archetype: str
    affiliations: list[str] = []  # phrases matched against author affiliations
    thesis: str = ""
    compute: str = ""
    data: str = ""
    modalities: list[str] = []
    areas: list[str] = []


# --- Deals ------------------------------------------------------------------------


class DealType(StrEnum):
    ASSET_LICENCE = "asset-licence"
    MODEL_LICENCE = "model-licence"
    DATA_GENERATION = "data-generation"
    DATA_LICENCE = "data-licence"
    CONSORTIUM = "consortium"
    FEDERATED = "federated"
    COMPUTE = "compute"
    ENTERPRISE_LLM = "enterprise-llm"
    ACQUISITION = "acquisition"
    EQUITY = "equity"  # minority stake alongside another structure


class Money(Record):
    # Always millions of `currency`: "$115M" is 115, "$2.75B" is 2750. The unit is in
    # the field name so no writer can omit it; the upper bound (a trillion) is a
    # plausibility check that catches a figure written out in full units.
    millions: Decimal = Field(ge=0, lt=1_000_000)
    currency: CurrencyCode = "USD"
    as_of: date | None = None  # date of the figure, for later FX conversion


class Deal(Provenance):
    id: Slug
    pharma: Slug
    partner: str  # an Org id when the partner has a profile, otherwise free text
    announced: PartialDate
    deal_types: list[DealType] = Field(min_length=1)  # first entry is the primary
    layer: str
    area: str
    terms: str = ""
    upfront: Money | None = None  # None means undisclosed, not zero
    headline: Money | None = None  # headline "up to" value including milestones

    @model_validator(mode="after")
    def check_deal(self) -> Self:
        if len(set(self.deal_types)) != len(self.deal_types):
            raise ValueError("deal_types contains duplicates")
        if self.upfront and self.headline:
            if self.upfront.currency != self.headline.currency:
                raise ValueError("upfront and headline must use the same currency")
            if self.headline.millions < self.upfront.millions:
                raise ValueError("headline value is below the upfront payment")
        return self

    @property
    def partner_is_org(self) -> bool:
        """True when partner looks like an Org id; existence is checked on load."""
        return re.fullmatch(SLUG_PATTERN, self.partner) is not None


def sort_key(announced: str) -> tuple[int, int]:
    """Order partial dates chronologically; a bare year sorts before its months."""
    year, _, month = announced.partition("-")
    return int(year), int(month) if month else 0


# --- Compute ----------------------------------------------------------------------


class Ownership(StrEnum):
    OWNED = "owned"  # on-prem or wholly owned, e.g. LillyPod
    SHARED = "shared"  # joint venture or consortium, e.g. Gefion, Sferical
    RENTED = "rented"  # cloud capacity under contract


class ComputeCluster(Provenance):
    id: Slug
    org: Slug  # the Org that uses it; must exist (checked on load)
    system: str  # e.g. "LillyPod, B300, on-prem"
    gpus: int = Field(gt=0)
    ownership: Ownership
    announced: PartialDate | None = None  # the page doesn't date clusters yet
    note: str = ""


# --- Regulation -------------------------------------------------------------------


class RegulatoryEvent(Provenance):
    id: Slug
    announced: PartialDate  # "2026-01", or "2024" when only the year is known
    regulators: list[Slug] = Field(min_length=1)  # e.g. ["fda"], ["fda", "ema"]
    title: str
    summary: str
    # An editorial judgement ("matters most for sponsors"), so only a reviewed
    # record may carry it. An extractor reports events; a person ranks them.
    milestone: bool = False

    @model_validator(mode="after")
    def milestone_needs_review(self) -> Self:
        if self.milestone and not self.reviewed:
            raise ValueError("only a reviewed event can be marked as a milestone")
        return self


# --- Papers -----------------------------------------------------------------------


class PaperSource(StrEnum):
    EUROPEPMC = "europepmc"
    OPENALEX = "openalex"


class Paper(Record):
    key: str  # see paper_key(); shared across sources so duplicates collapse
    source: PaperSource
    source_id: str
    title: str
    abstract: str = ""
    published: date
    venue: str = ""
    cited_by: int = Field(default=0, ge=0)
    url: str = ""
    is_preprint: bool = False
    ml_venue: bool = False
    affiliations: list[str] = []  # raw author affiliation strings, for auditing


_DOI_PREFIX = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:)", re.IGNORECASE)


def paper_key(doi: str | None, source: PaperSource, source_id: str) -> str:
    """Stable identity for a paper: the normalised DOI, else source:id."""
    if doi and doi.strip():
        return _DOI_PREFIX.sub("", doi.strip()).lower()
    return f"{source}:{source_id}"
