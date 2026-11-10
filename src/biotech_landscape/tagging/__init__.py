"""Paper tagging: one label taxonomy, several interchangeable taggers.

Every tagger takes papers and returns PaperTags drawn from the same labels
(see labels.py), so results can be swapped, combined or compared directly.
The keyword tagger is the dependency-free default; embedding and Claude taggers
are optional extras.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import Field

from biotech_landscape.schemas import Paper, Record


class PaperTags(Record):
    """Labels for one paper, one list per dimension, plus a relevance flag."""

    method: list[str] = []
    modality: list[str] = []
    data: list[str] = []
    area: list[str] = []
    bio_relevant: bool = False  # does the paper concern drug discovery / biomedicine?
    tagger: str = Field(min_length=1)  # "name@version" that produced these tags


@runtime_checkable
class Tagger(Protocol):
    """Anything that can label papers. Batch-shaped so model taggers can batch calls."""

    @property
    def id(self) -> str:
        """Stable "name@version"; changing it marks previously cached tags stale."""
        ...

    def tag(self, papers: Sequence[Paper]) -> dict[str, PaperTags]:
        """Return tags keyed by paper.key, for every paper given."""
        ...
