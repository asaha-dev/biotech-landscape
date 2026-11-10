"""Keyword tagger: the dependency-free default.

Deterministic and instant, so it runs at build time on every paper. Its known
weakness is context: it can't tell "develops an LLM" from "mentions LLMs". The
embedding and Claude taggers exist to address that, measured against this one.
"""

import re
from collections.abc import Iterable, Sequence

from biotech_landscape.schemas import Paper
from biotech_landscape.tagging import PaperTags
from biotech_landscape.tagging.labels import BIO_KEYWORDS, TAXONOMY, Dimension

NAME = "keyword"
VERSION = "2"  # bump when keywords or matching rules change


def compile_keywords(keywords: Iterable[str]) -> re.Pattern[str]:
    """One case-insensitive regex for a keyword list, matched on word boundaries."""
    parts = []
    for kw in keywords:
        prefix = kw.endswith("*")
        words = re.split(r"[\s-]+", kw.rstrip("*"))
        body = r"[\s-]+".join(re.escape(w) for w in words)
        parts.append(rf"\b{body}" + (r"\w*" if prefix else r"\b"))
    return re.compile("|".join(parts), re.IGNORECASE)


class KeywordTagger:
    def __init__(self) -> None:
        self._patterns = {
            dim: [(label.name, compile_keywords(label.keywords)) for label in labels]
            for dim, labels in TAXONOMY.items()
        }
        self._bio = compile_keywords(BIO_KEYWORDS)

    @property
    def id(self) -> str:
        return f"{NAME}@{VERSION}"

    def tag_text(self, text: str) -> PaperTags:
        """Tag free text; exposed separately so rules are easy to test."""
        found = {
            dim: [name for name, pattern in patterns if pattern.search(text)]
            for dim, patterns in self._patterns.items()
        }
        return PaperTags(
            method=found[Dimension.METHOD],
            modality=found[Dimension.MODALITY],
            data=found[Dimension.DATA],
            area=found[Dimension.AREA],
            bio_relevant=bool(self._bio.search(text)),
            tagger=self.id,
        )

    def tag(self, papers: Sequence[Paper]) -> dict[str, PaperTags]:
        return {p.key: self.tag_text(f"{p.title}\n{p.abstract}") for p in papers}
