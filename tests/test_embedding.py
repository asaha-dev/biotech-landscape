"""Embedding tagger tests with a fake, deterministic embedder (no model download)."""

from datetime import date
from pathlib import Path

import pytest

from biotech_landscape.schemas import Paper, PaperSource
from biotech_landscape.store import PaperStore
from biotech_landscape.tagging import PaperTags, Tagger
from biotech_landscape.tagging.cache import tag_with_cache
from biotech_landscape.tagging.embedding import (
    EmbeddingConfig,
    EmbeddingTagger,
)
from biotech_landscape.tagging.labels import Dimension
from tests.fakes import TINY_TAXONOMY, CountingEmbedder


def paper(key: str, title: str, abstract: str = "") -> Paper:
    return Paper(
        key=key,
        source=PaperSource.OPENALEX,
        source_id=key,
        title=title,
        abstract=abstract,
        published=date(2026, 1, 1),
    )


def make(
    config: EmbeddingConfig | None = None,
) -> tuple[EmbeddingTagger, CountingEmbedder]:
    embed = CountingEmbedder()
    cfg = config or EmbeddingConfig(
        paper_prefix="", thresholds={d: 0.3 for d in Dimension}
    )
    return EmbeddingTagger(embed, cfg, taxonomy=TINY_TAXONOMY), embed


def test_assigns_closest_labels_per_dimension() -> None:
    tagger, _ = make()
    tags = tagger.tag([paper("a", "diffusion for antibody design in cancer patients")])[
        "a"
    ]
    assert tags.method == ["Generative design"]
    assert tags.modality == ["Antibody / protein"]
    assert tags.area == ["Oncology"]
    assert tags.data == []  # empty dimension in the tiny taxonomy
    assert tags.bio_relevant


def test_threshold_filters_weak_matches() -> None:
    strict = EmbeddingConfig(paper_prefix="", thresholds={d: 0.99 for d in Dimension})
    tagger, _ = make(strict)
    tags = tagger.tag([paper("a", "diffusion graph antibody cancer heart")])["a"]
    assert tags.method == [] and tags.modality == [] and tags.area == []


def test_top_k_caps_labels() -> None:
    cfg = EmbeddingConfig(
        paper_prefix="",
        thresholds={d: 0.1 for d in Dimension},
        top_k={d: 1 for d in Dimension},
    )
    tagger, _ = make(cfg)
    tags = tagger.tag([paper("a", "diffusion diffusion graph")])["a"]
    assert tags.method == ["Generative design"]  # only the single best


def test_general_ml_paper_is_not_bio_relevant() -> None:
    tagger, _ = make()
    tags = tagger.tag([paper("a", "imitation learning for robotics benchmarks")])["a"]
    assert not tags.bio_relevant


def test_labels_embedded_once_and_papers_batched() -> None:
    cfg = EmbeddingConfig(
        paper_prefix="", batch_size=2, thresholds={d: 0.3 for d in Dimension}
    )
    tagger, embed = make(cfg)
    tagger.tag([paper(str(i), "graph") for i in range(5)])
    tagger.tag([paper("x", "graph")])
    # 3 non-empty dimensions + 1 prototype call, then batches of 2,2,1 and later 1.
    assert embed.calls == [2, 2, 2, 2, 2, 2, 1, 1]


def test_empty_input_makes_no_calls() -> None:
    tagger, embed = make()
    assert tagger.tag([]) == {}
    assert embed.calls == []


def test_id_changes_with_anything_that_changes_output() -> None:
    base = EmbeddingTagger(CountingEmbedder())
    other_threshold = EmbeddingTagger(
        CountingEmbedder(), EmbeddingConfig(thresholds={d: 0.6 for d in Dimension})
    )
    other_model = EmbeddingTagger(
        CountingEmbedder(), EmbeddingConfig(model="BAAI/bge-large-en-v1.5")
    )
    assert base.id.startswith("embedding:Qwen3-Embedding-0.6B@1-")
    assert len({base.id, other_threshold.id, other_model.id}) == 3
    assert base.id == EmbeddingTagger(CountingEmbedder()).id  # stable


def test_satisfies_tagger_protocol() -> None:
    tagger, _ = make()
    assert isinstance(tagger, Tagger)


# --- cache ---------------------------------------------------------------------------


def test_tag_with_cache_only_tags_new_papers(tmp_path: Path) -> None:
    tagger, embed = make()
    papers = [paper("a", "diffusion"), paper("b", "graph")]
    with PaperStore(tmp_path / "papers.db") as store:
        store.add_papers(papers, org="roche")
        first = tag_with_cache(tagger, papers, store)
        calls_after_first = len(embed.calls)
        store.add_papers([paper("c", "antibody")], org="roche")
        second = tag_with_cache(tagger, [*papers, paper("c", "antibody")], store)
    assert first["a"].method == ["Generative design"]
    assert set(second) == {"a", "b", "c"}
    # Second run embedded only the one new paper (one batch call).
    assert len(embed.calls) == calls_after_first + 1
    assert embed.calls[-1] == 1


def test_cache_is_scoped_to_tagger_id(tmp_path: Path) -> None:
    with PaperStore(tmp_path / "papers.db") as store:
        store.add_papers([paper("a", "x")], org="gsk")
        store.put_tags({"a": PaperTags(method=["Generative design"], tagger="old@1")})
        assert store.get_tags("old@1", ["a"])["a"].method == ["Generative design"]
        assert store.get_tags("new@2", ["a"]) == {}


@pytest.mark.parametrize("n", [0, 1, 1200])
def test_get_tags_handles_many_keys(tmp_path: Path, n: int) -> None:
    with PaperStore(tmp_path / "papers.db") as store:
        assert store.get_tags("t@1", [f"k{i}" for i in range(n)]) == {}
