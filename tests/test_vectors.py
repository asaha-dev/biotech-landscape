from datetime import date
from pathlib import Path

import numpy as np
import pytest

from biotech_landscape.schemas import Paper, PaperSource
from biotech_landscape.store import PaperStore
from biotech_landscape.tagging.embedding import EmbeddingConfig, EmbeddingTagger
from biotech_landscape.tagging.labels import Dimension
from biotech_landscape.vectors import VectorIndex
from tests.fakes import TINY_TAXONOMY, CountingEmbedder

DIM = 4


def vec(*xs: float) -> np.ndarray:
    return np.asarray(xs, dtype=np.float32)


def paper(key: str, title: str = "graph") -> Paper:
    return Paper(
        key=key,
        source=PaperSource.OPENALEX,
        source_id=key,
        title=title,
        published=date(2026, 1, 1),
    )


@pytest.fixture
def store(tmp_path: Path) -> PaperStore:
    return PaperStore(tmp_path / "papers.db")


def test_add_get_round_trip(store: PaperStore) -> None:
    index = VectorIndex(store, space="m|p", dim=DIM)
    index.add({"a": vec(1, 0, 0, 0), "b": vec(0, 1, 0, 0)})
    got = index.get(["a", "b", "missing"])
    assert set(got) == {"a", "b"}
    np.testing.assert_array_equal(got["a"], vec(1, 0, 0, 0))
    assert index.count() == 2


def test_add_replaces_existing_vector(store: PaperStore) -> None:
    index = VectorIndex(store, space="m|p", dim=DIM)
    index.add({"a": vec(1, 0, 0, 0)})
    index.add({"a": vec(0, 0, 0, 1)})
    np.testing.assert_array_equal(index.get(["a"])["a"], vec(0, 0, 0, 1))
    assert index.count() == 1


def test_spaces_are_isolated(store: PaperStore) -> None:
    qwen = VectorIndex(store, space="qwen|x", dim=DIM)
    bge = VectorIndex(store, space="bge|x", dim=DIM)
    qwen.add({"a": vec(1, 0, 0, 0)})
    assert bge.get(["a"]) == {}
    assert bge.nearest(vec(1, 0, 0, 0)) == []


def test_nearest_orders_by_cosine_distance(store: PaperStore) -> None:
    index = VectorIndex(store, space="m|p", dim=DIM)
    index.add(
        {"same": vec(1, 0, 0, 0), "close": vec(1, 0.2, 0, 0), "far": vec(0, 0, 1, 0)}
    )
    result = index.nearest(vec(2, 0, 0, 0), k=2)  # magnitude doesn't matter for cosine
    assert [key for key, _ in result] == ["same", "close"]
    assert result[0][1] == pytest.approx(0.0, abs=1e-6)


def test_nearest_can_filter_by_org(store: PaperStore) -> None:
    store.add_papers([paper("g1"), paper("g2")], org="gsk")
    store.add_papers([paper("r1")], org="roche")
    index = VectorIndex(store, space="m|p", dim=DIM)
    index.add({"g1": vec(1, 0, 0, 0), "g2": vec(0, 1, 0, 0), "r1": vec(1, 0.01, 0, 0)})
    assert [k for k, _ in index.nearest(vec(1, 0, 0, 0), k=5, org="gsk")] == [
        "g1",
        "g2",
    ]


def test_wrong_dimension_is_rejected(store: PaperStore) -> None:
    index = VectorIndex(store, space="m|p", dim=DIM)
    with pytest.raises(ValueError, match="4-dim"):
        index.add({"a": vec(1, 0, 0)})


def test_vectors_survive_reopening_and_backup(tmp_path: Path) -> None:
    from datetime import datetime

    db = tmp_path / "papers.db"
    with PaperStore(db) as s:
        VectorIndex(s, space="m|p", dim=DIM).add({"a": vec(1, 2, 3, 4)})
        snap = s.backup(tmp_path / "backups", now=datetime(2026, 11, 10))
    for path in (db, snap):
        with PaperStore(path) as s:
            np.testing.assert_array_equal(
                VectorIndex(s, space="m|p", dim=DIM).get(["a"])["a"], vec(1, 2, 3, 4)
            )


def test_retuning_thresholds_reuses_stored_vectors(store: PaperStore) -> None:
    """The reason vectors are stored: changing thresholds must not re-embed papers."""
    papers = [paper("a", "diffusion"), paper("b", "graph")]
    store.add_papers(papers, org="roche")
    loose = EmbeddingConfig(paper_prefix="", thresholds={d: 0.1 for d in Dimension})
    strict = EmbeddingConfig(paper_prefix="", thresholds={d: 0.9 for d in Dimension})
    assert loose.vector_space() == strict.vector_space()  # same vectors
    index = VectorIndex(store, space=loose.vector_space(), dim=10)

    first = CountingEmbedder()
    EmbeddingTagger(first, loose, TINY_TAXONOMY, vectors=index).tag(papers)
    second = CountingEmbedder()
    tags = EmbeddingTagger(second, strict, TINY_TAXONOMY, vectors=index).tag(papers)

    assert 2 in first.calls  # papers were embedded once
    assert sum(second.calls) == 6 + 2  # only labels (6) and prototypes (2); no papers
    assert set(tags) == {"a", "b"}


def test_prefix_change_starts_new_space() -> None:
    a = EmbeddingConfig(paper_prefix="Instruct: x\nQuery: ")
    b = EmbeddingConfig(paper_prefix="Instruct: y\nQuery: ")
    assert a.vector_space() != b.vector_space()
