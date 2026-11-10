import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from biotech_landscape.schemas import Paper, PaperSource
from biotech_landscape.store import PaperStore, merge, restore, window_start

EPMC, OA = PaperSource.EUROPEPMC, PaperSource.OPENALEX


def paper(**overrides: Any) -> Paper:
    data: dict[str, Any] = {
        "key": "10.1/abc",
        "source": EPMC,
        "source_id": "PMC1",
        "title": "A virtual cell model",
        "abstract": "short",
        "published": date(2026, 3, 1),
        "venue": "Nature Methods",
        "cited_by": 3,
        "url": "https://doi.org/10.1/abc",
        "is_preprint": False,
        "affiliations": ["GSK, Stevenage"],
    }
    data.update(overrides)
    return Paper(**data)


@pytest.fixture
def store(tmp_path: Path) -> PaperStore:
    return PaperStore(tmp_path / "papers.db")


# --- merge policy (pure) ------------------------------------------------------------


def test_merge_keeps_most_informative_fields() -> None:
    a = paper(abstract="short", cited_by=3, affiliations=["GSK"], is_preprint=True)
    b = paper(
        source=OA,
        source_id="W9",
        abstract="a much longer abstract",
        cited_by=10,
        venue="NeurIPS",
        ml_venue=True,
        published=date(2026, 2, 1),
        affiliations=["GSK", "Oxford"],
        is_preprint=False,
    )
    m = merge(a, b)
    assert m.abstract == "a much longer abstract"
    assert m.cited_by == 10
    assert m.venue == "NeurIPS"  # OpenAlex venue preferred
    assert m.ml_venue
    assert m.published == date(2026, 2, 1)  # earliest
    assert m.affiliations == ["GSK", "Oxford"]  # union, order kept, no duplicates
    assert not m.is_preprint  # published version wins
    assert m.source is EPMC  # first-seen source is kept


def test_merge_rejects_different_papers() -> None:
    with pytest.raises(ValueError, match="different papers"):
        merge(paper(), paper(key="10.1/other"))


# --- incremental window (pure) ------------------------------------------------------


def test_window_starts_at_default_on_first_run() -> None:
    assert window_start(None, date(2024, 11, 1), 14) == date(2024, 11, 1)


def test_window_overlaps_last_run() -> None:
    assert window_start(date(2026, 11, 8), date(2024, 11, 1), 14) == date(2026, 10, 25)


def test_window_never_starts_before_default() -> None:
    assert window_start(date(2024, 11, 5), date(2024, 11, 1), 14) == date(2024, 11, 1)


# --- store ------------------------------------------------------------------------


def test_add_and_read_back_round_trips(store: PaperStore) -> None:
    assert store.add_papers([paper()], org="gsk") == 1
    assert store.get("10.1/abc") == paper()


def test_same_paper_twice_is_not_new(store: PaperStore) -> None:
    store.add_papers([paper()], org="gsk")
    assert store.add_papers([paper(cited_by=7)], org="gsk") == 0
    stored = store.get("10.1/abc")
    assert stored is not None and stored.cited_by == 7


def test_paper_from_two_sources_is_one_row(store: PaperStore) -> None:
    store.add_papers([paper()], org="gsk")
    store.add_papers([paper(source=OA, source_id="W9", venue="ICML")], org="gsk")
    assert store.sources_of("10.1/abc") == {EPMC, OA}
    stored = store.get("10.1/abc")
    assert stored is not None and stored.venue == "ICML"


def test_paper_can_belong_to_several_orgs(store: PaperStore) -> None:
    store.add_papers([paper()], org="gsk")
    store.add_papers([paper()], org="roche")
    assert store.orgs_of("10.1/abc") == {"gsk", "roche"}


def test_papers_for_filters_by_org_and_window(store: PaperStore) -> None:
    store.add_papers(
        [
            paper(key="k-old", published=date(2023, 1, 1)),
            paper(key="k-in", published=date(2025, 6, 1)),
        ],
        org="gsk",
    )
    store.add_papers([paper(key="k-other", published=date(2025, 6, 1))], org="roche")
    got = store.papers_for("gsk", date(2024, 11, 1), date(2026, 11, 8))
    assert [p.key for p in got] == ["k-in"]


def test_fetch_runs_track_progress(store: PaperStore) -> None:
    assert store.fetched_to("gsk", EPMC) is None
    store.record_run("gsk", EPMC, date(2026, 11, 8))
    store.record_run("gsk", EPMC, date(2026, 12, 8))
    assert store.fetched_to("gsk", EPMC) == date(2026, 12, 8)
    assert store.fetched_to("gsk", OA) is None


def test_failed_batch_writes_nothing(store: PaperStore) -> None:
    def batch() -> Any:
        yield paper(key="k1")
        raise RuntimeError("network dropped mid-batch")

    with pytest.raises(RuntimeError):
        store.add_papers(batch(), org="gsk")
    assert store.get("k1") is None


def test_store_persists_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "data" / "papers.db"
    with PaperStore(path) as s:
        s.add_papers([paper()], org="gsk")
    with PaperStore(path) as s:
        assert s.get("10.1/abc") is not None


# --- backups ------------------------------------------------------------------------


def test_backup_creates_a_consistent_copy(tmp_path: Path) -> None:
    with PaperStore(tmp_path / "papers.db") as s:
        s.add_papers([paper()], org="gsk")
        snap = s.backup(tmp_path / "backups", now=datetime(2026, 11, 9, 18, 0, 0))
    assert snap.name == "papers-20261109-180000.db"
    with PaperStore(snap) as copy:
        assert copy.get("10.1/abc") == paper()


def test_backup_keeps_only_the_newest(store: PaperStore, tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    for day in range(1, 8):
        store.backup(backups, now=datetime(2026, 11, day, 9, 0, 0), keep=3)
    names = sorted(p.name for p in backups.iterdir())
    assert names == [
        "papers-20261105-090000.db",
        "papers-20261106-090000.db",
        "papers-20261107-090000.db",
    ]


def test_backup_rotation_ignores_unrelated_files(
    store: PaperStore, tmp_path: Path
) -> None:
    backups = tmp_path / "backups"
    backups.mkdir()
    (backups / "notes.txt").write_text("keep me")
    store.backup(backups, now=datetime(2026, 11, 9), keep=1)
    store.backup(backups, now=datetime(2026, 11, 10), keep=1)
    assert (backups / "notes.txt").exists()


def test_backup_rejects_keep_below_one(store: PaperStore, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep"):
        store.backup(tmp_path, now=datetime(2026, 11, 9), keep=0)


def test_restore_brings_back_the_snapshot(tmp_path: Path) -> None:
    db = tmp_path / "papers.db"
    with PaperStore(db) as s:
        s.add_papers([paper()], org="gsk")
        snap = s.backup(tmp_path / "backups", now=datetime(2026, 11, 9))
        s.add_papers([paper(key="10.1/later")], org="gsk")
    restore(snap, db)
    with PaperStore(db) as s:
        assert s.get("10.1/abc") is not None
        assert s.get("10.1/later") is None


def test_restore_refuses_a_non_store_file(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.db"
    sqlite3.connect(str(bogus)).close()  # valid SQLite file, but not our schema
    target = tmp_path / "papers.db"
    with pytest.raises(ValueError, match="not a compatible"):
        restore(bogus, target)
    assert not target.exists()  # target untouched


def test_restore_refuses_missing_or_same_file(tmp_path: Path) -> None:
    db = tmp_path / "papers.db"
    with pytest.raises(FileNotFoundError):
        restore(tmp_path / "missing.db", db)
    PaperStore(db).close()
    with pytest.raises(ValueError, match="same file"):
        restore(db, db)


# --- citation refresh -----------------------------------------------------------------


def test_update_citations_only_raises_counts(store: PaperStore) -> None:
    store.add_papers(
        [paper(key="a", cited_by=5), paper(key="b", cited_by=9)], org="gsk"
    )
    changed = store.update_citations({"a": 8, "b": 4, "missing": 100})
    assert changed == 1
    a, b = store.get("a"), store.get("b")
    assert a is not None and a.cited_by == 8
    assert b is not None and b.cited_by == 9  # a lagging source never lowers a count
    assert store.keys() == ["a", "b"]
