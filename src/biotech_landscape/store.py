"""Local SQLite store that makes paper fetching incremental.

Each paper is stored once (keyed by paper_key), however many sources or
organisations it arrives from. Two link tables record which sources have seen it
and which organisation's query returned it, and fetch_runs records how far each
(organisation, source) pair has been fetched, so the next run only asks for what
is new.
"""

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Self

from biotech_landscape.schemas import Paper, PaperSource
from biotech_landscape.tagging import PaperTags

# v2: paper_tags cache for model-based taggers. Older stores gain the table on open.
SCHEMA_VERSION = 2
BACKUP_PREFIX = "papers-"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    key          TEXT PRIMARY KEY,
    source       TEXT NOT NULL,     -- first source that returned it
    source_id    TEXT NOT NULL,
    title        TEXT NOT NULL,
    abstract     TEXT NOT NULL,
    published    TEXT NOT NULL,     -- ISO date
    venue        TEXT NOT NULL,
    cited_by     INTEGER NOT NULL,
    url          TEXT NOT NULL,
    is_preprint  INTEGER NOT NULL,
    ml_venue     INTEGER NOT NULL,
    affiliations TEXT NOT NULL      -- JSON list of raw strings
);
CREATE TABLE IF NOT EXISTS paper_sources (
    key       TEXT NOT NULL REFERENCES papers(key),
    source    TEXT NOT NULL,
    source_id TEXT NOT NULL,
    PRIMARY KEY (key, source)
);
CREATE TABLE IF NOT EXISTS paper_orgs (
    key    TEXT NOT NULL REFERENCES papers(key),
    org    TEXT NOT NULL,
    source TEXT NOT NULL,           -- which source's query matched this org
    PRIMARY KEY (key, org, source)
);
CREATE TABLE IF NOT EXISTS fetch_runs (
    org          TEXT NOT NULL,
    source       TEXT NOT NULL,
    fetched_to   TEXT NOT NULL,     -- ISO date the last successful run covered up to
    PRIMARY KEY (org, source)
);
CREATE TABLE IF NOT EXISTS paper_tags (
    key     TEXT NOT NULL REFERENCES papers(key),
    tagger  TEXT NOT NULL,          -- Tagger.id, e.g. "embedding:bge-small@1-3f2a"
    tags    TEXT NOT NULL,          -- PaperTags as JSON
    PRIMARY KEY (key, tagger)
);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);
CREATE INDEX IF NOT EXISTS idx_paper_orgs_org ON paper_orgs(org);
"""


def merge(existing: Paper, new: Paper) -> Paper:
    """Combine two records of the same paper, keeping the most informative fields.

    Pure, so the merge policy is testable without a database.
    """
    if existing.key != new.key:
        raise ValueError(f"cannot merge different papers: {existing.key} != {new.key}")
    openalex_venue = new.venue if new.source is PaperSource.OPENALEX else ""
    return existing.model_copy(
        update={
            "title": existing.title or new.title,
            "abstract": max(existing.abstract, new.abstract, key=len),
            "published": min(existing.published, new.published),
            # OpenAlex knows conference venues (NeurIPS, ICML...); prefer it.
            "venue": openalex_venue or existing.venue or new.venue,
            "cited_by": max(existing.cited_by, new.cited_by),
            "url": existing.url or new.url,
            # Once any source has the published version, it is no longer a preprint.
            "is_preprint": existing.is_preprint and new.is_preprint,
            "ml_venue": existing.ml_venue or new.ml_venue,
            "affiliations": list(
                dict.fromkeys(existing.affiliations + new.affiliations)
            ),
        }
    )


def window_start(
    fetched_to: date | None, default_start: date, overlap_days: int
) -> date:
    """Where the next fetch should begin.

    Indexes add papers late, so each run re-reads a short overlap; the primary key
    absorbs the duplicates. Never start before the configured window.
    """
    if fetched_to is None:
        return default_start
    return max(default_start, fetched_to - timedelta(days=overlap_days))


def restore(backup: Path, target: Path) -> None:
    """Replace the store at `target` with the snapshot at `backup`.

    Checks the snapshot is a readable store from a compatible schema version before
    touching `target`. Close any PaperStore open on `target` first.
    """
    if backup.resolve() == target.resolve():
        raise ValueError("backup and target are the same file")
    if not backup.is_file():
        raise FileNotFoundError(f"no backup at {backup}")
    source = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    try:
        (version,) = source.execute("PRAGMA user_version").fetchone()
        has_papers = source.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'papers'"
        ).fetchone()
        if not has_papers or not 1 <= version <= SCHEMA_VERSION:
            raise ValueError(f"{backup} is not a compatible paper store (v{version})")
        target.parent.mkdir(parents=True, exist_ok=True)
        dest = sqlite3.connect(str(target))
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()


class PaperStore:
    """Thin wrapper over one SQLite file. Use as a context manager."""

    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying connection, for extensions such as the vector index."""
        return self._conn

    def _migrate(self) -> None:
        (version,) = self._conn.execute("PRAGMA user_version").fetchone()
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"store schema v{version} is newer than this code (v{SCHEMA_VERSION})"
            )
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._conn:  # commits on success, rolls back on any exception
            yield self._conn

    # --- writes -------------------------------------------------------------------

    def add_papers(self, papers: Iterable[Paper], org: str) -> int:
        """Insert or merge papers returned by `org`'s query. Returns how many were new.

        All-or-nothing: if anything fails, none of the batch is written.
        """
        new = 0
        with self._transaction() as conn:
            for paper in papers:
                existing = self.get(paper.key)
                if existing is None:
                    new += 1
                    stored = paper
                else:
                    stored = merge(existing, paper)
                conn.execute(
                    """INSERT INTO papers VALUES
                       (:key, :source, :source_id, :title, :abstract, :published,
                        :venue, :cited_by, :url, :is_preprint, :ml_venue, :affiliations)
                       ON CONFLICT(key) DO UPDATE SET
                         title=excluded.title, abstract=excluded.abstract,
                         published=excluded.published, venue=excluded.venue,
                         cited_by=excluded.cited_by, url=excluded.url,
                         is_preprint=excluded.is_preprint, ml_venue=excluded.ml_venue,
                         affiliations=excluded.affiliations""",
                    _to_row(stored),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO paper_sources VALUES (?, ?, ?)",
                    (paper.key, paper.source.value, paper.source_id),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO paper_orgs VALUES (?, ?, ?)",
                    (paper.key, org, paper.source.value),
                )
        return new

    def record_run(self, org: str, source: PaperSource, fetched_to: date) -> None:
        """Mark (org, source) as fetched up to `fetched_to`.

        Call only after a fetch has fully succeeded, so a failed run is retried
        from the same point next time.
        """
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO fetch_runs VALUES (?, ?, ?)
                   ON CONFLICT(org, source) DO UPDATE SET fetched_to=excluded.fetched_to""",
                (org, source.value, fetched_to.isoformat()),
            )

    def update_citations(self, counts: Mapping[str, int]) -> int:
        """Apply refreshed citation counts; returns how many papers changed.

        Takes the higher of stored and new, matching the merge policy, so a
        source that lags behind never lowers a count.
        """
        with self._transaction() as conn:
            changed = conn.executemany(
                "UPDATE papers SET cited_by = ? WHERE key = ? AND cited_by < ?",
                [(n, key, n) for key, n in counts.items()],
            ).rowcount
        return max(changed, 0)

    def put_tags(self, tags: Mapping[str, PaperTags]) -> None:
        """Cache tags by (paper key, tagger id); re-tagging overwrites."""
        with self._transaction() as conn:
            conn.executemany(
                """INSERT INTO paper_tags VALUES (?, ?, ?)
                   ON CONFLICT(key, tagger) DO UPDATE SET tags=excluded.tags""",
                [(key, t.tagger, t.model_dump_json()) for key, t in tags.items()],
            )

    def get_tags(self, tagger_id: str, keys: Iterable[str]) -> dict[str, PaperTags]:
        """Cached tags from one tagger version, for whichever of `keys` it has tagged."""
        wanted = list(keys)
        found: dict[str, PaperTags] = {}
        # SQLite caps bound parameters per statement, so look keys up in chunks.
        for i in range(0, len(wanted), 500):
            chunk = wanted[i : i + 500]
            marks = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT key, tags FROM paper_tags WHERE tagger = ? AND key IN ({marks})",
                (tagger_id, *chunk),
            )
            for row in rows:
                found[row["key"]] = PaperTags.model_validate_json(row["tags"])
        return found

    def keys(self) -> list[str]:
        """All stored paper keys, for batch jobs such as citation refresh."""
        return [
            r["key"] for r in self._conn.execute("SELECT key FROM papers ORDER BY key")
        ]

    # --- backups ------------------------------------------------------------------

    def backup(self, backup_dir: Path, now: datetime, keep: int = 5) -> Path:
        """Snapshot the store into `backup_dir` and keep only the newest `keep`.

        Uses SQLite's online backup API, so the copy is consistent even while the
        store is open. `now` is passed in (rather than read from the clock) to keep
        this deterministic in tests.
        """
        if keep < 1:
            raise ValueError("keep must be at least 1")
        backup_dir.mkdir(parents=True, exist_ok=True)
        dest = backup_dir / f"{BACKUP_PREFIX}{now:%Y%m%d-%H%M%S}.db"
        target = sqlite3.connect(str(dest))
        try:
            self._conn.backup(target)
        finally:
            target.close()
        # Names sort chronologically, so the oldest come first.
        snapshots = sorted(backup_dir.glob(f"{BACKUP_PREFIX}*.db"))
        for old in snapshots[:-keep]:
            old.unlink()
        return dest

    # --- reads --------------------------------------------------------------------

    def get(self, key: str) -> Paper | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE key = ?", (key,)
        ).fetchone()
        return _from_row(row) if row else None

    def fetched_to(self, org: str, source: PaperSource) -> date | None:
        row = self._conn.execute(
            "SELECT fetched_to FROM fetch_runs WHERE org = ? AND source = ?",
            (org, source.value),
        ).fetchone()
        return date.fromisoformat(row["fetched_to"]) if row else None

    def papers_for(self, org: str, start: date, end: date) -> list[Paper]:
        """Papers linked to `org` and published in [start, end], newest first."""
        rows = self._conn.execute(
            """SELECT DISTINCT p.* FROM papers p
               JOIN paper_orgs o ON o.key = p.key
               WHERE o.org = ? AND p.published BETWEEN ? AND ?
               ORDER BY p.published DESC, p.key""",
            (org, start.isoformat(), end.isoformat()),
        ).fetchall()
        return [_from_row(r) for r in rows]

    def sources_of(self, key: str) -> set[PaperSource]:
        rows = self._conn.execute(
            "SELECT source FROM paper_sources WHERE key = ?", (key,)
        ).fetchall()
        return {PaperSource(r["source"]) for r in rows}

    def orgs_of(self, key: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT org FROM paper_orgs WHERE key = ?", (key,)
        ).fetchall()
        return {r["org"] for r in rows}


def _to_row(p: Paper) -> dict[str, object]:
    return {
        "key": p.key,
        "source": p.source.value,
        "source_id": p.source_id,
        "title": p.title,
        "abstract": p.abstract,
        "published": p.published.isoformat(),
        "venue": p.venue,
        "cited_by": p.cited_by,
        "url": p.url,
        "is_preprint": int(p.is_preprint),
        "ml_venue": int(p.ml_venue),
        "affiliations": json.dumps(p.affiliations),
    }


def _from_row(row: sqlite3.Row) -> Paper:
    return Paper(
        key=row["key"],
        source=PaperSource(row["source"]),
        source_id=row["source_id"],
        title=row["title"],
        abstract=row["abstract"],
        published=date.fromisoformat(row["published"]),
        venue=row["venue"],
        cited_by=row["cited_by"],
        url=row["url"],
        is_preprint=bool(row["is_preprint"]),
        ml_venue=bool(row["ml_venue"]),
        affiliations=json.loads(row["affiliations"]),
    )
