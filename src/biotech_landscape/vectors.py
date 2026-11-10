"""Vector index for paper embeddings, stored in the same SQLite file as the papers.

Uses the sqlite-vec extension's `vec0` virtual table: fixed-size float32 vectors
with exact cosine nearest-neighbour search in SQL. At this project's scale
(thousands to ~100k papers) exact search takes milliseconds, so no approximate
index is needed, and keeping vectors beside the papers means one file, one backup,
transactions, and SQL joins with organisations.

Vectors are grouped by *space*: the model plus anything that changes the vectors
(e.g. an instruction prefix). Changing similarity thresholds or labels reuses the
stored vectors; changing the model or prefix starts a new space.

Requires the optional extra: `uv sync --extra embeddings`.
"""

import sqlite3
from collections.abc import Iterable, Mapping

import numpy as np
import numpy.typing as npt

from biotech_landscape.store import PaperStore

Vector = npt.NDArray[np.float32]

_SEP = "::"


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    try:
        import sqlite_vec
    except ImportError as e:
        raise RuntimeError(
            "the vector index needs the optional extra: uv sync --extra embeddings"
        ) from e
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


class VectorIndex:
    """Embeddings for one space (model + prefix), one row per paper."""

    def __init__(self, store: PaperStore, space: str, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        if _SEP in space:
            raise ValueError(f"space must not contain {_SEP!r}")
        self._conn = store.connection
        self.space = space
        self.dim = dim
        # One table per dimension: vec0 columns have a fixed size.
        self._table = f"paper_vectors_{dim}"
        _load_sqlite_vec(self._conn)
        with self._conn:
            self._conn.execute(
                f"""CREATE VIRTUAL TABLE IF NOT EXISTS {self._table} USING vec0(
                      id TEXT PRIMARY KEY,           -- space::paper_key
                      space TEXT PARTITION KEY,      -- searches stay within one space
                      +paper_key TEXT,               -- auxiliary column, for joins
                      embedding float[{dim}] distance_metric=cosine
                    )"""
            )

    def _id(self, key: str) -> str:
        return f"{self.space}{_SEP}{key}"

    def _check(self, vector: Vector) -> Vector:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"expected a {self.dim}-dim vector, got {v.shape[0]}")
        return v

    def add(self, vectors: Mapping[str, Vector]) -> None:
        """Store or replace vectors keyed by paper key, in one transaction."""
        rows = [
            (self._id(k), self.space, k, self._check(v).tobytes())
            for k, v in vectors.items()
        ]
        with self._conn:
            self._conn.executemany(
                f"DELETE FROM {self._table} WHERE id = ?", [(r[0],) for r in rows]
            )
            self._conn.executemany(
                f"INSERT INTO {self._table}(id, space, paper_key, embedding) VALUES (?, ?, ?, ?)",
                rows,
            )

    def get(self, keys: Iterable[str]) -> dict[str, Vector]:
        """Stored vectors for whichever of `keys` this space has."""
        wanted = list(keys)
        found: dict[str, Vector] = {}
        for i in range(0, len(wanted), 500):
            chunk = [self._id(k) for k in wanted[i : i + 500]]
            marks = ",".join("?" * len(chunk))
            for paper_key, blob in self._conn.execute(
                f"SELECT paper_key, embedding FROM {self._table} WHERE id IN ({marks})",
                chunk,
            ):
                found[paper_key] = np.frombuffer(blob, dtype=np.float32).copy()
        return found

    def count(self) -> int:
        (n,) = self._conn.execute(
            f"SELECT count(*) FROM {self._table} WHERE space = ?", (self.space,)
        ).fetchone()
        return int(n)

    def nearest(
        self, vector: Vector, k: int = 10, org: str | None = None
    ) -> list[tuple[str, float]]:
        """The k closest papers by cosine distance (0 = identical), optionally for one org.

        With `org`, a wider candidate set is searched and then filtered by the
        paper_orgs table, so very small orgs may return fewer than k results.
        """
        if k < 1:
            return []
        search_k = k if org is None else min(k * 20, 4096)
        candidates = self._conn.execute(
            f"""SELECT paper_key, distance FROM {self._table}
                WHERE embedding MATCH ? AND space = ? AND k = ?
                ORDER BY distance""",
            (self._check(vector).tobytes(), self.space, search_k),
        ).fetchall()
        if org is None:
            return [(key, float(d)) for key, d in candidates]
        members = {
            key
            for (key,) in self._conn.execute(
                "SELECT key FROM paper_orgs WHERE org = ?", (org,)
            )
        }
        return [(key, float(d)) for key, d in candidates if key in members][:k]
