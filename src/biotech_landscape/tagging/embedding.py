"""Embedding tagger: zero-shot labels from semantic similarity, run locally.

Each paper (title + abstract) and each label (name + description) is embedded;
a paper gets a label when their cosine similarity clears that dimension's
threshold, keeping at most `top_k` labels per dimension. Biomedical relevance is
decided by which of two prototype descriptions the paper is closer to.

The embedding function is injected, so scoring is testable without downloading
a model. `fastembed_embedder()` provides the real one. Thresholds are starting
points to calibrate in the evaluation step.
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from fastembed import TextEmbedding

from biotech_landscape.schemas import Paper
from biotech_landscape.tagging import PaperTags
from biotech_landscape.tagging.labels import TAXONOMY, Dimension, Label

NAME = "embedding"
VERSION = "1"  # bump when scoring logic changes

Matrix = npt.NDArray[np.float32]
Embed = Callable[[Sequence[str]], Matrix]

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# fastembed defaults to the system temp dir, which is wiped on reboot; a 2.4 GB
# model should persist.
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "fastembed"
# Qwen3 embeddings are instruction-aware: the query side carries the task.
DEFAULT_PAPER_PREFIX = (
    "Instruct: Identify the machine learning methods, therapeutic modality, data "
    "type and disease area of this research paper\nQuery: "
)

BIO_PROTOTYPE = (
    "Biomedical or drug discovery research involving proteins, molecules, cells, "
    "genes, patients, clinical data or disease."
)
GENERAL_PROTOTYPE = (
    "General machine learning or computer science research with no biological, "
    "chemical or medical application, such as robotics, vision or language benchmarks."
)


def _default_thresholds() -> dict[Dimension, float]:
    return {dim: 0.45 for dim in Dimension}


def _default_top_k() -> dict[Dimension, int]:
    return {
        Dimension.METHOD: 3,
        Dimension.MODALITY: 2,
        Dimension.DATA: 2,
        Dimension.AREA: 2,
    }


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_MODEL
    paper_prefix: str = DEFAULT_PAPER_PREFIX
    label_prefix: str = ""
    thresholds: Mapping[Dimension, float] = field(default_factory=_default_thresholds)
    top_k: Mapping[Dimension, int] = field(default_factory=_default_top_k)
    batch_size: int = 32

    def fingerprint(self) -> str:
        """Short hash of everything that changes the output, for the tagger id."""
        payload: dict[str, Any] = {
            "model": self.model,
            "paper_prefix": self.paper_prefix,
            "label_prefix": self.label_prefix,
            "thresholds": {str(k): v for k, v in sorted(self.thresholds.items())},
            "top_k": {str(k): v for k, v in sorted(self.top_k.items())},
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return digest[:8]

    def vector_space(self) -> str:
        """Identity of the paper vectors: only what changes them (model + prefix).

        Thresholds and labels are deliberately excluded, so re-tuning them reuses
        stored vectors instead of re-embedding every paper.
        """
        prefix = hashlib.sha256(self.paper_prefix.encode()).hexdigest()[:8]
        return f"{self.model}|{prefix}"


class VectorCache(Protocol):
    """Where paper vectors are kept between runs (see vectors.VectorIndex)."""

    def get(self, keys: Iterable[str]) -> dict[str, Matrix]: ...

    def add(self, vectors: Mapping[str, Matrix]) -> None: ...


def _normalise(m: Matrix) -> Matrix:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def label_text(label: Label) -> str:
    return f"{label.name}: {label.description}"


class EmbeddingTagger:
    def __init__(
        self,
        embed: Embed,
        config: EmbeddingConfig | None = None,
        taxonomy: Mapping[Dimension, Sequence[Label]] = TAXONOMY,
        vectors: VectorCache | None = None,
    ) -> None:
        self._embed = embed
        self._vectors = vectors
        self.config = config or EmbeddingConfig()
        self._taxonomy = taxonomy
        self._label_vectors: dict[Dimension, Matrix] | None = None
        self._prototypes: Matrix | None = None

    @property
    def id(self) -> str:
        short = self.config.model.rsplit("/", 1)[-1]
        return f"{NAME}:{short}@{VERSION}-{self.config.fingerprint()}"

    def _labels_ready(self) -> tuple[dict[Dimension, Matrix], Matrix]:
        """Embed label and prototype texts once, on first use."""
        if self._label_vectors is None or self._prototypes is None:
            prefix = self.config.label_prefix
            self._label_vectors = {
                dim: _normalise(
                    self._embed([prefix + label_text(lab) for lab in labels])
                )
                for dim, labels in self._taxonomy.items()
                if labels  # a dimension with no labels simply yields no tags
            }
            self._prototypes = _normalise(
                self._embed([prefix + BIO_PROTOTYPE, prefix + GENERAL_PROTOTYPE])
            )
        return self._label_vectors, self._prototypes

    def _embed_papers(self, papers: Sequence[Paper]) -> Matrix:
        """Vectors for papers in order, reusing stored ones and storing new ones."""
        stored = self._vectors.get(p.key for p in papers) if self._vectors else {}
        missing = [p for p in papers if p.key not in stored]
        if missing:
            texts = [
                f"{self.config.paper_prefix}{p.title}\n{p.abstract}" for p in missing
            ]
            size = self.config.batch_size
            chunks = [
                self._embed(texts[i : i + size]) for i in range(0, len(texts), size)
            ]
            fresh = dict(zip((p.key for p in missing), np.vstack(chunks), strict=True))
            if self._vectors:
                self._vectors.add(fresh)
            stored = {**stored, **fresh}
        return _normalise(np.vstack([stored[p.key] for p in papers]))

    def tag(self, papers: Sequence[Paper]) -> dict[str, PaperTags]:
        if not papers:
            return {}
        label_vectors, prototypes = self._labels_ready()
        vectors = self._embed_papers(papers)
        bio = (vectors @ prototypes.T).argmax(axis=1) == 0

        chosen: dict[Dimension, list[list[str]]] = {}
        for dim, matrix in label_vectors.items():
            sims = vectors @ matrix.T
            names = [lab.name for lab in self._taxonomy[dim]]
            threshold = self.config.thresholds[dim]
            k = self.config.top_k[dim]
            per_paper = []
            for row in sims:
                best = np.argsort(-row)[:k]
                per_paper.append([names[i] for i in best if row[i] >= threshold])
            chosen[dim] = per_paper

        return {
            paper.key: PaperTags(
                method=chosen.get(Dimension.METHOD, [[]] * len(papers))[i],
                modality=chosen.get(Dimension.MODALITY, [[]] * len(papers))[i],
                data=chosen.get(Dimension.DATA, [[]] * len(papers))[i],
                area=chosen.get(Dimension.AREA, [[]] * len(papers))[i],
                bio_relevant=bool(bio[i]),
                tagger=self.id,
            )
            for i, paper in enumerate(papers)
        }


def fastembed_embedder(
    model: str = DEFAULT_MODEL, cache_dir: Path = DEFAULT_MODEL_DIR
) -> Embed:
    """A real embedding function backed by fastembed (ONNX, no PyTorch).

    The model is downloaded on first use and cached by fastembed.
    """
    encoder = TextEmbedding(model_name=model, cache_dir=str(cache_dir))

    def embed(texts: Sequence[str]) -> Matrix:
        return np.asarray(list(encoder.embed(list(texts))), dtype=np.float32)

    return embed
