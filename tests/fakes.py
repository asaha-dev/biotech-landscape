"""Shared test doubles for the tagging tests (no model downloads)."""

import re
from collections.abc import Sequence

import numpy as np

from biotech_landscape.tagging.embedding import Matrix
from biotech_landscape.tagging.labels import Dimension, Label

# A tiny vocabulary: each text becomes a vector of word counts over it, so texts
# that share words are similar. Prototype words decide bio relevance.
VOCAB = [
    "diffusion",
    "graph",
    "antibody",
    "molecule",
    "cancer",
    "heart",
    "biomedical",
    "patients",
    "robotics",
    "benchmarks",
]

TINY_TAXONOMY: dict[Dimension, tuple[Label, ...]] = {
    Dimension.METHOD: (
        Label("Generative design", "diffusion diffusion", ()),
        Label("Geometric / graph learning", "graph graph", ()),
    ),
    Dimension.MODALITY: (
        Label("Antibody / protein", "antibody antibody", ()),
        Label("Small molecule", "molecule molecule", ()),
    ),
    Dimension.DATA: (),
    Dimension.AREA: (
        Label("Oncology", "cancer cancer", ()),
        Label("Cardiometabolic", "heart heart", ()),
    ),
}


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, texts: Sequence[str]) -> Matrix:
        self.calls.append(len(texts))
        rows = []
        for text in texts:
            words = re.findall(r"[a-z]+", text.lower())
            vec = [float(words.count(w)) for w in VOCAB]
            # Prototype texts mention these concepts; map them onto vocab words.
            if "biomedical" in text.lower():
                vec[VOCAB.index("biomedical")] += 1
                vec[VOCAB.index("patients")] += 1
            if "robotics" in text.lower():
                vec[VOCAB.index("robotics")] += 1
                vec[VOCAB.index("benchmarks")] += 1
            rows.append(vec)
        return np.asarray(rows, dtype=np.float32)
