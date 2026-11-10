from datetime import date

import pytest

from biotech_landscape.schemas import Paper, PaperSource
from biotech_landscape.tagging import Tagger
from biotech_landscape.tagging.keyword import KeywordTagger, compile_keywords
from biotech_landscape.tagging.labels import BIO_KEYWORDS, TAXONOMY, Dimension


@pytest.fixture(scope="module")
def tagger() -> KeywordTagger:
    return KeywordTagger()


# --- taxonomy integrity -------------------------------------------------------------


def test_every_label_has_description_and_keywords() -> None:
    for dim, labels in TAXONOMY.items():
        names = [label.name for label in labels]
        assert len(names) == len(set(names)), f"duplicate label in {dim}"
        for label in labels:
            assert label.description.strip(), label.name
            assert label.keywords, label.name
            assert all(k == k.lower() for k in label.keywords), label.name


def test_all_dimensions_covered() -> None:
    assert set(TAXONOMY) == set(Dimension)


# --- matching rules -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("keyword", "text", "hit"),
    [
        ("cell line*", "in two cell lines", True),
        ("cellular", "an excellent cellular assay", True),
        ("drug*", "excellent results", False),  # no match inside other words
        ("cancer*", "in cancers of the lung", True),  # prefix
        ("mri", "primitive features", False),  # word boundary
        ("single-cell", "single cell RNA-seq", True),  # hyphen ~ space
        ("single-cell", "single-cell RNA-seq", True),
        ("llm*", "two LLMs were compared", True),  # case-insensitive
    ],
)
def test_compile_keywords(keyword: str, text: str, hit: bool) -> None:
    assert bool(compile_keywords([keyword]).search(text)) is hit


# --- tagging behaviour --------------------------------------------------------------


def test_tags_across_dimensions(tagger: KeywordTagger) -> None:
    t = tagger.tag_text(
        "A diffusion model for antibody design using single-cell data in lung cancer"
    )
    assert t.method == ["Generative design"]
    assert t.modality == ["Antibody / protein"]
    assert t.data == ["Omics / single-cell"]
    assert t.area == ["Oncology"]
    assert t.bio_relevant
    assert t.tagger == "keyword@1"


def test_shap_is_explainability_not_interpretability(tagger: KeywordTagger) -> None:
    t = tagger.tag_text("We used SHAP values to explain the gradient boosting model.")
    assert t.method == ["Post-hoc explainability"]


def test_masked_alone_is_not_self_supervised(tagger: KeywordTagger) -> None:
    assert tagger.tag_text("a double-blind, masked trial design").method == []
    assert tagger.tag_text("a masked language model over proteins").method == [
        "Self-supervised learning"
    ]


def test_general_ml_paper_is_not_bio_relevant(tagger: KeywordTagger) -> None:
    # Real case: an imitation-learning paper by a dual-affiliated author.
    t = tagger.tag_text(
        "BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for "
        "Imitation Learning with transformer policies"
    )
    assert not t.bio_relevant


def test_trial_statistics_get_no_method_tag(tagger: KeywordTagger) -> None:
    # Real case: counted as an "ML venue" paper for Lilly, but it is statistics.
    t = tagger.tag_text(
        "From Estimands to Robust Inference of Treatment Effects in Master Protocol Trials"
    )
    assert t.method == []


def test_empty_text_gets_empty_tags(tagger: KeywordTagger) -> None:
    t = tagger.tag_text("")
    assert (t.method, t.modality, t.data, t.area, t.bio_relevant) == (
        [],
        [],
        [],
        [],
        False,
    )


def test_bio_keywords_compile() -> None:
    assert compile_keywords(BIO_KEYWORDS).search("patients with disease")


# --- interface ----------------------------------------------------------------------


def test_keyword_tagger_satisfies_protocol(tagger: KeywordTagger) -> None:
    assert isinstance(tagger, Tagger)


def test_tag_uses_title_and_abstract_keyed_by_paper(tagger: KeywordTagger) -> None:
    paper = Paper(
        key="10.1/x",
        source=PaperSource.OPENALEX,
        source_id="W1",
        title="Graph neural networks for ADMET",
        abstract="We predict toxicity of compounds.",
        published=date(2026, 1, 1),
    )
    tags = tagger.tag([paper])
    assert list(tags) == ["10.1/x"]
    assert tags["10.1/x"].method == ["Geometric / graph learning"]
    assert tags["10.1/x"].modality == ["Small molecule"]
    assert tags["10.1/x"].bio_relevant
