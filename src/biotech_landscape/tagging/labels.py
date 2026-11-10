"""The label taxonomy shared by every tagger.

Each label has a description (used by the embedding and Claude taggers, and shown
on the page) and keywords (used by the keyword tagger).

Keyword syntax: lower-case; matched on word boundaries; a trailing "*" matches
any word ending ("cancer*" matches "cancers", "cancerous"). Hyphens and spaces
in a keyword match either a hyphen or whitespace ("single-cell" ~ "single cell").
"""

from dataclasses import dataclass
from enum import StrEnum


class Dimension(StrEnum):
    METHOD = "method"
    MODALITY = "modality"
    DATA = "data"
    AREA = "area"


@dataclass(frozen=True)
class Label:
    name: str
    description: str
    keywords: tuple[str, ...]


TAXONOMY: dict[Dimension, tuple[Label, ...]] = {
    Dimension.METHOD: (
        Label(
            "LLMs & agents",
            "Develops or applies large language models, LLM-based agents or "
            "retrieval-augmented generation.",
            (
                "large language model*",
                "llm*",
                "gpt*",
                "agentic",
                "ai agent*",
                "multi-agent",
                "retrieval-augmented",
                "chatbot*",
            ),
        ),
        Label(
            "Generative design",
            "Generates new molecules, proteins or sequences with generative models "
            "such as diffusion, flow matching or VAEs.",
            (
                "diffusion model*",
                "flow matching",
                "generative model*",
                "generative ai",
                "de novo design",
                "inverse folding",
                "variational autoencoder*",
            ),
        ),
        Label(
            "Protein / genomic language models",
            "Language models trained on protein, DNA or RNA sequences.",
            (
                "protein language model*",
                "plm",
                "plms",
                "esm*",
                "dna language model*",
                "genomic language model*",
                "nucleotide transformer*",
            ),
        ),
        Label(
            "Geometric / graph learning",
            "Graph neural networks or geometric deep learning on molecules and "
            "structures, including equivariant models.",
            (
                "graph neural network*",
                "gnn*",
                "equivariant",
                "message passing",
                "geometric deep learning",
            ),
        ),
        Label(
            "Biological foundation models",
            "Large pre-trained models of cells, tissues, pathology or biology "
            "intended for many downstream tasks, including virtual cells.",
            (
                "foundation model*",
                "virtual cell*",
                "pretrained model*",
                "pre-trained model*",
            ),
        ),
        Label(
            "Multimodal learning",
            "Models that combine several data modalities, e.g. images with omics or text.",
            ("multimodal", "multi-modal", "vision-language"),
        ),
        Label(
            "Self-supervised learning",
            "Learns representations without labels, e.g. contrastive or masked modelling.",
            (
                "self-supervised",
                "contrastive learning",
                "masked language model*",
                "masked autoencoder*",
            ),
        ),
        Label(
            "RL & active learning",
            "Reinforcement learning, active learning, Bayesian optimisation or "
            "closed-loop / lab-in-the-loop experimentation.",
            (
                "reinforcement learning",
                "active learning",
                "bayesian optimi*",
                "closed-loop",
                "lab-in-the-loop",
            ),
        ),
        Label(
            "Federated / privacy-preserving",
            "Trains models across organisations without pooling raw data.",
            (
                "federated learning",
                "federated",
                "privacy-preserving",
                "differential privacy",
            ),
        ),
        Label(
            "Interpretability research",
            "Studies what models have learned internally, e.g. mechanistic "
            "interpretability or sparse autoencoders.",
            ("mechanistic interpretab*", "sparse autoencoder*", "interpretability"),
        ),
        Label(
            "Post-hoc explainability",
            "Uses attribution tools such as SHAP or saliency maps to explain a "
            "model's predictions.",
            ("shap", "shapley", "explainab*", "saliency map*", "feature attribution*"),
        ),
    ),
    Dimension.MODALITY: (
        Label(
            "Small molecule",
            "Small-molecule drugs, chemistry, ADMET or docking.",
            (
                "small molecule*",
                "small-molecule",
                "ligand*",
                "admet",
                "docking",
                "medicinal chemistry",
            ),
        ),
        Label(
            "Antibody / protein",
            "Antibodies, binders, nanobodies or designed proteins as therapeutics.",
            (
                "antibod*",
                "nanobod*",
                "binder*",
                "protein design",
                # Not "biologic*": that also matches "biological(ly)".
                "biologics",
                "biologic drug*",
                "biologic therap*",
                "protein therapeutic*",
            ),
        ),
        Label("Peptide", "Peptide therapeutics.", ("peptide*",)),
        Label(
            "Nucleic acid",
            "mRNA, siRNA, antisense or gene-editing modalities.",
            (
                "mrna",
                "sirna",
                "oligonucleotide*",
                "antisense",
                "crispr",
                "gene editing",
            ),
        ),
        Label("Vaccine", "Vaccines and immunogen design.", ("vaccin*", "immunogen*")),
        Label(
            "Cell / gene therapy",
            "CAR-T, other cell therapies or viral gene therapy.",
            ("car-t", "cell therap*", "gene therap*", "aav"),
        ),
    ),
    Dimension.DATA: (
        Label(
            "Imaging / pathology",
            "Histopathology, microscopy or radiology images.",
            (
                "histopatholog*",
                "pathology",
                "whole slide",
                "whole-slide",
                "microscopy",
                "radiolog*",
                "mri",
                "ct scan*",
                "imaging",
            ),
        ),
        Label(
            "Omics / single-cell",
            "Transcriptomics, proteomics, spatial or single-cell data, including "
            "perturbation screens.",
            (
                "single-cell",
                "transcriptom*",
                "proteom*",
                "spatial transcriptom*",
                "multi-omic*",
                "multiomic*",
                "perturb*",
            ),
        ),
        Label(
            "Genetics",
            "Human genetics: GWAS, variants, polygenic scores or biobanks.",
            ("gwas", "genetic*", "genome-wide", "variant*", "polygenic", "biobank*"),
        ),
        Label(
            "Clinical / real-world data",
            "Electronic health records, claims, real-world or clinical-trial data.",
            (
                "electronic health record*",
                "ehr*",
                "real-world",
                "claims data",
                "clinical trial*",
            ),
        ),
        Label(
            "Text / literature",
            "Scientific text, literature or clinical notes as input data.",
            ("literature", "clinical notes", "natural language", "text mining"),
        ),
    ),
    Dimension.AREA: (
        Label(
            "Oncology",
            "Cancer.",
            (
                "cancer*",
                "tumor*",
                "tumour*",
                "oncolog*",
                "carcinoma*",
                "lymphoma*",
                "leukemia*",
                "leukaemia*",
            ),
        ),
        Label(
            "Cardiometabolic",
            "Obesity, diabetes, cardiovascular, liver (MASH) or kidney disease.",
            (
                "obesity",
                "diabet*",
                "cardiovascular",
                "cardiac",
                "heart failure",
                "masld",
                "mash",
                "nash",
                "kidney",
                "metabolic",
            ),
        ),
        Label(
            "Immunology",
            "Autoimmune and inflammatory disease.",
            (
                "autoimmun*",
                "inflammat*",
                "arthritis",
                "lupus",
                "psoriasis",
                "atopic",
                "immunolog*",
            ),
        ),
        Label(
            "Neuroscience",
            "Neurological and psychiatric disease.",
            (
                "alzheimer*",
                "parkinson*",
                "neurodegenerat*",
                "neurolog*",
                "depression",
                "schizophren*",
                "als",
            ),
        ),
        Label(
            "Respiratory",
            "Asthma, COPD and other lung disease.",
            ("asthma", "copd", "pulmonary", "respiratory", "lung fibrosis"),
        ),
        Label(
            "Infectious disease",
            "Bacterial, viral and other infections.",
            (
                "infectio*",
                "viral",
                "virus*",
                "bacteri*",
                "antimicrobial*",
                "hiv",
                "covid*",
                "sars-cov-2",
            ),
        ),
    ),
}

# Signals that a paper concerns biomedicine at all; used to separate drug-discovery
# work from general ML papers by dual-affiliated authors.
BIO_KEYWORDS: tuple[str, ...] = (
    "drug*",
    "protein*",
    "molecul*",
    "compound*",
    "ligand*",
    "antibod*",
    "peptide*",
    "gene",
    "genes",
    "genom*",
    "omic*",
    "transcriptom*",
    "cell line*",
    "single-cell",
    "cellular",
    "clinical",
    "patient*",
    "disease*",
    "cancer*",
    "tumor*",
    "tumour*",
    "pathology",
    "biomarker*",
    "assay*",
    "toxic*",
    "pharma*",
    "therapeut*",
    "medical",
    "medicine",
    "health*",
    "biolog*",
)


def labels(dimension: Dimension) -> tuple[Label, ...]:
    return TAXONOMY[dimension]
