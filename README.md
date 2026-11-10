# biotech-landscape

A refreshable map of how large pharmaceutical companies and AI-native biotechs are using AI: which partnerships they sign and on what terms, what compute and data they have access to, which modalities and disease areas they target, what they publish, how far their AI-derived molecules have progressed, and how regulators are responding.

The output is a single self-contained HTML page with an industry overview, one tab per organisation, a sortable deal ledger and cross-cutting views (deal structures, compute, value-chain focus, publications, clinical progress, FDA).

## How it works

```
data/curated/*.yaml   hand-reviewed facts: deals, profiles, focus ratings, regulatory events
        │                 every fact carries a source link, a verification status and a last-checked date
        ▼
sources/  ──► store (SQLite) ──► tagging ──► build ──► dist/*.html
Europe PMC, OpenAlex           incremental: only new papers are fetched on each run
```

- **Curated facts** change only through review. Each record is validated against a schema, so a malformed edit fails loudly instead of breaking the page.
- **Fetched data** (publications, and later trial status) is pulled incrementally into a local store, so re-running is cheap and never re-downloads what it already has.
- **The page** is generated from both, so it can be rebuilt at any time.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
cp .env.example .env          # add a free OpenAlex API key
uv run biotech-landscape refresh
```

The command-line interface is still being built; see the project history for what is currently available.

## Reuse it for your own purposes

This project is meant to be forked and adapted. You can:

- swap in your own organisations, competitors or therapeutic areas in `data/curated/`;
- change the publication window, search terms or tagging rules;
- add sources (for example ClinicalTrials.gov or your own deal database);
- use the deal ledger and structures as a starting point for your own analysis.

Treat it as a starting point rather than an authority. Facts are marked *verified* (press release, filing or two independent reports), *unverified* (single secondary source) or *memory* (not yet re-checked), and deal values are usually headline "up to" figures rather than cash paid. Check anything you plan to rely on.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
uv run pytest -q
```

## Licence

- Code: [MIT](LICENSE).
- Curated data and page content (`data/curated/`, generated pages): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reuse and adapt freely with attribution.

Publication metadata comes from [Europe PMC](https://europepmc.org/) and [OpenAlex](https://openalex.org/) and remains subject to their terms.
