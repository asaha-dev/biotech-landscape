"""Run a tagger only on papers it hasn't tagged yet.

Model-based taggers (embedding, Claude) are slow or cost money, so their tags are
cached in the store under the tagger's id. A new id (different model, prompt,
thresholds or version) naturally misses the cache and re-tags everything.
"""

from collections.abc import Sequence

from biotech_landscape.schemas import Paper
from biotech_landscape.store import PaperStore
from biotech_landscape.tagging import PaperTags, Tagger


def tag_with_cache(
    tagger: Tagger, papers: Sequence[Paper], store: PaperStore
) -> dict[str, PaperTags]:
    """Tags for every paper: cached ones from the store, the rest freshly computed."""
    cached = store.get_tags(tagger.id, (p.key for p in papers))
    missing = [p for p in papers if p.key not in cached]
    fresh = tagger.tag(missing) if missing else {}
    if fresh:
        store.put_tags(fresh)
    return {**cached, **fresh}
