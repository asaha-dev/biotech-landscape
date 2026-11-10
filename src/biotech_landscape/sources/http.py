"""Shared GET-with-retry for source clients.

Retries rate limits (429), server errors (5xx) and transport failures (DNS,
timeouts, dropped connections) with capped exponential backoff, honouring a
numeric Retry-After header. Anything else fails immediately as a SourceError.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from biotech_landscape.sources import SourceError, TransientSourceError

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
USER_AGENT = "biotech-landscape (+https://github.com/asaha-dev/biotech-landscape)"
MAX_DELAY_S = 60.0


def default_client(timeout_s: float = 90.0) -> httpx.Client:
    """An httpx client with sensible timeouts; the caller owns and closes it."""
    return httpx.Client(timeout=timeout_s, headers={"User-Agent": USER_AGENT})


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("Retry-After", "")
    try:
        return min(float(value), MAX_DELAY_S)
    except ValueError:
        return None


def get_json(
    http: httpx.Client,
    url: str,
    params: Mapping[str, Any],
    *,
    name: str,
    max_attempts: int,
    sleep: Callable[[float], None],
    hints: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """GET `url` and return the JSON object, retrying transient failures.

    `hints` maps permanent status codes to actionable advice, e.g. what to do
    about a 401.
    """
    reason = "no attempt made"
    for attempt in range(1, max_attempts + 1):
        delay: float | None = None
        try:
            resp = http.get(url, params=params)
        except httpx.TransportError as e:
            reason = f"{type(e).__name__}: {e}"
        else:
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError as e:
                    raise SourceError(f"{name} returned invalid JSON: {e}") from e
                if not isinstance(payload, dict):
                    raise SourceError(
                        f"{name} returned {type(payload).__name__}, not an object"
                    )
                return payload
            if resp.status_code not in RETRY_STATUS:
                hint = (hints or {}).get(resp.status_code, "")
                raise SourceError(
                    f"{name} returned HTTP {resp.status_code}"
                    + (f": {hint}" if hint else f": {resp.text[:200]}")
                )
            reason = f"HTTP {resp.status_code}"
            delay = _retry_after(resp)
        if attempt == max_attempts:
            break
        wait = delay if delay is not None else min(2.0**attempt, MAX_DELAY_S)
        log.info("%s %s; retry %d in %ss", name, reason, attempt, wait)
        sleep(wait)
    raise TransientSourceError(
        f"{name} still failing after {max_attempts} attempts ({reason})"
    )
