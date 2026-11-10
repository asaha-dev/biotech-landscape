import httpx
import pytest

from biotech_landscape.sources import SourceError, TransientSourceError
from biotech_landscape.sources.http import get_json


def call(
    responses: list[httpx.Response], attempts: int = 4
) -> tuple[object, list[float]]:
    queue = list(responses)
    sleeps: list[float] = []
    http = httpx.Client(transport=httpx.MockTransport(lambda _: queue.pop(0)))
    result = get_json(
        http,
        "https://api.example/x",
        {},
        name="Example",
        max_attempts=attempts,
        sleep=sleeps.append,
    )
    return result, sleeps


def test_honours_numeric_retry_after() -> None:
    result, sleeps = call(
        [
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json={"ok": 1}),
        ]
    )
    assert result == {"ok": 1}
    assert sleeps == [7.0]


def test_retry_after_is_capped() -> None:
    _, sleeps = call(
        [
            httpx.Response(503, headers={"Retry-After": "3600"}),
            httpx.Response(200, json={}),
        ]
    )
    assert sleeps == [60.0]


def test_non_object_json_is_a_source_error() -> None:
    with pytest.raises(SourceError, match="list, not an object"):
        call([httpx.Response(200, json=[1, 2])])


def test_invalid_json_is_a_source_error() -> None:
    with pytest.raises(SourceError, match="invalid JSON"):
        call([httpx.Response(200, text="<html>maintenance</html>")])


def test_exhausted_retries_are_transient() -> None:
    with pytest.raises(TransientSourceError, match="after 2 attempts"):
        call([httpx.Response(502), httpx.Response(502)], attempts=2)
