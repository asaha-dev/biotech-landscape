"""Clients for external publication sources.

Errors are split so callers can decide what to do: a TransientSourceError (rate
limits, server errors, dropped connections) is worth retrying on the next run; any
other SourceError (bad request, unexpected payload) needs a code or config fix.
"""


class SourceError(Exception):
    """A source request failed in a way that retrying will not fix."""


class TransientSourceError(SourceError):
    """A source request kept failing for reasons that may clear up later."""
