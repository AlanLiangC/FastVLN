class StreamNavError(Exception):
    """Base error; production paths never silently substitute fake data/models."""


class CacheStateError(StreamNavError):
    pass


class EpisodeMismatchError(StreamNavError):
    pass


class DatasetIntegrityError(StreamNavError):
    pass


class KDACompatibilityError(StreamNavError):
    pass


class OracleUnavailableError(StreamNavError):
    """A reachable episode state has no valid discrete greedy-oracle action."""


class ReplayConsistencyError(RuntimeError):
    """Replay rejected on every rank before any optimizer step in this update."""
