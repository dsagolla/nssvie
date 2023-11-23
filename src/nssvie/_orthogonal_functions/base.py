"""Orthogonal functions base class."""
from abc import ABC


class OrthogonalFunctions(ABC):
    """Orthogonal functions base class."""
    # pylint: disable=too-few-public-methods

    def __init__(self, endpoint: float, intervals: int) -> None:
        self.endpoint = endpoint
        self.intervals = intervals
