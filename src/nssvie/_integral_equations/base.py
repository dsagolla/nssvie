"""Integral equation base class."""
from abc import ABC


class IntegralEquation(ABC):
    """Base class for integral equations."""
    # pylint: disable=too-few-public-methods

    def __init__(self, endpoint: float) -> None:
        self.endpoint = endpoint
