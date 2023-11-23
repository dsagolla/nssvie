"""Stochastic process base class."""
import abc
import numpy as np


class StochasticProcess():
    """Stochastic process base class."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, endpoint: float) -> None:
        """Initialize."""
        self.endpoint = endpoint

    @abc.abstractmethod
    def sample(self, steps: int, seed: int) -> np.array:
        """Sample method."""
        pass
