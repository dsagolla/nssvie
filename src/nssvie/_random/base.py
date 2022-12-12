"""Random process base class.
"""
from abc import ABC
from abc import abstractmethod

from numpy.random import default_rng

from nssvie._random.random_utils import generate_times


class RandomProcess(ABC):

    rng = default_rng

    def __init__(self, T):
        self.T = T

    def _set_times(self, n):
        self._times = generate_times(self.T, n)

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def sample_at(self):
        pass
