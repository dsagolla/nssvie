"""Integral equation base class.
"""
from abc import ABC
from abc import abstractmethod


class IntegralEquation(ABC):
    def __init__(self, T: float) -> None:
        self.T = T

    @abstractmethod
    def solve_numerical(self) -> None:
        pass
