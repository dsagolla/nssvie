"""Orthogonal functions base class.
"""
from abc import ABC


class OrthogonalFunctions(ABC):
    def __init__(self, T: float, m: int) -> None:
        self.T = T
        self.m = m
