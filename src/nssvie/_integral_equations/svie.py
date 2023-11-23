"""Stochastic Volterra integral equation.

Class
-----
SVIE
"""
from typing import Callable

import numpy as np

from scipy.linalg import solve_triangular

from nssvie._orthogonal_functions import BlockPulseFunctions
from nssvie._integral_equations.base import IntegralEquation


class SVIE(IntegralEquation):
    """
    Stochastic Volterra integral equation

    .. math::
        :label: svie_doc

        X_t = f(t) + \\int\\limits_0^t k_1(s,t) X_s \\ ds
        + \\int\\limits_0^t k_2(s,t) X_s \\ dB_s \\qquad t \\in [0,T),

    where

    + :math:`X_t` is an unknown stochastic process,
    + :math:`B_t` the Brownian motion,
    + the last integral in :eq:`svie_doc`

      .. math::

        \\int\\limits_0^t k_2(s,t) X_s \\ dB_s

      is the Itô-integral and
    + :math:`f \\in L^2([0,T))` and
      :math:`k_1, \\ k_2 \\in L^2([0,T) \\times [0,T))` are continuous
      and square integrable functions.

    Parameters
    ----------
    function_f : Callable[[float], float]
        Function :math:`f` in :eq:`svie_doc`.
    kernel_1, kernel_2 : Callable[[float, float], float]
        The kernels :math:`k_1` and :math:`k_2` in :eq:`svie_doc`.
    endpoint : float, optional
        The right hand side of the interval :math:`[0,T)`, by default
        1.0.

    Methods
    -------
    solve_numerical
    """

    def __init__(
        self,
        function_f: Callable[[float], float],
        kernel_1: Callable[[float, float], float],
        kernel_2: Callable[[float, float], float],
        endpoint: float = 1.0
    ) -> None:
        self.function_f = function_f
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        super().__init__(endpoint=endpoint)

    def __str__(self) -> None:
        str_string = (
            f"Stochastic Volterra integral equation  on "
            f"interval [0, {self.endpoint})"
        )
        return str_string

    def __repr__(self) -> None:
        return (
            f'SVIE(f, k1, k2, interval = [0, {self.endpoint}))'
        )

    def solve_numerical(
        self,
        intervals: int,
        seed: int = None
    ) -> np.array:
        """
        Compute a numerical solution for the given stochastic Volterra
        integral equation.

        Parameters
        ----------
        intervals : int
            The number of equidistant intervals to divide :math:`[0,T)`.

        Returns
        -------
        :class:`numpy.ndarray`
            The approximate block pulse function coefficient of the
            unknown stochastic process :math:`X_t`.
        """
        # Approximate with an operational matrix of integration
        # based on block pulse functions as suggested in
        # Maleknejad et. al (2012).
        bpf = BlockPulseFunctions(
            endpoint=self.endpoint, intervals=intervals)
        matrix_m = np.transpose(
            np.eye(intervals)
            - bpf._matrix_b1(
                kernel_1=self.kernel_1
            )
            - bpf._matrix_b2(
                kernel_2=self.kernel_2,
                seed=seed
            )
        )
        func_f_coeff_vector = bpf._coeff_vector(
            function=self.function_f
        )
        approx_solution = solve_triangular(
            matrix_m,
            func_f_coeff_vector,
            lower=True
        )
        return np.insert(approx_solution, [0], self.function_f(0))


# Maybe add other methods to solve the given stochastic Volterra
# integral equation numerically based on a set of orthogonal and
# disjoint functions.
