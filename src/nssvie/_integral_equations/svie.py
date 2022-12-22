"""Stochastic Volterra integral equation.

Class
-----
SVIE
"""
import numpy as np

from scipy.linalg import solve_triangular

from nssvie._orthogonal_functions import BlockPulseFunctions
from nssvie._integral_equations.base import IntegralEquation

from typing import Callable


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
    f : Callable[[float], float]
        Function :math:`f` in :eq:`svie_doc`.
    kernel_1, kernel_2 : Callable[[float, float], float]
        The kernels :math:`k_1` and :math:`k_2` in :eq:`svie_doc`.
    T : float, optional
        The right hand side of the interval :math:`[0,T)`, by default
        1.0.

    Methods
    -------
    solve_numerical
    """

    def __init__(
        self,
        f: Callable[[float], float],
        kernel_1: Callable[[float, float], float],
        kernel_2: Callable[[float, float], float],
        T: float = 1.0
    ) -> None:
        self.f = f
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        super().__init__(T=T)

    def __str__(self) -> None:
        return (
            f'Stochastic Volterra integral equation on interval [0, {self.T})'
        )

    def __repr__(self) -> None:
        return (
            f'SVIE(f, k1, k2, interval = [0, {self.T}))'
        )

    def solve_numerical(
        self,
        m: int,
        solve_method: str = 'bpf'
    ) -> np.array:
        """
        Compute a numerical solution for the given stochastic Volterra
        integral equation.

        Parameters
        ----------
        m : int
            The number of equidistant intervals to divide :math:`[0,T)`.
        solve_method : str, optional
            If ``solve_methods='bpf'`` an algorithm presented in
            `Maleknejad et. al (2012)
            <https://www.sciencedirect.com/science/
            article/pii/S0895717711005504/>`_ is used which relies on an
            operational matrix of integration based on block pulse
            functions. For the solution of :math:`MX=F`, where :math:`M`
            is a triangular matrix
            :func:`scipy.linalg.solve_triangular` is used, by default
            ``bpf``.

        Returns
        -------
        :class:`numpy.ndarray`
            The approximate block pulse function coefficient of the
            unknown stochastic process :math:`X_t`.
        """
        if solve_method == 'bpf':
            # Approximate with an operational matrix of integration
            # based on block pulse functions as suggested in
            # Maleknejad et. al (2012).
            bpf = BlockPulseFunctions(T=self.T, m=m)
            M = np.transpose(
                np.eye(m)
                - bpf._matrix_b1(kernel_1=self.kernel_1)
                - bpf._matrix_b2(kernel_2=self.kernel_2)
            )
            F = bpf._coeff_vector(f=self.f)
            return solve_triangular(M, F, lower=True)


# Maybe add other methods to solve the given stochastic Volterra
# integral equation numerically based on a set of orthogonal and
# disjoint functions.
