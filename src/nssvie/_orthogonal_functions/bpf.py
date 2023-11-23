"""Block Pulse Functions.

Classes
-------
BlockPulseFunctions
"""
from typing import Callable

import numpy as np

from scipy.integrate import quad
from scipy.integrate import dblquad

from nssvie._orthogonal_functions.base import OrthogonalFunctions

from nssvie._process import BrownianMotion


class BlockPulseFunctions(OrthogonalFunctions):
    """
    Generate a :math:`m`-set of block pulse functions.

    Parameters
    ----------
    endpoint : float, optional
        The right endpoint of the interval :math:`[0,T)`, by default
        1.0.
    intervals : int, optional
        The number of equidistant intervals to divide :math:`[0,T)`, by
        defaul 50.

    Attributes
    ----------
    interval_width : float
        The width of one of the :math:`m` intervals.
    """

    def __init__(self, endpoint: float = 1.0, intervals: int = 20) -> None:
        super().__init__(endpoint=endpoint, intervals=intervals)
        self.interval_width = float(endpoint / intervals)

    def __str__(self) -> str:
        str_name = (
            f"{self.intervals}-Set of block pulse functions on "
            f"[0, {self.endpoint})"
        )
        return str_name

    def __repr__(self) -> str:
        repr_string = (
            f"BlockPulseFunctions("
            f"m={self.intervals}, [0, {self.endpoint}))"
        )
        return repr_string

    def _bpf_i(self, i: int, time_t: float) -> float:
        """Calculates the value of a block pulse function at :math:`t`.

        .. math::

            \\phi_i(t) = \\begin{cases} 1 & , \\ (i-1)h \\leq t < ih \\\\
                0 & , \\text{ otherwise.} \\end{cases}.

        Parameters
        ----------
        i: int
            Block pulse function no. :math:`i`.
        time_t: float
            The value :math:`t`.

        Returns
        -------
        float
            :math:`\\phi_i(t)``
        """
        if (i - 1) * self.interval_width <= time_t < i * self.interval_width:
            return 1.0
        return 0.0

    def _bpf_vector(self, time_t):
        """Calculates the value of the :math:`m`-set of block pulse
        functions at :math:`t`.

        .. math::

            \\phi(t) = (\\phi_1(t), \\ldots , \\phi_m(t))

        Parameters
        ----------
        time_t: float
            The value :math:`t`.

        Returns
        -------
        :class:`numpy.ndarray`
            :math:`\\phi(t)`
        """
        bpf_vector = np.array(
            [
                self._bpf_i(i, time_t)
                for i in range(1, self.intervals + 1)
            ]
        )
        return bpf_vector

    def _coeff_i(self, i: int, function: Callable[[float], float]) -> float:
        """Calculates the block pulse coefficient.

        .. math::

            f_i = \\frac{1}{h} \\int\\limits_0^T f(t) \\phi_i(t) dt

        For calculation of the definite integral
        :external:func:`scipy.integrate.quad` is used.

        Parameters
        ----------
        i: int
            Block pulse coefficient no. :math:`i`.
        function: callable
            The function :math:`f`.

        Returns
        -------
        float
            The block pulse function coefficient :math:`f_i` for the
            function :math:`f`.
        """
        return float(
            (1 / self.interval_width)
            * quad(
                function,
                (i-1) * self.interval_width,
                i * self.interval_width)[0]
        )

    def _coeff_vector(self, function: Callable[[float], float]):
        """Calculates the block pulse coefficient vector.

        .. math::

            F = (f_1 , \\ldots, f_m)

        Parameters
        ----------
        function: Callable
            The function :math:`f`.

        Returns
        -------
        :class:`numpy.ndarray`
            The block pulse function coefficient vector :math:`F` for
            the function :math:`f`.
        """
        return np.array(
            [
                self._coeff_i(i, function)
                for i in range(1, self.intervals + 1)
            ]
        ).T

    def _coeff_ij(
            self,
            i: int,
            j: int,
            function: Callable[[float, float], float]
    ) -> float:
        """Calculate the block pulse coefficient.

        .. math::

            k_{ij} = \\frac{1}{h^2} \\int\\limits_0^T \\int\\limits_0^T
                k(s,t) \\phi_i(s) \\phi_j(t) dt ds

        For calculation of the definite integral
        :external:func:`scipy.integrate.dblquad` is used.

        Parameters
        ----------
        i,j: int
            Block pulse coefficient no. :math:`i,j`.
        function: callable
            The function :math:`f`.

        Returns
        -------
        float
            Block pulse coefficient :math:`k_{ij`.
        """
        return dblquad(
            function,
            (i-1) * self.interval_width,
            i * self.interval_width,
            (j-1) * self.interval_width,
            j * self.interval_width,
        )[0]

    def _coeff_matrix(
        self,
        kernel: Callable[[float, float], float]
    ) -> np.array:
        """Calculate the block pulse coefficient matrix.

        .. math::

            K = (k_{ij})_{i,j = 1 , \\ldots , m}

        Parameters
        ----------
        kernel:  Callable[[float, float], float]
            The function :math:`k`.

        Returns
        -------
        :class:`numpy.ndarray`
            Block pulse coefficient matrix :math:`K`.
        """

        # Switch variables, see `scipy.integrate.dblquad`
        def kernel_var_switched(second_var, first_var):
            return kernel(first_var, second_var)

        kernel_coeff_matrix = np.array(
            [
                [
                    self._coeff_ij(i, j, kernel_var_switched)
                    for j in range(1, self.intervals + 1)
                ]
                for i in range(1, self.intervals + 1)
            ]
        )
        return self.interval_width ** (-2) * kernel_coeff_matrix

    def _operational_matrix_of_integration(self) -> np.array:
        """Calculates the operational matrix of integration.

        .. math::

            P = \\frac{h}{2} \\begin{pmatrix} 1 & 2 & 2 & \\ldots & 2
                \\\\ 0 & 1 & 2 & \\ldots & 2 \\\\ 0 & 0 & 1 & \\ldots &
                2 \\\\ \\vdots & \\vdots & \\vdots & \\ddots & \\vdots
                \\\\ 0 & 0 & 0 & \\ldots & 1 \\end{pmatrix}

        Returns
        -------
        :class:`numpy.ndarray`
            Operational matrix of integration :math:`P`.

        Notes
        -----
        For detail see `Maleknejad et. al (2012)
        <https://www.sciencedirect.com/science/article/pii/
        S0895717711005504/>`_
        """
        # Construct the diagonal part
        diagonal = np.eye(self.intervals)

        # Contruct the upper triangular part
        upper_triu = np.triu(
            2 * np.ones((self.intervals, self.intervals)), k=1)

        return self.interval_width * 0.5 * (diagonal + upper_triu)

    def _stochastic_operational_matrix_of_integration(
            self,
            seed: int = None
    ) -> np.array:
        """Calculates the stochastic operational matrix of integration.

        .. math::

            P = \\frac{h}{2} \\begin{pmatrix} B_{0.5h} & B_h & B_h &
                \\ldots & B_h \\\\ 0 & B_{1.5h} - B_h & B_{2h} - B_h &
                \\ldots & B_{2h} - B_h \\\\ 0 & 0 & B_{2.5h} - B_{2h} &
                \\ldots & B_{2h} - B_{2h} \\\\ \\vdots & \\vdots &
                \\vdots & \\ddots & \\vdots \\\\ 0 & 0 & 0 & \\ldots &
                B_{(m-0.5)h} - B_{(m-1)h} \\end{pmatrix}

        where :math:`B_t` is the Brownian motion. For sampling the
        Brownian motion
        :class:`stochastic.processes.continuous.BrownianMotion` is used.

        Parameters
        ----------
        seed : int, default 1
            The seed for the random number generator.

        Returns
        -------
        :class:`numpy.ndarray`
            Stochastical operational matrix of integration :math:`P_S`.

        Notes
        -----
        For detail see `Maleknejad et. al (2012)
        <https://www.sciencedirect.com/science/article/pii/
        S0895717711005504/>`_
        """
        if seed is None:
            seed = np.random.default_rng().integers(low=1, high=10**6)

        b_motion = BrownianMotion(endpoint=self.endpoint)

        # Generate a sample from the Brownian Motion
        bb_sample_func = b_motion.sample_as_func(seed=seed)

        # Construct the upper triangular part
        const_column = [
            bb_sample_func(i*self.interval_width) -
            bb_sample_func((i-1)*self.interval_width)
            for i in range(1, self.intervals+1)
        ]
        triu_matrix = np.triu(
            np.full((self.intervals, self.intervals),
                    np.transpose([const_column])),
            k=1
        )

        # Construct the diagonal part
        diagonal = [
            bb_sample_func((i-0.5)*self.interval_width) -
            bb_sample_func((i-1)*self.interval_width)
            for i in range(1, self.intervals+1)
        ]
        diag_matrix = np.diagflat(diagonal)

        return diag_matrix + triu_matrix

    def _matrix_b1(
        self,
        kernel_1: Callable[[float, float], float]
    ) -> np.array:
        """Generates the matrix :math:`B_1`from the article
        `Maleknejad et. al (2012)
        <https://www.sciencedirect.com/science/article/pii/
        S0895717711005504/>`_.

        Parameters
        ----------
        kernel_1: Callable[[float, float], float]

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix ``B_1``.
        """
        return np.multiply(
            self._operational_matrix_of_integration(),
            self._coeff_matrix(kernel_1),
        )

    def _matrix_b2(
        self,
        kernel_2: Callable[[float, float], float],
        seed: int = 1
    ) -> np.array:
        """Generates the matrix :math:`B_2`from the article
        `Maleknejad et. al (2012)
        <https://www.sciencedirect.com/science/article/pii/
        S0895717711005504/>`_.

        Parameters
        ----------
        kernel_2 : Callable[[float, float], float]
        seed :
        Returns
        -------
        :class:`numpy.ndarray`
            Matrix ``B_2``.
        """
        return np.multiply(
            self._stochastic_operational_matrix_of_integration(seed=seed),
            self._coeff_matrix(kernel_2),
        )
