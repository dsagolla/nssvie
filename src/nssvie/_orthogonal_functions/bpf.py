"""Block Pulse Functions.

Submodule containing the class BlockPulseFunctions.
"""
from numpy import array, eye, triu, ones, full, transpose, diagflat, multiply

from scipy.integrate import quad, dblquad

from stochastic.processes import BrownianMotion


class BlockPulseFunctions:
    """
    Generates a ``m``-set of block pulse functions.

    Parameters
    ----------

        interval_end : float, default=1.0
            The right hand side of the interval :math:`[0,T)`
        m : int, default=50
            The number of intervals to divide :math:`[0,T)`

    Attributes
    ----------

        bandwidth : float, default=1/50
            The width of one of the ``m`` intervals.
    """

    def __init__(self, interval_end=1.0, m=50):
        self.m = m
        self.interval_end = float(interval_end)
        self.bandwidth = float(interval_end / m)

    def __str__(self):
        str_name = (
            f"{self.m}-Set of block pulse functions on "
            "the interval [0, {self.interval_end})"
        )
        return str_name

    def __repr__(self):
        repr_string = f"BlockPulseFunctions(m={self.m}, " + f"[0, {self.m}))"
        return repr_string

    def _bpf_i(self, i, t_value) -> float:
        """
        Calculates block pulse no. ``i`` function at given value.
        phi_i(t) = 1 if and only if (i-1)*h <= t < i*h
        phi_i(t) = 0 otherwise

        Parameters
        ----------
            i: int
                The number of the block pulse function.
            t_value: float
                The value for which the block pulse function should be
                evaluated.

        Returns
        -------
        float
            Value of the ``i``-th block pulse function at time
            ``t_value``.
        """
        if (i - 1) * self.bandwidth <= t_value < i * self.bandwidth:
            return float(1)
        else:
            return float(0)

    def _bpf_vector(self, t_value) -> list[float]:
        """
        phi(t) = (phi_1(t), ... , phi_m(t))

        Parameters:
        -----------
        t: float

        Returns:
        --------
        list[float]
            Values of the block pulse functions at time ``t`` as a vector.
        """
        return [self._bpf_i(i, t_value) for i in range(1, self.m + 1)]

    def _coeff_i(self, i, func) -> float:
        """Using :external:func:`scipy.integrate.quad`.

        Parameters:
        -----------
        i: int
        func: callable

        Returns:
        --------
        float
            The ``i``-th block pulse function coefficient for the function
            ``func``.
        """
        return float(
            (1 / self.bandwidth)
            * quad(func, (i - 1) * self.bandwidth, i * self.bandwidth)[0]
        )

    def _coefficient_vector(self, func) -> array:
        """
        Parameters:
        -----------
        func: callable

        Returns:
        --------
        np.array
            The block pulse function coefficient vector for the function
            ``func``.
        """
        return array([self._coeff_i(i, func) for i in range(1, self.m + 1)]).T

    def _coeff_ij(self, i, j, func) -> float:
        """
        Using :external:func:`scipy.integrate.dblquad`.

        Parameters:
        -----------
        i,j: int
        func: callable

        Returns:
        --------
        float
            Block pulse coefficient of the function ``k`` with respect
            to the ``i``-th and ``j``-th block pulse function.
        """
        return dblquad(
            func,
            (i - 1) * self.bandwidth,
            i * self.bandwidth,
            (j - 1) * self.bandwidth,
            j * self.bandwidth,
        )[0]

    def _coefficient_matrix(self, func) -> array:
        """
        Parameters:
        -----------
        func:  callable

        Returns:
        --------
        np.ndarray
            Block pulse coefficient matrix of the function ``func``.
        """

        # Switch variables, see documentation for
        # :func:`scipy.integrate.dblquad`
        def func_var_switched(second_var, first_var):
            return func(first_var, second_var)

        coeff_matrix = array(
            [
                [
                    self._coeff_ij(i, j, func_var_switched)
                    for j in range(1, self.m + 1)
                ]
                for i in range(1, self.m + 1)
            ]
        )
        return self.bandwidth ** (-2) * coeff_matrix

    def _operational_matrix_of_integration(self) -> array:
        """
        Returns:
        --------
        np.ndarray
            Operational matrix of integration.
        """
        # Construct the diagonal part
        diagonal = eye(self.refineness)

        # Contruct the upper triangular part
        upper_triu = triu(2 * ones((self.m, self.m)), k=1)

        return self.bandwidth * 0.5 * (diagonal + upper_triu)

    def _stochastic_operational_matrix_of_integration(self) -> array:
        """
        Returns:
        --------
        np.ndarray
            Stochastical operational matrix of integration.
        """
        brownian_motion = BrownianMotion(drift=0, scale=1, t=self.bandwidth)

        # Generate a sample from the Brownian Motion
        brownian_motion_sample = brownian_motion.sample_at(
            [0.5 * self.bandwidth * i for i in range(2 * self.m + 1)]
        )

        # Construct the upper triangulart part
        const_column = [
            brownian_motion_sample[2 * i] - brownian_motion_sample[2 * (i - 1)]
            for i in range(1, self.m + 1)
        ]
        triu_matrix = full((self.m, self.m), transpose([const_column]))

        # Construct the diagonal part
        diagonal = [
            (brownian_motion_sample[i] - brownian_motion_sample[i - 1])
            for i in range(1, 2 * self.refineness + 1, 2)
        ]
        diag_matrix = diagflat(diagonal)

        return diag_matrix + triu(triu_matrix, k=1)

    def _matrix_b1(self, kernel_1) -> array:
        """
        Parameters:
        -----------
        kernel_1: callable

        Returns:
        --------
        np.ndarray
            Matrix ``B_1``.
        """
        return multiply(
            self._operational_matrix_of_integration(),
            self._coefficient_matrix(kernel_1),
        )

    def _matrix_b2(self, kernel_2) -> array:
        """
        Parameter:
        ----------
        kernel_2: callable

        Returns:
        --------
        np.ndarray
            Matrix ``B_2``.
        """
        return multiply(
            self._stochastic_operational_matrix_of_integration(),
            self._coefficient_matrix(kernel_2),
        )
