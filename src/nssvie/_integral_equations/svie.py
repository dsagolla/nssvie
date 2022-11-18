"""_summary_
Returns
-------
_type_
    _description_
"""
from numpy import eye

from numpy.linalg import solve

from nssvie._orthogonal_functions import BlockPulseFunctions


class SVIE:
    """
    Generate a stochastic Volterra integral equation

    .. math::
        :label: svie


        X_t = f(t) + \\int\\limits_0^t k_1(s,t) X_s \\ ds
        + \\int\\limits_0^t k_2(s,t) X_s \\ dB_s \\qquad t \\in [0,T),

    where :math:`X_t` is an unknown stochastic process, :math:`B_t` the
    Brownian motion,

    .. math::
        :label: ito-int

        \\int\\limits_0^t k_2(s,t) X_s \\ dB_s

    is the Itô-integral and :math:`f \\in L^2([0,T))` and
    :math:`k_1, \\ k_2 \\in L^2([0,T) \\times [0,T))`.

    Parameters
    ----------
    func : callable
        Function :math:`f` in :eq:`svie`
    kernel_1, kernel_2 : callable
        The kernels :math:`k_1` and :math:`k_2` in :eq:`svie`
    T : float, default 1.0
        The right hand side of the interval :math:`[0,T)`.

    Methods
    -------
    solve_numerical
    """
    def __init__(self, f, kernel_1, kernel_2, T=1.0):
        self.f = f
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.T = float(T)

    def solve_numerical(self, m=50, solve_method="bpf"):
        """
        Compute a numerical solution of the given stochastic Volterra
        integral equation.

        Parameters
        ----------
        m : int, default 50
            The number of intervals
        solve_method : str, default bpf
            If ``solve_methods="bpf"`` an algorithm presented in
            `Maleknejad et. al (2012)
            <https://www.sciencedirect.com/science/
            article/pii/S0895717711005504/>`_ is used which relies on an
            operational matrix of integration based on block pulse
            functions.

        Returns
        -------
        :class:`numpy.ndarray`
            The approximate block pulse function coefficient of the
            unknown stochastic process :math:`X_t`.
        """
        if solve_method == "bpf":
            # Approximate with an operational matrix of integration
            # based on block pulse functions as suggested in
            # Maleknejad et. al (2012).
            bpf = BlockPulseFunctions(m, self.interval_end)
            matrix_m = (
                eye(m)
                - bpf._matrix_b1(self.kernel_1)
                - bpf._matrix_b2(self.kernel_2)
            )
            return solve(matrix_m.T, bpf._coefficient_vector(self.func))


# Maybe add other methods to solve the given stochastic Volterra
# integral equation numerically based on a set of orthogonal and
# disjoint functions.
