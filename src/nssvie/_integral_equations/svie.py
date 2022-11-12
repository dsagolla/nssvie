"""_summary_
Returns
-------
_type_
_description_
"""
from numpy import eye

from numpy.linalg import solve

from nssvie._orthogonal_functions import BlockPulseFunctions


class StochasticVolterraIntegralEquation:
    """
    Generates a stochastic Volterra integral equation of the second
    kind.

    Parameters:
    -----------
        func : callable
        kernel_1 : callable
        kernel_2 : callable
        interval_end : float, default=1.0

    Methods:
    --------
    solve_numerical
    """

    def __init__(self, func, kernel_1, kernel_2, interval_end=1.0):
        self.func = func
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.interval_end = float(interval_end)

    def solve_numerical(self, refinement=50, solve_method="bpf"):
        """Return a numerical solution for the given linear stochastic
        Volterra integral equation of the second kind.

		Parameters:
		-----------
		refinement : int, default 50
			descr
		solve_method : str, default "bpf"
			descr

        Returns:
		--------
		np.ndrarray
			descr
		"""
        if solve_method == "bpf":
            # Approximate with an operational matrix of integration
            # based on block pulse functions as suggested in
            # Maleknejad et. al (2012).
            bpf = BlockPulseFunctions(refinement, self.interval_end)
            matrix_m = (
					eye(refinement)
					- bpf._matrix_b1(self.kernel_1)
					- bpf._matrix_b2(self.kernel_2)
			)
            return solve(
				matrix_m.T,
				bpf._coefficient_vector(self.func)
			)
# Maybe add other methods to solve the given stochastic Volterra
# integral equation numerically based on a set of orthogonal and
# disjoint functions.
