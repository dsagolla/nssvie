"""Block Pulse Functions.

Submodule containing the class BlockPulseFunctions.
"""
from numpy import (
	array,
	eye,
	triu,
	ones,
	full,
	transpose,
	diagflat,
	multiply
)

from scipy.integrate import (
	quad,
	dblquad
)

from stochastic.processes import BrownianMotion


class BlockPulseFunctions:
	"""
	Generates a ``m``-set of block pulse functions.

	Parameters
	----------

		interval_end : float, default=1.0
			The right hand side of the interval :math:`[0,T)`
		m : int, default=50
			The number of intervals to divie :math:`[0,T)`

    Attributes
    ----------
    
        bandwidth : float, default=1/50
            The width of one of the ``m`` intervals.
        
	"""
    
    # ----
    # Init
    # ----
    
	def __init__(self, interval_end=1.0, m=50):
		self.m = m
		self.interval_end = float(interval_end)
		self.bandwidth = float(interval_end / m)

	def __str__(self):
		return f'{self.__m}-Set of block pulse functions on the interval ' \
		        f'[0, {self.__interval_end})'

	def __repr__(self):
		return f'BlockPulseFunctions(m={self.__m}, [0, {self.__interval_end}))'

	def _bpf_i(self, i , t) -> float:
		""" Calculate block pulse function at given value.
		phi_i(t) = 1 if and only if (i-1)*h <= t < i*h
		phi_i(t) = 0 otherwise

		Parameters
		----------
            i: int
                The number of the block pulse function.
            t: float
                The value for which the block pulse function should be
                evaluated.

		Returns
		-------
		float
			Value of the ``i``-th block pulse function at time
			``t``.
		"""
		if (i - 1) * self.__bandwidth <= t < i * self.__bandwidth:
			return float(1)
		else:
			return float(0)

	def _bpf_vector(self, t) -> list[float]:
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
		return [self._bpf_i(i, t) for i in range(1, self.__m + 1)]

	def _coeff_i(self, i, func) -> float:
		""" Using :external:func:`scipy.integrate.quad`.

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
			(1 / self.__bandwidth)
			* quad(
				func,
				(i - 1) * self.__bandwidth,
				i * self.__bandwidth
			)[0]
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
		return array(
			[
				self._coeff_i(i, func)
				for i in range(1, self.__m + 1)
			]
		).T

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
			(i - 1) * self.__bandwidth,
			i * self.__bandwidth,
			(j - 1) * self.__bandwidth,
			j * self.__bandwidth)[0]

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
		def func_var_switched(t, s): return func(s, t)

		coeff_matrix = array([
			[
				self._coeff_ij(i, j, func_var_switched)
				for j in range(1, self.__m + 1)
			] for i in range(1, self.__m + 1)
		])
		return self.__bandwidth ** (-2) * coeff_matrix

	def _operational_matrix_of_integration(self) -> array:
		"""
		Returns:
		--------
		np.ndarray
			Operational matrix of integration.
		"""
		# Construct the diagonal part
		diagonal = eye(self.__m)

		# Contruct the upper triangular part
		upper_triu = triu(2 * ones((self.__m, self.__m)), k=1)

		return self.__bandwidth / 2 * (diagonal + upper_triu)

	def _stochastic_operational_matrix_of_integration(self) -> array:
		"""
		Returns:
		--------
		np.ndarray
			Stochastical operational matrix of integration.
		"""
		bb = BrownianMotion(drift=0, scale=1, t=self.__interval_end)

		# Generate a sample from the Brownian Motion
		bb_sample = bb.sample_at(
			[0.5 * self.__bandwidth * i for i in range(2 * self.__m + 1)]
		)

		# Construct the upper triangulart part
		const_column = [
			bb_sample[2 * i] - bb_sample[2 * (i - 1)] for i in
			range(1, self.__m + 1)]
		triu_matrix = full(
			(self.__m, self.__m),
			transpose(
				[const_column]
			))

		# Construct the diagonal part
		diagonal = [
			(bb_sample[i] - bb_sample[i - 1])
			for i in range(1, 2 * self.__m + 1, 2)
		]
		diag_matrix = diagflat(diagonal)

		return diag_matrix + triu(triu_matrix, k=1)

	def _matrix_b1(self, k1) -> array:
		"""
		Parameters:
		-----------
		k1: callable

		Returns:
		--------
		np.ndarray
			Matrix ``B_1``.
		"""
		return multiply(
			self._operational_matrix_of_integration(),
			self._coefficient_matrix(k1))

	def _matrix_b2(self, k2) -> array:
		"""
		Parameter:
		----------
		k2: callable

		Returns:
		--------
		np.ndarray
			Matrix ``B_2``.
		"""
		return multiply(
			self._stochastic_operational_matrix_of_integration(),
			self._coefficient_matrix(k2))