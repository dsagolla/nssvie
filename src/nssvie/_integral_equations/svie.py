from numpy import eye

from numpy.linalg import solve

from nssvie._orthogonal_functions import BlockPulseFunctions

class StochasticVolterraIntegralEquation(object):
	"""
	Generates a stochastic Volterra integral equation of the second
	kind.
	
	Parameters:
	-----------
		func : callable
		k1 : callable
		k2 : callable
		interval_end : float, default=1.0
	
	Methods:
	--------
	solve_numerical
	"""
	
	def __init__(self, func, k1, k2, interval_end=1.0):
		self.__f = func
		self.__k1 = k1
		self.__k2 = k2
		self.__interval_end = float(interval_end)
	
	@property
	def f(self):
		"""Function f."""
		return self.__f
	
	@f.setter
	def f(self, new_func):
		self.__f = new_func
	
	@property
	def k1(self):
		"""Kernel k1."""
		return self.__k1
	
	@k1.setter
	def k1(self, new_k1):
		self.__k1 = new_k1
	
	@property
	def k2(self):
		"""Kernel k2."""
		return self.__k2
	
	@k2.setter
	def k2(self, new_k2):
		self.__k2 = new_k2
	
	@property
	def interval_end(self):
		"""The right hand side of ``[0,T)``."""
		return self.__interval_end
	
	@interval_end.setter
	def interval_end(self, new_interval_end):
		"""Right hand side of [0, interval_end)"""
		self.__interval_end = new_interval_end
	
	def solve_numerical(self, m=50, solve_method="bpf"):
		"""Return a numerical solution for the given linear stochastic
		Volterra integral equation of the second kind.

		Parameters:
		-----------
		m : int, default 50
			descr
		solve_method : str, default "bpf"
			descr
		

		Returns:
		--------
		np.ndrarray
			descr
		"""
		match solve_method:
      		case "bpf":
				# Approximate with an operational matrix of integration
				# based on block pulse functions as suggested in
				# Maleknejad et. al (2012).
				bpf = BlockPulseFunctions(m, self.__interval_end)
				M = (
						eye(m)
						- bpf._matrix_b1(self.__k1)
						- bpf._matrix_b2(self.__k2)
				)
				return solve(
					M.T,
					bpf._coefficient_vector(self.__f)
				)
# Maybe add other methods to solve the given stochastic Volterra
# integral equation numerically based on a set of orthogonal and
# disjoint functions.
