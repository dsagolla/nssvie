"""
A python package for computing a numerical solution of stochastic
Volterra integral equations of the second kind based on block pulse
functions as suggested in `Maleknejad et. al (2012)
<https://www.sciencedirect.com/science/article/pii/S0895717711005504/>`_

In this package we implemented the suggested methods from the cited
paper. For a detailed theoretical background we refer to this paper.
"""
from ._integral_equations import StochasticVolterraIntegralEquation

__all__ = ["StochasticVolterraIntegralEquation"]

from ._version import (
	author,
	author_email,
	copyright,
	docs_copyright,
	license,
	package_name,
	url,
	version,
	description
)

__author__ = author
__author_email__ = author_email
__copyright__ = copyright
__docs_copyright = docs_copyright
__license__ = license
__package_name__ = package_name
__url__ = url
__version__ = version
__description__ = description

__all__ += [
	__package_name__,
	__version__,
	__license__,
	__url__,
	__author__,
	__copyright__,
	__description__
]

del author
del author_email
del copyright
del docs_copyright
del license
del package_name
del url
del version
del description
