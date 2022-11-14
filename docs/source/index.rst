|project| documentation
***********************

|pypi| |pyversions| |docs| |licence|

.. |stochastic-volterra-integral-equation| image:: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/images/stochastic-volterra-integral-equation.svg
.. |X-t| image:: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/images/X-t.svg
.. |f| image:: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/images/f.svg
.. |k-1-k-2| image:: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/images/k-1-k-2.svg
.. |B-t| image:: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/images/B-t.svg
.. |ito-integral| image:: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/images/ito-integral.svg
.. |example-1-eq| image:: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/images/example-1-eq.svg
.. |example-1-f| image:: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/images/example-1-f.svg
.. |example-1-k-1| image:: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/images/example-1-k-1.svg
.. |example-1-k-2| image:: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/images/example-1-k-2.svg

.. toctree::
   :maxdepth: 1
   :caption: Documentation
   :hidden:

   Home <self>
   Theory <theory/index>
   API Reference <api>

.. .. include:: ../../README.rst
..    :start-after: .. include_after_this_line

A python package for computing a numerical solution of stochastic Volterra 
integral equations.

.. grid:: 4

   .. grid-item-card:: GitHub
      :link: https://github.com/dsagolla/nssvie
      :text-align: center
      :class-card: custom-card-link

   .. grid-item-card:: PyPI
      :link: https://pypi.org/project/nssvie/
      :text-align: center
      :class-card: custom-card-link

   .. grid-item-card:: Theory
      :link: theory/index.html
      :text-align: center
      :class-card: custom-card-link

   .. grid-item-card:: API reference
      :link: api
      :link-type: ref
      :text-align: center
      :class-card: custom-card-link

Overview
========

|project| is a python package for computing a numerical solution of stochastic Volterra 
integral equations of the second kind

|stochastic-volterra-integral-equation|

where

+ |X-t| is an unknown process,
+ |f| is a continuous function,
+ |k-1-k-2| are continuous and square integrable functions,
+ |B-t| is the Brownian Motion and
+ |ito-integral| is the Itô-integral

by a stochastic operational matrix based on block
pulse functions as suggested in ``Maleknejad et. al (2012)`` [1]_.

Install
=======

Install using either of the following two methods.

1. Install from PyPi
~~~~~~~~~~~~~~~~~~~~

|pypi| |pyversions| |format| 

The ``nssvie`` package is available on `PyPi <https://pypi.org/project/nssvie/>`__ and can be installed using ``pip``

.. code-block:: bash

   $ pip install nssvie

2. Install from Source
~~~~~~~~~~~~~~~~~~~~~~

|release| |licence|

Install directly from the source code by

.. code-block:: bash

	$ git clone https://github.com/dsagolla/nssvie.git
	$ cd nssvie
	$ pip install .

Usage
=====

.. code-block:: python
	
		>>> from nssvie import StochasticVolterraIntegralEquations
				
		>>> # Define the function and the kernels of the stochastic Volterra 
		>>> # integral equation
		>>> def f(t):
		>>> 	return 1.0
		
		>>> def k1(s,t):
		>>> 	return s**2
		
		>>> def k2(s,t):
		>>> 	return s
		
		>>> # Generate the stochastic Volterra integral equation
		>>> svie = StochasticVolterraIntegralEquations(
		>>> 	func=f, k1=k1, k2=k2, interval_end=0.5
		>>> )
		
		>>> # Calculate numerical solution with prec=100 intervals  
		>>> svie_solution = svie.solve_method(prec=100, solve_method="bpf")

The parameters are

+ ``func``: the function :math:`f`.
+ ``k1``, ``k2``: the kernels :math:`k_1` and :math:`k_2`.
+ ``interval_end``: the right hand side of :math:`[0,T)`. Default is ``1.0``.
+ ``prec``: the number of intervals to divide :math:`[0,T)`. Default is ``50``.
+ ``solve_method``: the choosen method based on orthogonal functions. Default is ``bpf``. 

for the stochastic Volterra integral equation above.

.. include:: cite.rst

.. |licence| image:: https://img.shields.io/github/license/dsagolla/nssvie
   :target: https://www.gnu.org/licenses/gpl-3.0.en.html
.. |pypi| image:: https://img.shields.io/pypi/v/nssvie
   :target: https://pypi.org/project/nssvie
.. |release| image:: https://img.shields.io/github/v/release/dsagolla/nssvie?display_name=release&sort=semver
   :target: https://github.com/dsagolla/nssvie/releases
.. |format| image:: https://img.shields.io/pypi/format/nssvie
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/nssvie
   :target: https://www.python.org/
.. |docs| image:: https://readthedocs.org/projects/nssvie/badge/?version=latest
   :target: https://nssvie.readthedocs.io/en/latest/?badge=latest
