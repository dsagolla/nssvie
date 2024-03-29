nssvie 
######

.. toctree::
   :maxdepth: 1
   :caption: Documentation
   :hidden:

   Home <self>
   API Reference <reference/api>

|tests| |build| |docs| |pypi| |pyversions| |licence| |linter|

A python package for computing a numerical solution of stochastic Volterra 
integral equations.

.. grid:: 3

    .. grid-item-card:: GitHub
        :img-top: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/_static/card-icons/github.svg
        :text-align: center
        :link: https://github.com/dsagolla/nssvie/

    .. grid-item-card:: PyPi
        :img-top: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/_static/card-icons/python.svg
        :text-align: center
        :link: https://pypi.org/project/nssvie/

    .. grid-item-card:: API Reference
        :img-top: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/_static/card-icons/magnifying-glass-solid.svg
        :text-align: center
        :link: reference/api
        :link-type: any


Overview
--------

A python package for computing a numerical solution of stochastic Volterra 
integral equations of the second kind

.. math::
	:label: svie_index

	X_t = f(t) + \int\limits_0^t k_1(s,t) X_s \ ds
        + \int\limits_0^t k_2(s,t) X_s \ dB_s \qquad t \in [0,T),

where 

+ :math:`X_t` is an unknown process,
+ :math:`f \in L^2([0,T))` is a continuous function,
+ :math:`k_1, \ k_2 \in L^2([0,T) \times [0,T))` are continuous and square integrable functions,
+ :math:`B_t` is the Brownian motion (see `Wiener process <https://en.wikipedia.org/wiki/Wiener_process>`_) and
+ :math:`\int_0^t k_2(s,t) X_s dB_s` is the Itô-integral (see `Itô calculus <https://en.wikipedia.org/wiki/It%C3%B4_calculus>`_)

by a stochastic operational matrix based on block
pulse functions as suggested in `Maleknejad et. al (2012) 
<https://www.sciencedirect.com/science/article/pii/S0895717711005504/>`_ [1]_.


Install
-------

Install using either of the following two methods.

1. Install from PyPi
~~~~~~~~~~~~~~~~~~~~

|pypi| |pyversions| |format| 

The ``nssvie`` package is available on
`PyPi <https://pypi.org/project/nssvie/>`_ and can be installed using ``pip``

.. code-block:: bash

    pip install nssvie


2. Install from Source
~~~~~~~~~~~~~~~~~~~~~~

|release| |licence|

Install directly from the source code by

.. code-block:: bash

	git clone https://github.com/dsagolla/nssvie.git
	cd nssvie
	pip install .	

Dependencies
~~~~~~~~~~~~

``nssvie`` uses 

+ `NumPy <https://numpy.org/>`_  for many calculations,
+ `SciPy <https://scipy.org>`_ for computing the block pulse coefficients and

Usage 
-----

Consider the following example of a stochastic Volterra integral equation

.. math:: 

	X_t = 1 + \int\limits_0^t s^2 X_s \ ds
        + \int\limits_0^t s X_s \ dB_s \qquad t \in [0,T),

so :math:`f \equiv 1`, :math:`k_1(s,t) = s^2` and :math:`k_2(s,t) = s` in
:eq:`svie_index`.

.. code-block:: python

	>>> from nssvie import SVIE
	>>> import matplotlib.pyplot as plt
			
	>>> # Define the function and the kernels of the stochastic Volterra 
	>>> # integral equation
	>>> def f(t):
	>>> 	return 1.0
	>>> def k1(s,t):
	>>> 	return s**2
	>>> def k2(s,t):
	>>> 	return s
	>>> # Generate the stochastic Volterra integral equation
	>>> svie = SVIE(
	>>> 	function_f=f, kernel_2=k1, kernel_1=k2, endpoint=0.5
	>>> )
	>>> # Calculate numerical solution with m=20 intervals  
	>>> approximative_solution = svie.solve_numerical(
			intervals=20
		)
	>>> fig, ax = plt.subplots()
	>>> times = [i * T/]
	>>> ax.step(approximative_solution, c='k')
	>>> plt.show()

|

.. image:: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/images/example.png
	:align: center

|

The parameters are

+ ``f``: the function :math:`f`.
+ ``kernel_1``, ``kernel_2``: the kernels :math:`k_1` and :math:`k_2`.
+ ``endpoint``: the right hand side of :math:`[0,T)`. Default is ``1.0``.
+ ``intervals``: the number of intervals to divide :math:`[0,T)`. Default is ``50``.

for the stochastic Volterra integral equation in :eq:`svie_index`.

Citation
--------

.. [1] Maleknejad, K., Khodabin, M., & Rostami, M. (2012). Numerical solution of stochastic Volterra integral equations by a stochastic operational matrix based on block pulse functions. Mathematical and computer Modelling, 55(3-4), 791-800. |maleknejad-et-al-2012-doi|    

.. |licence| image:: https://img.shields.io/github/license/dsagolla/nssvie
		:target: https://www.gnu.org/licenses/gpl-3.0.en.html
.. |pypi| image:: https://img.shields.io/pypi/v/nssvie
		:target: https://pypi.org/project/nssvie
.. |release| image:: https://img.shields.io/github/v/release/dsagolla/nssvie?display_name=release&sort=semver
		:target: https://github.com/dsagolla/nssvie/releases
.. |format| image:: https://img.shields.io/pypi/format/nssvie
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/nssvie
		:target: https://www.python.org/
.. |maleknejad-et-al-2012-doi| image:: https://img.shields.io/badge/DOI-10.1016%2Fj.mcm.2011.08.053-blue
		:target: https://doi.org/10.1016/j.mcm.2011.08.053
		:alt: doi: 10.1016/j.mcm.2011.08.053
.. |docs| image:: https://readthedocs.org/projects/nssvie/badge/?version=latest
		:target: https://nssvie.readthedocs.io/en/latest/?badge=latest
.. |build| image:: https://github.com/dsagolla/nssvie/actions/workflows/build.yml/badge.svg
		:target: https://github.com/dsagolla/nssvie/actions/workflows/build.yml
.. |tests| image:: https://github.com/dsagolla/nssvie/actions/workflows/tests.yml/badge.svg
	:target: https://github.com/dsagolla/nssvie/actions/workflows/tests.yml
.. |linter| image:: https://warehouse-camo.ingress.cmh1.psfhosted.org/d6d741fdb0ae96663fc5e9fbfb16b9ee24d52dfd/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c696e74696e672d70796c696e742d79656c6c6f77677265656e
	:target: https://github.com/pylint-dev/pylint