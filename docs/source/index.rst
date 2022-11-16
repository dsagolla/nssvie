nssvie 
######

.. toctree::
   :maxdepth: 1
   :caption: Documentation
   :hidden:

   Home <self>
   Theory <theory/index>
   API Reference <reference/api>

|build| |docs| |pypi| |pyversions| |licence|

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

A python package for computing a numerical solution of stochastic Volterra 
integral equations.

.. grid:: 2

   .. grid-item-card:: GitHub
      :link: https://github.com/dsagolla/nssvie
      :text-align: center

   .. grid-item-card:: PyPI
      :link: https://pypi.org/project/nssvie/
      :text-align: center

   .. grid-item-card:: Theory
      :link: theory/index.html
      :text-align: center

   .. grid-item-card:: API reference
      :link: reference/api.html
      :text-align: center

.. grid:: 2

    .. grid-item-card::
        :img-top: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/_static/index-images/getting_started.svg

        Getting Started
        ^^^^^^^^^^^^^^^

        New to NumPy? Check out the Absolute Beginner's Guide. It contains an
        introduction to NumPy's main concepts and links to additional tutorials.

        +++

        .. button-link:: https://github.com/dsagolla/nssvie
            :expand:
            :color: secondary
            :click-parent:

            To the absolute beginner's guide

    .. grid-item-card::
        :img-top: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/_static/index-images/user_guide.svg

        User Guide
        ^^^^^^^^^^

        The user guide provides in-depth information on the
        key concepts of NumPy with useful background information and explanation.

        +++

        .. button-link:: https://pypi.org/project/nssvie/
            :expand:
            :color: secondary
            :click-parent:

            To the user guide

    .. grid-item-card::
        :img-top: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/_static/index-images/api.svg

        API Reference
        ^^^^^^^^^^^^^

        The reference guide contains a detailed description of the functions,
        modules, and objects included in NumPy. The reference describes how the
        methods work and which parameters can be used. It assumes that you have an
        understanding of the key concepts.

        +++

        .. button-ref:: theory/index
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide

    .. grid-item-card::
        :img-top: https://raw.githubusercontent.com/dsagolla/nssvie/main/docs/source/_static/index-images/contributor.svg

        Contributor's Guide
        ^^^^^^^^^^^^^^^^^^^

        Want to add to the codebase? Can help add translation or a flowchart to the
        documentation? The contributing guidelines will guide you through the
        process of improving NumPy.

        +++

        .. button-ref:: reference/api
            :expand:
            :color: secondary
            :click-parent:

            To the contributor's guide

Overview
--------

A python package for computing a numerical solution of stochastic Volterra 
integral equations of the second kind

|stochastic-volterra-integral-equation|

where

+ |X-t| is an unknown process,
+ |f| is a continuous function,
+ |k-1-k-2| are continuous and square integrable functions,
+ |B-t| is the Brownian Motion and
+ |ito-integral| is the Itô-integral

by a stochastic operational matrix based on block
pulse functions as suggested in `Maleknejad et. al (2012) 
<https://www.sciencedirect.com/science/article/pii/S0895717711005504/>`_ [1]_.


Install
-------

Install using either of the following two methods.

1. Install from PyPi
~~~~~~~~~~~~~~~~~~~~

|pypi| |pyversions| |format| 

The ``nssvie`` package is available on `PyPi <https://pypi.org/project/nssvie/>`_ and can be installed using ``pip``

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
+ `stochastic <https://pypi.org/project/stochastic/>`_ for sampling the Brownian Motion

Usage 
-----

Consider the following example of a stochastic Volterra integral equation

|example-1-eq|,

so 

+ |example-1-f| ,   
+ |example-1-k-1| and   
+ |example-1-k-2|.

.. code-block:: python

	from nssvie import StochasticVolterraIntegralEquations

	# Define the function and the kernels of the stochastic Volterra 
	# integral equation
	def f(t):
		return 1.0

	def k1(s,t):
		return s**2

	def k2(s,t):
		return s
	
	# Generate the stochastic Volterra integral equation
	svie = StochasticVolterraIntegralEquations(
		func=f, k1=k1, k2=k2, interval_end=0.5
	)

	# Calculate numerical solution with m=100 intervals  
	svie_solution = svie.solve_method(prec=100, solve_method="bpf")


The parameters are

+ ``func``: the function :math:`f`.
+ ``k1``, ``k2``: the kernels :math:`k_1` and :math:`k_2`.
+ ``interval_end``: the right hand side of :math:`[0,T)`. Default is ``1.0``.
+ ``prec``: the number of intervals to divide :math:`[0,T)`. Default is ``50``.
+ ``solve_method``: the choosen method based on orthogonal functions. Default is ``bpf``. 

for the stochastic Volterra integral equation above.

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
.. |build| image:: https://img.shields.io/github/workflow/status/dsagolla/nssvie/Upload%20Python%20Package
		:target: https://github.com/dsagolla/nssvie/actions/workflows/python-publish.yml
