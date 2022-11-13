|project| documentation
***********************

|pypi| |pyversions| |docs| |licence|

.. toctree::
   :maxdepth: 1
   :caption: Documentation
   :hidden:

   Home <self>
   Theory <theory>
   API Reference <api>

.. .. include:: ../../README.rst
..    :start-after: .. include_after_this_line

A python package for computing a numerical solution of stochastic Volterra 
integral equations by a stochastic operational matrix based on block
pulse functions as suggested in ``Maleknejad et. al (2012)`` [1]_.

.. grid:: 3

   .. grid-item-card:: GitHub
      :link: https://github.com/dsagolla/nssvie
      :text-align: center
      :class-card: custom-card-link

   .. grid-item-card:: PyPI
      :link: https://pypi.org/project/nssvie/
      :text-align: center
      :class-card: custom-card-link

   .. grid-item-card:: API reference
      :link: api
      :link-type: ref
      :text-align: center
      :class-card: custom-card-link

Overview
========

Build and Test Status
=====================



.. |y| unicode:: U+2714
.. |n| unicode:: U+2716

+----------+--------+-------+-------+-------+-------+-------+-----------------+
| Platform | Arch   | Python Version                        | Continuous      |
+          |        +-------+-------+-------+-------+-------+ Integration     +
|          |        |  3.6  |  3.7  |  3.8  |  3.9  |  3.10 |                 |
+==========+========+=======+=======+=======+=======+=======+=================+
| Linux    | X86-64 |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-linux|   |
+----------+--------+-------+-------+-------+-------+-------+-----------------+
| macOS    | X86-64 |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-macos|   |
+----------+--------+-------+-------+-------+-------+-------+-----------------+
| Windows  | X86-64 |  |y|  |  |y|  |  |y|  |  |y|  |  |y|  | |build-windows| |
+----------+--------+-------+-------+-------+-------+-------+-----------------+

.. |build-linux| image:: https://img.shields.io/github/workflow/status/ameli/imate/build-linux
   :target: https://github.com/ameli/imate/actions?query=workflow%3Abuild-linux 
.. |build-macos| image:: https://img.shields.io/github/workflow/status/ameli/imate/build-macos
   :target: https://github.com/ameli/imate/actions?query=workflow%3Abuild-macos
.. |build-windows| image:: https://img.shields.io/github/workflow/status/ameli/imate/build-windows
   :target: https://github.com/ameli/imate/actions?query=workflow%3Abuild-windows

Install
=======

Install using either of the following two methods.

1. Install from PyPi
~~~~~~~~~~~~~~~~~~~~

|pypi| |pyversions| |format| 

The ``nssvie`` package is available on `PyPi <https://pypi.org/project/nssvie/>`_ and can be installed using ``pip``

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

Testing
=======

To test the package, download the source code and use one of the following methods in the directory of the source code:

- *Method 1*: test locally by:

  .. prompt:: bash
      
      python setup.py test

- *Method 2*: test in a virtual environment using ``tox``:

  .. prompt:: bash

      pip install tox
      tox

Usage
=====

The package can be used in two ways:

1. Import as a Module
---------------------

.. code-block:: python

    >>> from nssvie import StochasticVolterraIntegralEquation
    
    >>> # Generate object of orthogonal functions
    >>> OF = OrthogonalFunctions(
    ...        start_index=1,
    ...        num_func=9,
    ...        end_interval=1,
    ...        verbose=True)
    
    >>> # Get numeric coefficients alpha[i] and a[i][j]
    >>> alpha = OF.alpha
    >>> a = OF.coeffs

    >>> # Get symbolic coefficients alpha[i] and a[i][j]
    >>> sym_alpha = OF.sym_alpha
    >>> sym_a = OF.sym_coeffs

    >>> # Get symbolic functions phi[i]
    >>> sym_phi = OF.sym_phi
    
    >>> # Print Functions
    >>> OF.print()
    
    >>> # Check mutual orthogonality of Functions
    >>> status = OF.check(verbose=True)
    
    >>> # Plot Functions
    >>> OF.plot()

The parameters are:

- ``start_index``: the index of the starting function, . Default is ``1``.
- ``num_func``: number of orthogonal functions to generate, . Default is ``9``.
- ``end_interval``: the right interval of orthogonality, . Default is ``1``.

2. Use As Standalone Application
--------------------------------

The standalone application can be executed in the terminal in two ways:

#. If you have installed the package, call ``ortho`` executable in terminal:

   .. prompt:: bash

       ortho [options]

   The optional argument ``[options]`` will be explained in the next section. When the package *ortho* is installed, the executable ``ortho`` is located in the ``/bin`` directory of the python.

#. Without installing the package, the main script of the package can be executed directly from the source code by

   .. prompt:: bash

       # Download the package
       git clone https://github.com/ameli/ortho.git

       # Go to the package source directory
       cd ortho

       # Execute the main script of the package
       python -m ortho [options]

Optional arguments
------------------

When the *standalone application* (the second method in the above) is called, the executable accepts some optional arguments as follows.

+--------------------------------------+------------------------------------------------------------------------------------------+
| Option                               | Description                                                                              |
+======================================+==========================================================================================+
| ``-h``, ``--help``                   | Prints a help message.                                                                   |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-v``, ``--version``                | Prints version.                                                                          |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-l``, ``--license``                | Prints author info, citation and license.                                                |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-n``, ``--num-func[=int]``         | Number of orthogonal functions to generate. Positive integer. Default is 9.              |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-s``, ``--start-func[=int]``       | Starting function index. Non-negative integer. Default is 1.                             |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-e``, ``--end-interval[=float]``   | End of the interval of functions domains. A real number greater than zero. Default is 1. |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-c``,\ ``--check``                 | Checks orthogonality of generated functions.                                             |
+--------------------------------------+------------------------------------------------------------------------------------------+
| ``-p``, ``--plot``                   | Plots generated functions, also saves the plot as pdf file in the current directory.     |
+--------------------------------------+------------------------------------------------------------------------------------------+

Parameters
----------

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
