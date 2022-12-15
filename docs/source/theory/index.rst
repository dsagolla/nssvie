========
Theory
========

Introduction
------------

We present a numerical method for solving stochastic Volterra integral equations
based on block pulse functions and a stochastic operational matrix of
integration as suggested in [1]_. By applying this method the problem is reduced
to solve a linear lower triangular system.

Basics
------

First we we briefly introduce some basic properties about block pulse functions
and function / integral approximation with them. Then we focus on repeating some
results about random variables, conditional expectation, martingales and give a
short reminder about the construction of the itô-integral.

Block pulse functions
~~~~~~~~~~~~~~~~~~~~~

Define an :math:`m`-set of block pulse functions (BPFs) as

.. math::

    \phi_i(t) = \begin{cases} 1 & , \ (i-1)h \leq t < ih \\ 0 &
    \text{otherwise} \end{cases} \qquad (i=1,\ldots,m)

for :math:`t \in [0,T)` and an interval width of :math:`h=\frac{T}{m}`.

.. image:: ../images/bpfs.png

The BPF's are

- disjoint
    i.e. 

    .. math::

        \phi_i(t)\phi_j(t) = \delta_{ij} \phi_i(t),

    for :math:`i,j = 1, \ldots, m` and :math:`\delta_{ij}` the Kronecker delta

- orthogonal
    i.e.

    .. math::

        \int\limits_0^T \phi_i(t) \phi_j(t) dt = h \delta_{ij},

    for for :math:`i,j=1,\ldots,m`

- complete
    i.e.

    .. math::

        \frac{1}{T} \int\limits_0^T \left[ f(s) - \sum\limits_{k=0}^m f_j \phi_i(s) \right]^2 ds

    decreases monotonically to zero as :math:`m` tends to infinity, where

    .. math::

        f_i = \frac{1}{h} \int\limits_0^T f(s) \phi_i(s) ds

    are the so called block **pulse coefficients**.

Conditional expectation 
~~~~~~~~~~~~~~~~~~~~~~~

Given are a probability space :math:`(\Omega, \mathcal{A}, P)`, a
sub-:math:`\sigma`-Algebra :math:`\mathcal{F} \subset \mathcal{A}` and an
integrable random variable :math:`X \in \mathcal{L}^1(\Omega, \mathcal{A}, P)`.

.. rubric:: Definition

A random variable :math:`Y` is called **conditional expectation** of :math:`X`
given :math:`\mathcal{F}`, in symbols :math:`E(X|\mathcal{F}) := Y`, if

.. hlist:: 
    :columns: 1

    * :math:`Y` is measurable with respect to :math:`F`
    * :math:`E(X \mathbb{1}_A) = E(Y \mathbb{1}_A)` for all :math:`A \in \mathcal{F}`.

------------

Any random variable :math:`Y` satisfying these two conditions is called a
**version** of :math:`E(X|\mathcal{F})`. All these equations are equations in
the almost sure sense.

Brownian Motion
~~~~~~~~~~~~~~~

Itô-Integral
~~~~~~~~~~~~

Stochastic integration operational matrix
-----------------------------------------

Numerical solution of stochastic Volterra integral equations
-------------------------------------------------------------

We consider the following stochastic Volterra integral equation

.. math::

    X(t) = f(t) + \int\limits_0^t k_1(s,t) X_s ds + \int\limits_0^t k_2(s,t) X_s dB_s, \qquad t \in [0,T).

.. toggle:: 
    
    Test

Examples
--------

Example 1
~~~~~~~~~

Example 2
~~~~~~~~~

Example 3
~~~~~~~~~

Applications
------------

References
----------

.. [1] Maleknejad, K., Khodabin, M., & Rostami, M. (2012). Numerical solution of stochastic Volterra integral equations by a stochastic operational matrix based on block pulse functions. Mathematical and computer Modelling, 55(3-4), 791-800. |maleknejad-et-al-2012-doi|    

.. |maleknejad-et-al-2012-doi| image:: https://img.shields.io/badge/DOI-10.1016%2Fj.mcm.2011.08.053-blue
		:target: https://doi.org/10.1016/j.mcm.2011.08.053
		:alt: doi: 10.1016/j.mcm.2011.08.053