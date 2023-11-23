"""
Standard Brownian Motion.

Class
-----
BrownianMotion
"""
from typing import Callable

from numpy.typing import ArrayLike

import numpy as np

from nssvie._process.base import StochasticProcess


class BrownianMotion(StochasticProcess):
    """(Standard) Brownian motion.

    A (standard) Brownian motion is a real-valued stochastic process
    :math:`B = \\left( B_{t \\in [0, \\infty \\right)}` with

    + :math:`B_0 = 0` almost sure,
    + independant and stationary increments,
    + :math:`B_t` is :math:`\\mathcal{N}(0, t)`-distributed
    + continuous paths :math:`t \\mapsto B_t`.

    Parameters
    ----------
    endpoint : float, optional
        The right hand endpoint of the time interval [0, T] for the
        Brownian motion.
    """

    def __str__(self) -> None:
        return f"Standard Brownian motion on the interval [0, {self.endpoint}]"

    def __repr__(self) -> None:
        return f'StandardBrownianMotion(interval=[0, {self.endpoint}])'

    def sample(self, steps: int, seed: int = None) -> np.array:
        """Generate a path of a standard Brownian motion.

        Parameters
        ----------
        steps : int
            The number of increments to generate.
        seed : int, optional
            A seed to initialize the BitGenerator. If None, then fresh,
            unpredictable entropy will be pulled from the OS. By default
            None.

        Returns
        -------
        np.array
            A sample from a brownian motion in the interval
            :math:`[0, T]` with :math:`n` increments.
        """
        # Calculate the step width
        step_width = self.endpoint / steps

        # Calculate the increments
        increments = np.sqrt(step_width) * (
            np.random.default_rng(seed=seed).normal(
                loc=0,
                scale=1,
                size=steps
            )
        )

        # Calculate the path
        bm_sample = np.cumsum(increments)

        # Start in 0, B_0 = 0
        bm_sample = np.insert(bm_sample, [0], 0)

        return bm_sample

    def sample_at(self, times: ArrayLike, seed: int = None) -> np.array:
        """Generate a path of a standard Brownian motion at given times.

        Generate a sample of a brownian motion

        .. math::

            \\begin{pmatrix} B_{0} & B_{t_1} & \\ldots B_{t_n}
            \\end{pmatrix}

        at times :math:`0, t_1, \\ldots, t_n`.

        Parameters
        ----------
        times : ArrayLike
            A vector of increasing time values
            :math:`\\begin{pmatrix} t_1 & \\ldots & t_n \\end{pmatrix}`.
        seed : int, optional
            A seed to initialize the BitGenerator. If None, then fresh,
            unpredictable entropy will be pulled from the OS. By default
            None.

        Returns
        -------
        np.array
            A sample from a brownian motion at given times.
        """
        # Calculate the scales for the increments
        steps = len(times)
        scales = np.array([np.diff(times, prepend=0)])

        # Calculate the increments
        increments = np.sqrt(scales) * (
            np.random.default_rng(seed=seed).normal(
                loc=0,
                scale=1,
                size=steps
            )
        )

        # Calculate the path
        bm_sample = np.cumsum(increments)

        # Start in 0
        bm_sample = np.insert(bm_sample, [0], 0)

        return bm_sample

    def sample_as_func(self, seed: int = None) -> Callable[[float], float]:
        """Generate a sample as a function
        :math:`\\hat{B}_t \\colon [0, T] \\to \\mathcal{R}`.

        Parameters
        ----------
        seed : int, optional
            A seed to initialize the BitGenerator. If None, then fresh,
            unpredictable entropy will be pulled from the OS. By default
            None.

        Returns
        -------
        Callable[[float], float]
            A sampled path of a Brownian motion as a function
            :math:`\\hat{B}_t \\colon [0, T] \\to \\mathcal{R}`.
        """
        # Calculate the time index
        times = np.array([i * self.endpoint / 10**5 for i in range(10**5 + 1)])

        # Calculate a sample path with 10^5 increments in [0,T]
        bm_sample = self.sample(steps=10**5, seed=seed)

        # Define the sample function
        def sample_func(time_t: float):
            # Find the closest time value for 'time_t'
            abs_diffs = np.abs(times - time_t)
            closest_value_idx = abs_diffs.argmin()

            # Return the value B_s, where s is the closest time value to
            # 'time_t'
            return bm_sample[closest_value_idx]

        # Return the sample function as a vectorized function
        return np.vectorize(sample_func)
