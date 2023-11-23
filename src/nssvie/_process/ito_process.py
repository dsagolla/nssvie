"""
Itô process.

Class
-----
ItoProcess
"""
from typing import Callable

import numpy as np

from scipy.integrate import quad

from nssvie._process.base import StochasticProcess
from nssvie._process.brownian_motion import BrownianMotion


class ItoProcess(StochasticProcess):
    """Generate an itô process.

    An itô process is a stochastic process
    :math:`X=(X_t)_{t \\in [0, T]}', where

    .. math::
        :label: ito_process

        X_t = x_0 + \\int\\limits_0^t a(s) ds
        + \\int\\limits_0^t b(s) dB_s

    for all :math:`t \\in [0, T]`.

    Parameters
    ----------
    start : float
        The starting value :math:`x_0` in eq:`ito_process`.
    function_a, function_b : Callable[[float], float]
        Functions :math:`a, b` in :eq:`ito_process`.
    T : float, optional
        The right hand side of the interval :math:`[0,T)`, by default
        1.0.
    """

    def __init__(
            self,
            start: float,
            function_a: Callable[[float], float],
            function_b: Callable[[float], float],
            endpoint: float = 1.0
    ) -> None:
        self.start = start
        self.function_a = function_a
        self.function_b = function_b
        super().__init__(endpoint=endpoint)

    def __str__(self) -> None:
        return (
            f'Ito Process on interval [0, {self.endpoint}]'
        )

    def __repr__(self) -> None:
        return (
            f'ItoProcess(x0, a, b, interval = [0, {self.endpoint}])'
        )

    def sample(self, steps: int, seed: int = None) -> np.array:
        """
        Create a sample from the given itô process.

        Parameters
        ----------
        steps : int
            The number of steps.
        seed : int, optional
            A seed to initialize the BitGenerator. If None, then fresh,
            unpredictable entropy will be pulled from the OS. By default
            None.

        Returns
        -------
        np.array
            A sample from the given itô process.
        """
        # Define the mean function as m(t) = x0 + int(0,t) a(s) ds
        def mean_function(time_t: float) -> float:
            return self.start + quad(self.function_a, 0, time_t)[0]
        mean_function = np.vectorize(mean_function)

        # Define the scale function as sc(t) = int(s,t) b^2(x) dx
        def scale_function(time_s: float, time_t: float) -> float:
            return quad(self.function_b, time_s, time_t)[0]

        # Create the times vector
        step_width = self.endpoint / steps
        times = [k * step_width for k in range(steps+1)]

        # Calculate the scale vector
        scale = [scale_function(times[i], times[i+1]) for i in range(steps)]

        # Generate a sample of a Brownian motion
        b_motion = BrownianMotion(endpoint=self.endpoint)
        bm_sample_func = b_motion.sample_as_func(seed=seed)
        bm_sample = [bm_sample_func(times[1:])]

        gaussian_noise = (
            np.diff(bm_sample) * np.sqrt(scale) / np.sqrt(step_width)
        )

        # Generate the scale part of the itô process
        scale_part = np.cumsum(gaussian_noise)
        scale_part = np.insert(scale_part, [0], 0)

        # Generate the mean part of the itô process
        mean_part = mean_function(times)

        return mean_part + scale_part
