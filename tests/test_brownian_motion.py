"""Test sampling a brownian motion."""
import numpy as np

from nssvie._process import BrownianMotion


def test_sample_brownian_motion():
    """
    Test sampling from a brownian motion with n=4 steps in [0, 1.0].
    """
    seed = 1337
    sample_result = np.array(
        [
            0.0,
            0.019134111415207926,
            0.2561023293841905,
            0.18722937014124763,
            - 0.5074378503252513
        ]
    )

    b_motion = BrownianMotion(endpoint=1.0)
    sample = b_motion.sample(steps=4, seed=seed)
    assert np.equal(sample_result, sample).all()


def test_sample_at_brownian_motion():
    """
    Test sampling from a brownian motion at

        t = 0.1 | 0.3 | 0.32 | 0.7 | 0.99 | 1.0
    """
    seed = 23051949
    sample_result = np.array(
        [
            0.0,
            -0.10491921250193215,
            -0.4715104987300173,
            -0.48625307634203074,
            -0.3746896507354296,
            -1.0092947090635733,
            -0.9336231047319219
        ]
    )
    b_motion = BrownianMotion(endpoint=1.0)
    times = [0.1, 0.3, 0.32, 0.7, 0.99, 1.0]
    sample = b_motion.sample_at(times=times, seed=seed)

    assert np.equal(sample_result, sample).all()
