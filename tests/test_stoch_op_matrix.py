"""Test for the stochastic operational matrix."""
import numpy as np

from nssvie._orthogonal_functions import BlockPulseFunctions


def test_stoch_op_matrix():
    """
    Test for the stochastic operational matrix.

    T = 1.0, m = 4 intervals
    """
    matrix_s = np.array([
        [
            -0.3278178002307414, -0.2734952828162863, -0.2734952828162863,
            -0.2734952828162863
        ],
        [
            0.0, -0.23022862638599573, -0.5347863381753177, -0.5347863381753177
        ],
        [
            0.0, 0.0, 0.31314626357790165, 0.3053026435052142
        ],
        [
            0.0, 0.0, 0.0, 0.44976301384696993
        ]
    ])
    seed = 23051949
    bpf = BlockPulseFunctions(endpoint=1.0, intervals=4)
    assert np.array_equal(
        matrix_s,
        bpf._stochastic_operational_matrix_of_integration(seed=seed)
    )
