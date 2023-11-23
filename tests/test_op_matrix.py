"""Test for the operational matrix of integration."""
import numpy as np

from nssvie._orthogonal_functions import BlockPulseFunctions


def test_op_matrix():
    """Test for the operational matrix of integration."""
    bpf = BlockPulseFunctions(endpoint=1.0, intervals=5)
    matrix_p = np.array([
        [0.1, 0.2, 0.2, 0.2, 0.2],
        [0.0, 0.1, 0.2, 0.2, 0.2],
        [0.0, 0.0, 0.1, 0.2, 0.2],
        [0.0, 0.0, 0.0, 0.1, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.1],
    ])
    assert np.array_equal(matrix_p, bpf._operational_matrix_of_integration())
