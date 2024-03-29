"""Test for the block pulse coefficient vector and matrix."""
import numpy as np

from nssvie._orthogonal_functions import BlockPulseFunctions


def test_coeff_vector():
    """Test for the block pulse coefficient vector."""
    bpf = BlockPulseFunctions(endpoint=1.0, intervals=5)
    coeff_vector = np.array([1/75, 7/75, 19/75, 37/75, 61/75])
    assert np.array_equal(
        np.round(bpf._coeff_vector(lambda x: x**2), 11),
        np.round(coeff_vector, 11)
    )


def test_coeff_matrix():
    """Test for the block pulse coefficient matrix."""
    bpf = BlockPulseFunctions(endpoint=1.0, intervals=5)
    bpf_matrix = np.array([
        [1/750, 7/750, 19/750, 37/750, 61/750],
        [3/750, 21/750, 57/750, 111/750, 183/750],
        [5/750, 35/750, 95/750, 185/750, 305/750],
        [7/750, 49/750, 133/750, 259/750, 427/750],
        [9/750, 63/750, 171/750, 333/750, 549/750]
    ])
    assert np.array_equal(
        np.round(bpf._coeff_matrix(lambda x, y: x*(y**2)), 11),
        np.round(bpf_matrix, 11)
    )
