"""Test for the block pulse functions vector."""
import numpy as np

from nssvie._orthogonal_functions import BlockPulseFunctions


def test_bpf_vector():
    """Test for the block pulse functions vector."""
    bpf = BlockPulseFunctions(endpoint=1.0, intervals=5)
    bpf_vector = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.array_equal(bpf._bpf_vector(time_t=0.5), bpf_vector)
