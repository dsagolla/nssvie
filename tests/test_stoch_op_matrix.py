from nssvie._orthogonal_functions import BlockPulseFunctions

from numpy import (
    array,
    round,
    array_equal
)


def test_stoch_op_matrix():
    bpf = BlockPulseFunctions(T=1.0, m=5)
    P = array([
        [0.1, 0.2, 0.2, 0.2, 0.2],
        [0.0, 0.1, 0.2, 0.2, 0.2],
        [0.0, 0.0, 0.1, 0.2, 0.2],
        [0.0, 0.0, 0.0, 0.1, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.1],
    ])
    assert array_equal(
        round(P, 10),
        round(bpf._operational_matrix_of_integration(), 10)
    )
