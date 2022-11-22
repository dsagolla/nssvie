from nssvie._orthogonal_functions import BlockPulseFunctions
from numpy import (
    array,
    round,
    array_equal
)


def test_coeff_vector():
    bpf = BlockPulseFunctions(T=1.0, m=5)
    coeff_vector = array([1/75, 7/75, 19/75, 37/75, 61/75])
    assert array_equal(
        round(bpf._coeff_vector(lambda x: x**2), 10),
        round(coeff_vector, 10)
    )


def test_coeff_matrix():
    bpf = BlockPulseFunctions(T=1.0, m=5)
    bpf_matrix = array([
        [1/750, 7/750, 19/750, 37/75, 61/750],
        [3/750, 21/750, 57/750, 111/75, 183/750],
        [5/750, 35/750, 95/750, 185/75, 305/750],
        [7/750, 49/750, 133/750, 259/75, 427/750],
        [9/750, 63/750, 171/750, 333/75, 549/750]
    ])
    assert array_equal(
        bpf._coeff_matrix(lambda x, y: x*(y**2)),
        bpf_matrix
    )
