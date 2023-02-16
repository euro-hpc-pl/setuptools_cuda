import numpy as np
import pytest
from saxpycu import saxpy

NUM_TESTS = 100
VECTOR_SIZE = 1000000
NUM_BLOCKS = 64
NUM_THREADS = 1024


def generate_test_cases():
    rng = np.random.default_rng(1234)
    return [
        (rng.random(), rng.random(VECTOR_SIZE), rng.random(VECTOR_SIZE))
        for _ in range(NUM_TESTS)
    ]


@pytest.mark.parametrize("a, x, y", generate_test_cases())
def test_saxpy(a, x, y):
    expected_result = a * x + y
    saxpy(a, x, y, num_threads=NUM_THREADS, num_blocks=NUM_BLOCKS)
    np.testing.assert_almost_equal(y, expected_result)
