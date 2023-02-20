import numpy as np
import pytest
import thrust

NUM_TESTS = 100
VECTOR_SIZE = 1000000

FLOAT_DTYPES = (np.float32, np.float64)
INTEGER_DTYPES = (
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
)


def generate_test_cases():
    rng = np.random.default_rng(1234)
    return [
        *[rng.random(VECTOR_SIZE, dtype=dtype) for _ in range(NUM_TESTS) for dtype in FLOAT_DTYPES],
        *[
            rng.integers(np.iinfo(dtype).min, np.iinfo(dtype).max, VECTOR_SIZE, dtype=dtype)
            for _ in range(NUM_TESTS)
            for dtype in INTEGER_DTYPES
        ],
    ]


@pytest.mark.parametrize("data", generate_test_cases())
def test_sort(data):
    thrust.sort(data)
    assert (np.diff(data) >= 0).all()
