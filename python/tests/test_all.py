import bisect
import pytest

import numpy as np
import pyarrow

from pysect import bisect_python, bisect_rust, bisect_rust_arrow, bisect_rust_arrow_with_targets


@pytest.fixture
def sorted_input():
    return [1., 5.5, 11., 19.3]


@pytest.fixture
def targets():
    return [0., 4., 11., 19., 20.]


def test_python_imp(sorted_input, targets):
    for target in targets:
        assert bisect.bisect_left(sorted_input, target) == bisect_python(sorted_input, target)


def test_rust_imp(sorted_input, targets):
    for target in targets:
        assert bisect.bisect_left(sorted_input, target) == bisect_rust(sorted_input, target)


def test_rust_np(sorted_input, targets):
    np_input = np.array(sorted_input)
    for target in targets:
        assert bisect.bisect_left(sorted_input, target) == bisect_rust(np_input, target)


def test_rust_arrow(sorted_input, targets):
    arrow_input = pyarrow.array(sorted_input)
    for target in targets:
        assert bisect.bisect_left(sorted_input, target) == bisect_rust_arrow(arrow_input, target)


def test_rust_arrow_with_targets(sorted_input, targets):
    arrow_input = pyarrow.array(sorted_input)
    arrow_targets = pyarrow.array(targets)

    results = bisect_rust_arrow_with_targets(arrow_input, arrow_targets)

    for target, result in zip(targets, results):
        assert bisect.bisect_left(sorted_input, target) == result.as_py()
