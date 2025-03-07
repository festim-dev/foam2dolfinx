import numpy as np
import pytest

from foam2dolfinx import find_closest_value


@pytest.mark.parametrize(
    ("input_value", "expected_value"),
    [(1.2, 1), (2.1, 2), (1.55, 2), (4.01, 4), (3.9, 4)],
)
def test_error(input_value, expected_value):
    """Test that the find closest value function find the closest value within a
    predefined list to a given input value"""

    my_times = [1, 2, 3, 4, 5, 6]

    value = find_closest_value(values=my_times, target=input_value)

    assert np.isclose(value, expected_value)
