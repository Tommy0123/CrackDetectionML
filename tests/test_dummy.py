import pytest


@pytest.fixture
def sample_data():
    return [1, 5, 2, 4, 3]


def test_sort_list_with_pytest_fixture(sample_data):
    expected_list = [1, 2, 3, 4, 5]
    sorted_list = sorted(sample_data)
    assert sorted_list == expected_list, "should same list."
