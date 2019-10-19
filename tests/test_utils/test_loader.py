from src.utils.loader import Loader
from pathlib import Path
import pytest


@pytest.fixture
def paths_train():
    return Path(r'.\tests\datasets\train')


@pytest.fixture
def paths_masks():
    return Path(r'.\tests\datasets\train_masks')


@pytest.fixture
def paths_empty_directory():
    return Path(r'.\tests\datasets\empty')


@pytest.fixture
def paths_dummy_directory():
    return Path(r'.\tests\datasets\dummy')


def test_Loader(paths_train, paths_masks):
    loader = Loader(paths_train, paths_masks)
    assert len(loader.images_train) == 2, "should 2."
    assert len(loader.images_mask) == 2, "should 2."
    assert loader.paths_train[0].name == \
        "crack_00000.jpg", "should crack_00000.jpg."
    assert loader.paths_masks[0].name == \
        "crack_00000_mask.png", "should crack_00000.png."


def test_Loader_raise_File_Not_Found(paths_train, paths_empty_directory):
    Path(paths_empty_directory).mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError):
        Loader(paths_train, paths_empty_directory)


def test_Loader_raise_ValueError(paths_train, paths_dummy_directory):
    with pytest.raises(ValueError):
        Loader(paths_train, paths_dummy_directory)
