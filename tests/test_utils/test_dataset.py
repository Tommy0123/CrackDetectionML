from src.utils.loader import Loader
from src.utils.dataset import Dataset
from pathlib import Path
import pytest


@pytest.fixture
def sample_Dataset():
    sample_Loader = Loader(dir_train=Path(r'.\tests\datasets\train'),
                           dir_masks=Path(r'.\tests\datasets\train_masks'))
    return Dataset(sample_Loader.images_train, sample_Loader.images_mask,
                   sample_Loader.palette)


def test_train_valid_split(sample_Dataset):
    train, valid = sample_Dataset.train_valid_split(train_rate=0.5)
    assert train.length == 1, "should 1."
    assert valid.length == 1, "should 1."


def test_train_valid_split_raise_ValueError(sample_Dataset):
    with pytest.raises(ValueError):
        sample_Dataset.train_valid_split(train_rate=2.0)
