from pathlib import Path
import numpy as np
from utils.loader import Loader


class Dataset(object):
    def __init__(self, images_train, images_mask, palette):

        self._images_train = images_train
        self._images_mask = images_mask
        self._palette = palette

    @property
    def images_train(self):
        return self._images_train

    @property
    def images_mask(self):
        return self._images_mask

    @property
    def palette(self):
        return self._palette

    @property
    def length(self):
        return len(self._images_train)

    # def __call__(self, batch_size=20, shuffle=True, augment=True):

    #     if batch_size < 1:
    #         raise ValueError("batch_size must be more than 1.")
    #     if shuffle:
    #         self.shuffle()

    #     for start in range(0, self.length, batch_size):
    #         batch = self.perm(start, start + batch_size)
    #         if augment:
    #             assert self._augmenter is not None, "have to set an augmenter."
    #             yield self._augmenter.augment_dataset(
    #                 batch,
    #                 method=[ia.ImageAugmenter.NONE, ia.ImageAugmenter.FLIP])
    #         else:
    #             yield batch

    def train_valid_split(self, train_rate=0.8, shuffle=True):

        if train_rate < 0.0 or train_rate > 1.0:
            raise ValueError("train_rate must be from 0.0 to 1.0.")
        if shuffle:
            self._shuffle()

        train_size = int(self._images_train.shape[0] * train_rate)
        data_size = int(len(self._images_train))

        train_set = self._perm(0, train_size)
        test_set = self._perm(train_size, data_size)

        return train_set, test_set

    def _shuffle(self):
        index = np.arange(self._images_train.shape[0])
        np.random.shuffle(index)
        self._images_train, self._images_mask = \
            self._images_train[index], self._images_mask[index]

    def _perm(self, start, end):
        end = min(end, len(self._images_train))
        return Dataset(self._images_train[start:end],
                       self._images_mask[start:end], self._palette)


if __name__ == "__main__":
    data_loader = Loader(dir_train=Path(r'.\src\datasets\train'),
                         dir_masks=Path(r'.\src\datasets\train_masks'))
    my_dataset = Dataset(data_loader.images_train, data_loader.images_mask,
                         data_loader.palette)
    train, valid = my_dataset.train_valid_split(train_rate=0.8, shuffle=True)
    print(my_dataset)
