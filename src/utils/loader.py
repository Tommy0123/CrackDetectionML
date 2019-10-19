from pathlib import Path
from PIL import Image
import numpy as np


class Loader(object):
    def __init__(self,
                 dir_train: Path,
                 dir_masks: Path,
                 labels: tuple = ("ground", "crack", "void"),
                 init_size: (int, int) = (128, 128),
                 one_hot: bool = True):

        paths_train = list(dir_train.iterdir())
        paths_masks = list(dir_masks.iterdir())
        if len(paths_train) == 0 or len(paths_masks) == 0:
            raise FileNotFoundError('Could not load images.')
        if len(paths_train) != len(paths_masks):
            raise ValueError('Num of images does not match.')

        self.paths_train = paths_train
        self.paths_masks = paths_masks
        self.images_train = Loader.import_train_data(paths_train, init_size)
        self.images_mask = Loader.import_mask_data(paths_masks, init_size,
                                                   labels, one_hot)
        self.palette = Loader.get_palette(paths_masks[0])

    def import_train_data(paths_train: list, init_size: (int, int)):

        images_train = []
        for path_train in paths_train:
            image = Image.open(path_train)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image = Loader._crop_to_square(image)
            image = image.resize(init_size, Image.ANTIALIAS)
            image = np.asarray(image)
            image = image / 255.0
            images_train.append(image)
        images_train = np.asarray(images_train, dtype=np.float32)

        return images_train

    def import_mask_data(paths_masks: list, init_size: (int, int),
                         labels: tuple, one_hot: bool):

        images_mask = []
        for path_masks in paths_masks:
            mask = Image.open(path_masks)
            mask = Loader._crop_to_square(mask)
            mask = mask.resize(init_size)
            mask = np.asarray(mask)
            images_mask.append(mask)
        images_mask = np.asarray(images_mask, dtype=np.uint8)
        images_mask = np.where(images_mask == 255,
                               len(labels) - 1, images_mask)
        if one_hot:
            identity = np.identity(len(labels), dtype=np.uint8)
            images_mask = identity[images_mask]

        return images_mask

    def get_palette(path_mask):

        image_palette = Image.open(path_mask)
        palette = image_palette.getpalette()

        return palette

    def _crop_to_square(image):
        size = min(image.size)
        left, upper = (image.width - size) // 2, (image.height - size) // 2
        right, bottom = (image.width + size) // 2, (image.height + size) // 2
        return image.crop((left, upper, right, bottom))


if __name__ == "__main__":
    data_loader = Loader(dir_train=Path(r'.\src\datasets\train'),
                         dir_masks=Path(r'.\src\datasets\train_masks'),
                         labels=("ground", "crack", "void"),
                         init_size=(128, 128),
                         one_hot=True)
    print(data_loader)
