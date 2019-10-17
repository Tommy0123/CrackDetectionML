import argparse
from pathlib import Path
from utils import loader


def train(parser):
    """処理した引数を受け取り、一連の学習を行います。
    """
    data_loader = loader.Loader(dir_train=Path(r'.\src\datasets\train'),
                                dir_masks=Path(r'.\src\datasets\train_masks'),
                                labels=("ground", "crack", "void"),
                                init_size=(128, 128),
                                one_hot=True)
    print(data_loader)


def get_parser():
    """処理した引数を返します。
    """
    parser = argparse.ArgumentParser(
        prog='Crack Detection.',
        usage='python main.py',
        description='This module demonstrates crack detection.',
        add_help=True)

    parser.add_argument('-d',
                        '--dummy',
                        type=int,
                        default=1,
                        help='This is dummy type int.')
    return parser


if __name__ == "__main__":
    parser = get_parser().parse_args()
    train(parser)
