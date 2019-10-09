import argparse


def train(parser):
    """処理した引数を受け取り、一連の学習を行います。
    """


def get_parser():
    """処理した引数を返します。
    """
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python train.py',
        description='This module demonstrates image segmentation using U-Net.',
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
