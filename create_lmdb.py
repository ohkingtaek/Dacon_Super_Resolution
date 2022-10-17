from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs


def create_lmdb_for_dacon():
    # Create lmdb files for Dacon dataset.

    # HR images
    folder_path = 'datasets/open/train/hr'
    lmdb_path = 'datasets/open/train/hr.lmdb'
    img_path_list, keys = prepare_keys_dacon(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx4 images
    folder_path = 'datasets/open/train/lr'
    lmdb_path = 'datasets/open/train/lr.lmdb'
    img_path_list, keys = prepare_keys_dacon(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys_dacon(folder_path):
    """Prepare image path list and keys for Dacon dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


if __name__ == '__main__':
    create_lmdb_for_dacon()
