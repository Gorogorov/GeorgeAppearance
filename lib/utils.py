import os
import shutil
import random
from typing import List, Tuple, Any
from urllib.request import urlretrieve
from tqdm import tqdm


def load_dataset(
    urls: str, class_name: str, out_dir_path: str = "data/images", ext: str = "jpg"
) -> str:
    """
    Download files from urls file to the
    out_dir_path/class_name directory line by line.
    Add ext extension to the downloaded files.
    Create out_dir_path/class_name directory
    if it does not exist.
    Parameters
    ----------
    urls: str
    class_name: str
    out_dir_path: str
        path to folder with dataset.
    ext: str
    Returns
    -------
    class_dir: str
        out_dir_path/class_name
    """
    class_dir = os.path.join(out_dir_path, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    num_urls = sum(1 for url in open(urls))
    with open(urls, "r") as f:
        for idx, url in enumerate(tqdm(f, total=num_urls)):
            url = url.strip("\n")
            img_name = os.path.join(class_dir, f"{idx:08}.{ext}")
            urlretrieve(url, img_name)
    return class_dir


def train_val_test_split(
    class_dirs: List[str], train_size: int, val_size: int
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split the list with image paths class_dirs into
    train [0:train_size],
    validation [train_size:val_size]
    and test [val_size:] lists.
    Parameters
    ----------
    class_dirs: List[str]
        image paths
    train_size: int
    val_size: int
    Returns
    -------
    train_paths: List[str]
    val_paths: List[str]
    test_paths: List[str]
    """
    image_paths = []
    for d in class_dirs:
        pths = sorted([os.path.join(d, f) for f in os.listdir(d)])
        image_paths.extend(pths)

    num_imgs = len(image_paths)
    random.shuffle(image_paths)

    split = int(train_size * num_imgs)
    split2 = int((train_size + val_size) * num_imgs)
    train_paths = image_paths[:split]
    val_paths = image_paths[split:split2]
    test_paths = image_paths[split2:]
    return train_paths, val_paths, test_paths


def remove_dir(path: str) -> None:
    """
    Just remove directory.
    Parameters
    ----------
    dir_path: str
    """
    shutil.rmtree(path)
