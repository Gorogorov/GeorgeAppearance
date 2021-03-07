import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List


class GeorgeDataset(Dataset):
    def __init__(self, images_filepaths: str, transform=None):
        """
        Parameters
        ----------
        images_filepaths: str
        transform: object
            pytorch transform
        """
        self.images_filepaths = images_filepaths
        self.transform = transform
        self.targets = None

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx: int):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "georges":
            label = 1
        else:
            label = 0
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

    def get_targets(self):
        if self.targets is None:
            self.targets = []
            for fpath in self.images_filepaths:
                if os.path.normpath(fpath).split(os.sep)[-2] == "georges":
                    self.targets.append(1)
                else:
                    self.targets.append(0)
            self.targets = torch.Tensor(self.targets)
        return self.targets


class GeorgeDatasetTTA(GeorgeDataset):
    def __init__(self, images_filepaths: str, tta_attempts: int, transform=None):
        """
        Parameters
        ----------
        images_filepaths: str
        tta_attempts: int
        transform: object
            pytorch transform
        """
        self.images_filepaths = images_filepaths
        self.n_imgs = len(images_filepaths)
        self.transform = transform
        self.targets = None
        self.tta_attempts = tta_attempts

    def __len__(self):
        return self.n_imgs * self.tta_attempts

    def __getitem__(self, idx: int):
        image_filepath = self.images_filepaths[idx % self.n_imgs]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "georges":
            label = 1
        else:
            label = 0
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

    def get_n_attempts(self):
        return self.tta_attempts


def get_george_loaders(
    train: List[str],
    val: List[str],
    test: List[str],
    input_size: int,
    batch_size: int,
    num_workers: int,
    tta_attempts: int,
) -> dict:
    """
    Get train, validation and test Dataloaders using albumentations.
    Parameters
    ----------
    train: List[str]
        image train paths
    val: List[str]
        image validation paths
    test: List[str]
        image test paths
    input_size: int
        network's input size
    batch_size: int
    num_workers: int
    tta_attempts: int
    Returns
    ----------
    dataloaders_dict: dict
        {'train': pytorch Dataloader,
         'val': pytorch Dataloader,
         'test': pytorch Dataloader}
    """
    # create transforms
    train_trans = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=40, p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Blur(p=0.3),
            A.Cutout(8, p=0.3),
            A.SmallestMaxSize(max_size=input_size + 5),
            A.RandomCrop(height=input_size, width=input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_trans = A.Compose(
        [
            A.SmallestMaxSize(max_size=input_size + 5),
            A.CenterCrop(height=input_size, width=input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    # Test-Time Augmentations
    test_trans = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.2
            ),
            A.HorizontalFlip(p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
            A.Blur(p=0.2),
            A.SmallestMaxSize(max_size=input_size + 5),
            A.RandomCrop(height=input_size, width=input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # apply transforms
    train_data = GeorgeDataset(train, train_trans)
    val_data = GeorgeDataset(val, val_trans)
    test_data = GeorgeDatasetTTA(test, tta_attempts, test_trans)

    # create dataloaders
    dataloaders_dict = {
        "train": DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    return dataloaders_dict
