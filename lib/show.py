import matplotlib.pyplot as plt
import torch
import cv2
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader


def val_lr_curves(hist: dict, name: str) -> None:
    """
    Show train and validation curves (accuracy and loss)
    and learning rate variation
    Parameters
    ----------
    hist: dict
    n_epochs: int
    name: str
        name of the model
    """
    n_epochs = len(hist['train_acc'])
    epochs = list(range(1, n_epochs + 1))
    metrics = ["acc", "loss"]

    plt.figure(figsize=(25, 6))
    for i in range(2):
        plt.subplot(1, 3, i + 1)
        plt.plot(epochs, hist[f"train_{metrics[i]}"], "-k", label="train")
        plt.plot(epochs, hist[f"val_{metrics[i]}"], "-r", label="validation")
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel(f"{metrics[i]}")
        plt.title(f"{name} {metrics[i]}")
        plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs, hist["lr"], "-k")
    plt.grid()
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("learing rate")
    plt.title(f"{name} learning rate")


def show_roc_curve(preds: torch.Tensor, targets: torch.Tensor, name: str) -> None:
    """
    Show ROC curve using sklearn.metrics.roc_curve
    Parameters
    ----------
    preds: (n_samples,) or (n_samples, n_classes) torch.Tensor
    targets: (n_samples,) or (n_samples, n_classes) torch.Tensor
    """
    y_true = targets.numpy()
    y_pred = preds.numpy()
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(11, 9))
    plt.plot(fpr, tpr, "-k")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{name} ROC curve")


def show_fp_fn(
    test: DataLoader, preds: torch.Tensor, targets: torch.Tensor, n_imgs: int
) -> None:
    """
    Show False Positive and False negative images
    Parameters
    ----------
    test: pytorch Dataloader
    preds: (n_samples,) torch.Tensor
    targets: (n_samples,) torch.Tensor
    n_imgs: int
        max number of output images
    """

    fn = (targets > preds).nonzero(as_tuple=True)[0]
    fp = (targets < preds).nonzero(as_tuple=True)[0]
    n_fp = min(fp.shape[0], n_imgs)
    n_fn = min(fn.shape[0], n_imgs)

    plt.figure(figsize=(16, 17))
    if n_fp > 0:
        for i in range(n_fp):
            plt.subplot(1, n_fp, i + 1)
            plt.title(f"False positive {i+1}")
            plt.axis("off")
            img = cv2.imread(test.dataset.images_filepaths[fp[i]])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)

    plt.figure(figsize=(16, 17))
    if n_fn > 0:
        for i in range(n_fn):
            plt.subplot(1, n_fn, i + 1)
            plt.title(f"False negative {i+1}")
            plt.axis("off")
            img = cv2.imread(test.dataset.images_filepaths[fn[i]])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
