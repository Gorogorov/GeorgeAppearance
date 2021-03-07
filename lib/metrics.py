import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score


def accuracy_value(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Return accuracy.
    Parameters
    ----------
    preds: (n_samples,) torch.Tensor
    targets: (n_samples,) torch.Tensor
    """
    return float(torch.mean((preds == targets) * 1.0))


def roc_auc_value(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Return ROC AUC score using sklearn.metrics.roc_auc_score
    Parameters
    ----------
    preds: (n_samples,) or (n_samples, n_classes) torch.Tensor
    targets: (n_samples,) or (n_samples, n_classes) torch.Tensor
    """
    y_true = targets.numpy()
    y_pred = preds.numpy()
    return roc_auc_score(y_true, y_pred)


def precision_value(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Return precision using sklearn.metrics.precision_score
    Parameters
    ----------
    preds: (n_samples,) torch.Tensor
    targets: (n_samples,) torch.Tensor
    """
    y_true = targets.numpy()
    y_pred = preds.numpy()
    return precision_score(y_true, y_pred)


def recall_value(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Return recall using sklearn.metrics.recall_score
    Parameters
    ----------
    preds: (n_samples,) torch.Tensor
    targets: (n_samples,) torch.Tensor
    """
    y_true = targets.numpy()
    y_pred = preds.numpy()
    return recall_score(y_true, y_pred)


def f1_value(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Return F1 score using sklearn.metrics.f1_score
    Parameters
    ----------
    preds: (n_samples,) torch.Tensor
    targets: (n_samples,) torch.Tensor
    """
    y_true = targets.numpy()
    y_pred = preds.numpy()
    return f1_score(y_true, y_pred)
