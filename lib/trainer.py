import os
import time
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import logging
from typing import List, Tuple, Any


class ClassifierModel:
    def __init__(
        self, model: nn.Module, device: torch.device, n_classes: int, name: str
    ) -> None:
        """
        Train.
        Validate.
        Predict.

        Parameters
        ----------
        model: torch.nn.Module
        device: torch.device
        n_classes: int
        name: str
            best model path
        """
        self.model = model
        self.device = device
        self.n_classes = n_classes
        self.best_m_path = os.path.join("models", f"{name}.pt")
        self.history = {
            "train_acc": [],
            "train_loss": [],
            "val_acc": [],
            "val_loss": [],
            "lr": [],
        }

    @staticmethod
    def get_lr(optimizer: Optimizer):
        """
        Get current learning rate from the optimizer.
        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            pytorch optimizer
        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    @staticmethod
    def fine_tune_prepare_model(model: nn.Module, n_classes: int) -> nn.Module:
        """
        Change last Linear layer to Linear with n_classes neurons
        on the top if it exists.
        Require gradients from all model parameters.
        Parameters
        ----------
        model: torch.nn.Module
            pytorch model
        n_classes: int
        Returns
        ----------
        model: torch.nn.Module
        """
        # find last Linear layer
        if hasattr(model, "fc"):
            n_ftrs = model.fc.in_features
            model.fc = nn.Linear(n_ftrs, n_classes)
        elif hasattr(model, "_fc"):
            n_ftrs = model._fc.in_features
            model._fc = nn.Linear(n_ftrs, n_classes)
        for param in model.parameters():
            param.require_grad = True
        return model

    def save_model(self) -> None:
        """
        Save the model to the models/name path
        (name is a constructor's parameter)
        """
        torch.save(self.model.state_dict(), self.best_m_path)

    def load_model(self) -> None:
        """
        Load a model from the models/name path
        (name is a constructor's parameter)
        """
        self.model = self.fine_tune_prepare_model(self.model, self.n_classes)
        if os.path.exists(self.best_m_path):
            self.model.load_state_dict(
                torch.load(self.best_m_path, map_location=self.device)
            )
        else:
            logging.info("Model does not exist")

    def _run_batch(self, batch: list, train: bool = True) -> Tuple[Any, Any, Any]:
        """
        Process one batch.
        Parameters
        ----------
        batch: list
            batch from pytorch Dataloader
        train: bool
            if True, returns loss and batch_corrects
            if False, returns Nones instead
        Returns
        ----------
        loss: object
            pytorch loss
        outputs: object
            probabilities
        batch_corrects: object
            number of true predicts
        """
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        loss, batch_corrects = None, None
        if train:
            loss = self.criterion(outputs, labels)
            batch_corrects = torch.sum(preds == labels.data)
        return loss, outputs, batch_corrects

    def _run_epoch(self, dataloaders: dict):
        """
        Train model one epoch and save accuracy
        and loss to the history.
        Parameters
        ----------
        dataloaders: dict
            dataloaders = {'train': Dataloader,
                           'val': Dataloader}
        """
        for phase in ["train", "val"]:
            if phase == "train":
                self.model.train()
            else:
                self.model.eval()
            running_loss = 0
            running_corrects = 0
            iter_batches = tqdm(dataloaders[phase], total=len(dataloaders[phase]))
            with torch.set_grad_enabled(phase == "train"):
                for _, batch in enumerate(iter_batches):
                    loss, _, batch_corrects = self._run_batch(batch)
                    self.optimizer.zero_grad()
                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()

                    # statistics
                    running_loss += float(loss.item()) * len(batch)
                    running_corrects += int(batch_corrects)

            # update history
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            self.history[f"{phase}_acc"].append(epoch_acc)
            self.history[f"{phase}_loss"].append(epoch_loss)

        self.scheduler.step(self.history["val_acc"][-1])
        self.history["lr"].append(self.get_lr(self.optimizer))

    def train(
        self,
        dataloaders: dict,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optimizer,
        early_stopping: int,
        n_epochs: int,
    ) -> dict:
        """
        Train model and save best model to the models/name path
        (name is a constructor's parameter)
        Parameters
        ----------
        dataloaders: dict
            dataloaders = {'train': Dataloader,
                           'val': Dataloader}
        criterion: torch.nn.Module
            pytorch criterion
        optimizer: torch.optim.Optimizer
            pytorch optimizer
        scheduler: torch.optim.Optimizer
            pytorch lr_scheduler
        early_stopping: int
            stop train if validation accuracy does not
            increase early_stopping epochs
        n_epochs: int
        Returns
        ----------
        history: dict
            history = {
            'train_acc': List[int],
            'train_loss': List[int],
            'val_acc': List[int],
            'val_loss': List[int],
            'lr': List[int]
        }
        """
        since = time.time()
        self.model = self.fine_tune_prepare_model(self.model, self.n_classes)
        self.model = self.model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        n_train = len(dataloaders["train"])
        n_val = len(dataloaders["val"])
        batch_size = dataloaders["train"].batch_size

        logging.info(
            f"""Starting training:
            Epochs:          {n_epochs}
            Batch size:      {batch_size}
            Learning rate:   {self.get_lr(optimizer)}
            Training size:   {n_train}
            Validation size: {n_val}
            Best model path: {self.best_m_path}
            Device:          {self.device.type}
        """
        )

        best_model_wts = deepcopy(self.model.state_dict())
        best_acc = 0

        for epoch in range(n_epochs):
            logging.info(f"Epoch {epoch + 1}/{n_epochs}")
            self._run_epoch(dataloaders)
            # choose best model
            if self.history["val_acc"][-1] > best_acc:
                best_acc = self.history["val_acc"][-1]
                best_model_wts = deepcopy(self.model.state_dict())
            if (
                epoch >= early_stopping
                and max(self.history["val_acc"][-early_stopping:])
                == self.history["val_acc"][-early_stopping]
            ):
                logging.info(
                    f"Early stopping, best validation "
                    f"accuracy = {self.history['val_acc'][-early_stopping]}"
                )
                break

            logging.info(
                f"""
                Train loss:      {self.history['train_loss'][-1]:4f}
                Val loss:        {self.history['val_loss'][-1]:4f}
                Train accuracy:  {self.history['train_acc'][-1]:4f}
                Val accuracy:    {self.history['val_acc'][-1]:4f}
            """
            )

        time_elapsed = time.time() - since
        logging.info(
            f"Training complete in {time_elapsed // 60:.0f}m "
            f"{time_elapsed % 60:.0f}s"
        )
        logging.info(f"Best val Acc: {best_acc:4f}")
        self.model.load_state_dict(best_model_wts)
        self.save_model()
        return self.history

    @torch.no_grad()
    def predict_proba(self, test: DataLoader) -> torch.Tensor:
        """
        Predict probabilites using Test-Time Augmentations.
        Parameters
        ----------
        test: torch.Dataloader
        Returns
        ----------
        preds: (n_objects, n_classes) torch.Tensor
            probabilites
        """
        self.model = self.model.to(self.device)
        self.model.eval()
        tta_attempts = test.dataset.get_n_attempts()
        n_samples = int(len(test.dataset) / tta_attempts)
        logging.info(
            f"""
            TTA attempts:       {tta_attempts}
            number of samples:  {n_samples}
        """
        )
        tta_preds = torch.empty((0, self.n_classes))
        iter_batches = tqdm(test, total=len(test))
        for _, batch in enumerate(iter_batches):
            _, batch_preds, _ = self._run_batch(batch, train=False)
            tta_preds = torch.vstack((tta_preds, batch_preds.cpu().detach()))
        # accumulate and average TTA predictions
        preds = torch.zeros((n_samples, self.n_classes))
        for idx in range(tta_attempts):
            preds += tta_preds[idx * n_samples : (idx + 1) * n_samples, :]
        preds /= tta_attempts
        sm = nn.Softmax(dim=1)
        preds = sm(preds)
        return preds

    @torch.no_grad()
    def predict(self, test_data: DataLoader) -> torch.Tensor:
        """
        Predict classes using Test-Time Augmentations.
        Parameters
        ----------
        test_data: torch.Dataloader
        Returns
        ----------
        preds: (n_objects, ) torch.Tensor
            hard decisions
        """
        preds = self.predict_proba(test_data)
        _, preds = torch.max(preds, 1)
        return preds
