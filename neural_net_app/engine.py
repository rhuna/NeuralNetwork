from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


class LivePlotter:
    def __init__(self) -> None:
        plt.ion()
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(2, 1, figsize=(10, 8))
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []

        self.fig.suptitle("Training Progress")

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_accuracy: float,
        val_accuracy: float,
    ) -> None:
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)

        self.ax_loss.clear()
        self.ax_acc.clear()

        self.ax_loss.plot(self.epochs, self.train_losses, label="Train Loss")
        self.ax_loss.plot(self.epochs, self.val_losses, label="Val Loss")
        self.ax_loss.set_title("Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend()
        self.ax_loss.grid(True)

        self.ax_acc.plot(self.epochs, self.train_accuracies, label="Train Accuracy")
        self.ax_acc.plot(self.epochs, self.val_accuracies, label="Val Accuracy")
        self.ax_acc.set_title("Accuracy")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.legend()
        self.ax_acc.grid(True)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def save(self, output_path: str = "training_curves.png") -> None:
        self.fig.savefig(output_path, dpi=150, bbox_inches="tight")

    def close(self) -> None:
        plt.ioff()
        plt.show()


def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_examples += inputs.size(0)

    avg_loss = total_loss / total_examples
    avg_accuracy = total_correct / total_examples
    return avg_loss, avg_accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 20,
    enable_live_plot: bool = True,
) -> Dict[str, List[float]]:
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    plotter = LivePlotter() if enable_live_plot else None

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        with torch.no_grad():
            val_loss, val_accuracy = _run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
            )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f}"
        )

        if plotter is not None:
            plotter.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
            )

    if plotter is not None:
        plotter.save("training_curves.png")
        plotter.close()

    return history


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    with torch.no_grad():
        return _run_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )