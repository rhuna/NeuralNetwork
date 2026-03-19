from __future__ import annotations
# Enables modern type hints.

from dataclasses import dataclass
# Lets us define small structured classes easily.

from typing import Dict, List, Tuple
# Type hints for dictionaries, lists, and tuples.

import matplotlib.pyplot as plt
# Used for live training graphs.

import torch
# Main PyTorch library.

from torch import nn
# Neural network tools like modules and losses.

from torch.utils.data import DataLoader
# Lets us type-hint DataLoader arguments.


@dataclass
class EpochMetrics:
    # A simple data container for metrics from one epoch.
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


class LivePlotter:
    # Handles live-updating training graphs.

    def __init__(self) -> None:
        # Constructor for the plotting helper.

        plt.ion()
        # Turns on interactive plotting mode so the figure updates live.

        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(2, 1, figsize=(10, 8))
        # Creates one figure with 2 stacked subplots:
        # top = loss
        # bottom = accuracy

        self.epochs: List[int] = []
        # Stores epoch numbers.

        self.train_losses: List[float] = []
        # Stores training loss values.

        self.val_losses: List[float] = []
        # Stores validation loss values.

        self.train_accuracies: List[float] = []
        # Stores training accuracy values.

        self.val_accuracies: List[float] = []
        # Stores validation accuracy values.

        self.fig.suptitle("Training Progress")
        # Sets the title for the overall figure.

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_accuracy: float,
        val_accuracy: float,
    ) -> None:
        # Updates stored values and redraws the plots.

        self.epochs.append(epoch)
        # Adds current epoch number.

        self.train_losses.append(train_loss)
        # Adds current training loss.

        self.val_losses.append(val_loss)
        # Adds current validation loss.

        self.train_accuracies.append(train_accuracy)
        # Adds current training accuracy.

        self.val_accuracies.append(val_accuracy)
        # Adds current validation accuracy.

        self.ax_loss.clear()
        # Clears the loss subplot before redrawing.

        self.ax_acc.clear()
        # Clears the accuracy subplot before redrawing.

        self.ax_loss.plot(self.epochs, self.train_losses, label="Train Loss")
        # Draws training loss line.

        self.ax_loss.plot(self.epochs, self.val_losses, label="Val Loss")
        # Draws validation loss line.

        self.ax_loss.set_title("Loss")
        # Sets title for loss graph.

        self.ax_loss.set_xlabel("Epoch")
        # Sets x-axis label.

        self.ax_loss.set_ylabel("Loss")
        # Sets y-axis label.

        self.ax_loss.legend()
        # Shows legend.

        self.ax_loss.grid(True)
        # Adds grid to make graph easier to read.

        self.ax_acc.plot(self.epochs, self.train_accuracies, label="Train Accuracy")
        # Draws training accuracy line.

        self.ax_acc.plot(self.epochs, self.val_accuracies, label="Val Accuracy")
        # Draws validation accuracy line.

        self.ax_acc.set_title("Accuracy")
        # Sets title for accuracy graph.

        self.ax_acc.set_xlabel("Epoch")
        # Sets x-axis label.

        self.ax_acc.set_ylabel("Accuracy")
        # Sets y-axis label.

        self.ax_acc.legend()
        # Shows legend.

        self.ax_acc.grid(True)
        # Adds grid.

        self.fig.tight_layout()
        # Adjusts layout so labels don’t overlap.

        self.fig.canvas.draw()
        # Redraws the figure.

        self.fig.canvas.flush_events()
        # Forces GUI events to process.

        plt.pause(0.001)
        # Small pause so the graph can visibly update.

    def save(self, output_path: str = "training_curves.png") -> None:
        # Saves the current figure to an image file.

        self.fig.savefig(output_path, dpi=150, bbox_inches="tight")
        # Writes the image to disk.

    def close(self) -> None:
        # Finishes plotting session.

        plt.ioff()
        # Turns off interactive plotting mode.

        plt.show()
        # Keeps the final graph visible.


def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    # Runs one full pass through a dataset.
    # If optimizer is provided, it trains.
    # If optimizer is None, it only evaluates.

    training = optimizer is not None
    # Determines whether we are in training mode.

    model.train(training)
    # If training=True, model goes into training mode.
    # If training=False, model behaves in evaluation mode.

    total_loss = 0.0
    # Accumulates total loss across all batches.

    total_correct = 0
    # Counts how many predictions were correct.

    total_examples = 0
    # Counts total number of examples seen.

    for inputs, targets in loader:
        # Loops through batches from the data loader.

        inputs = inputs.to(device)
        # Moves input batch to GPU or CPU.

        targets = targets.to(device)
        # Moves target labels to GPU or CPU.

        if training:
            optimizer.zero_grad()
            # Clears old gradients before computing new ones.

        outputs = model(inputs)
        # Forward pass:
        # sends inputs through the neural network.

        loss = criterion(outputs, targets)
        # Computes how wrong the predictions are.

        if training:
            loss.backward()
            # Backward pass:
            # computes gradients of the loss w.r.t. model parameters.

            optimizer.step()
            # Uses gradients to update the model weights.

        total_loss += loss.item() * inputs.size(0)
        # Adds batch loss scaled by batch size.
        # loss.item() gives the average loss for the batch,
        # so multiplying by batch size lets us later compute a dataset-wide average.

        predictions = outputs.argmax(dim=1)
        # Picks the class with the highest output score for each example.

        total_correct += (predictions == targets).sum().item()
        # Counts how many predictions matched the true labels.

        total_examples += inputs.size(0)
        # Adds the number of examples in this batch.

    avg_loss = total_loss / total_examples
    # Computes average loss across the whole dataset.

    avg_accuracy = total_correct / total_examples
    # Computes accuracy across the whole dataset.

    return avg_loss, avg_accuracy
    # Returns both average loss and average accuracy.


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
    # Main training function.
    # Trains for multiple epochs and tracks history.

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    # Creates a dictionary to store metrics over time.

    plotter = LivePlotter() if enable_live_plot else None
    # Creates the plotting helper if live graphs are enabled.

    for epoch in range(1, epochs + 1):
        # Loops through all epochs.

        train_loss, train_accuracy = _run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        # Runs one full training epoch.

        with torch.no_grad():
            # Disables gradient tracking during validation to save memory/speed.

            val_loss, val_accuracy = _run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
            )
            # Runs one full validation pass.

        history["train_loss"].append(train_loss)
        # Stores training loss for this epoch.

        history["train_accuracy"].append(train_accuracy)
        # Stores training accuracy.

        history["val_loss"].append(val_loss)
        # Stores validation loss.

        history["val_accuracy"].append(val_accuracy)
        # Stores validation accuracy.

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f}"
        )
        # Prints a clean summary line for each epoch.

        if plotter is not None:
            # Only update graphs if plotting is enabled.

            plotter.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
            )
            # Sends latest metrics to the live graph.

    if plotter is not None:
        # If plotting was enabled, save and close graph at the end.

        plotter.save("training_curves.png")
        # Saves the figure as a PNG file.

        plotter.close()
        # Finalizes the plot display.

    return history
    # Returns all recorded metrics.


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    # Runs evaluation on a dataset and returns loss/accuracy.

    with torch.no_grad():
        # No gradients needed for evaluation.

        return _run_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )
        # Reuses the same one-epoch function in evaluation mode.