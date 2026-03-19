from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.neural_net_app.data import get_data_loaders
from src.neural_net_app.engine import train_model, evaluate_model
from src.neural_net_app.model import DigitClassifier
from src.neural_net_app.utils import get_device, set_seed


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)

    model = DigitClassifier(input_size=64, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=20,
        enable_live_plot=True,
    )

    test_loss, test_accuracy = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "digit_classifier.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")
    print("Saved training graph to: training_curves.png")


if __name__ == "__main__":
    main()