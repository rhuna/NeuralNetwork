from __future__ import annotations
# Lets us use modern type hints cleanly, even in older-style annotations.

from pathlib import Path
# Path gives us a cleaner, safer way to work with file/folder paths.

import torch
# Main PyTorch library.

from torch import nn
# Imports PyTorch neural network tools like loss functions.

from src.neural_net_app.data import get_data_loaders
# Imports the function that loads and prepares our dataset.

from src.neural_net_app.engine import train_model, evaluate_model
# Imports the training loop and evaluation function.

from src.neural_net_app.model import DigitClassifier
# Imports our neural network model class.

from src.neural_net_app.utils import get_device, set_seed
# Imports helper functions for reproducibility and device selection.


def main() -> None:
    # Main entry point for training the neural network.

    set_seed(42)
    # Fixes randomness so results are more reproducible.
    # This helps make runs more consistent.

    device = get_device()
    # Chooses GPU if available, otherwise CPU.

    print(f"Using device: {device}")
    # Prints which device PyTorch will use.

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
    # Loads the dataset and returns 3 data loaders:
    # - training data
    # - validation data
    # - test data
    # Each loader gives batches of 64 samples at a time.

    model = DigitClassifier(input_size=64, num_classes=10).to(device)
    # Creates the neural network.
    # input_size=64 because each digit image is 8x8 = 64 pixels.
    # num_classes=10 because digits go from 0 to 9.
    # .to(device) moves the model to GPU or CPU.

    criterion = nn.CrossEntropyLoss()
    # Defines the loss function.
    # CrossEntropyLoss is standard for multi-class classification.

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Defines the optimizer.
    # Adam updates the model’s weights using gradients.
    # lr=1e-3 means learning rate = 0.001.

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
    # Trains the model for 20 epochs.
    # Also shows live training graphs if enabled.
    # The returned history contains loss/accuracy values over time.

    test_loss, test_accuracy = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    # Evaluates the fully trained model on the test set.
    # This tells us how well it performs on unseen data.

    print(f"\nFinal Test Loss: {test_loss:.4f}")
    # Prints final test loss rounded to 4 decimal places.

    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    # Prints final test accuracy rounded to 4 decimal places.

    output_dir = Path("artifacts")
    # Creates a path object for the artifacts folder.

    output_dir.mkdir(exist_ok=True)
    # Makes the folder if it does not already exist.
    # exist_ok=True prevents an error if it already exists.

    model_path = output_dir / "digit_classifier.pt"
    # Builds the full path where the trained model will be saved.

    torch.save(model.state_dict(), model_path)
    # Saves only the model weights, not the full model object.
    # This is a common and clean PyTorch practice.

    print(f"Saved model to: {model_path}")
    # Confirms where the model was saved.

    print("Saved training graph to: training_curves.png")
    # Confirms the graph image file was saved.


if __name__ == "__main__":
    # Runs main() only if this file is executed directly.
    main()