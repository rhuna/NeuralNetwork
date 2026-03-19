from __future__ import annotations
# Enables modern type hint behavior.

import random
# Used here to select a random test sample for prediction.

import torch
# Main PyTorch library.

from sklearn.datasets import load_digits
# Loads the digits dataset from scikit-learn.

from sklearn.preprocessing import StandardScaler
# Used to normalize the input data the same way as during training.

from src.neural_net_app.model import DigitClassifier
# Imports the same model architecture used in training.

from src.neural_net_app.utils import get_device
# Chooses GPU or CPU.


def main() -> None:
    # Main prediction entry point.

    device = get_device()
    # Chooses the device.

    print(f"Using device: {device}")
    # Shows which device is being used.

    digits = load_digits()
    # Loads the handwritten digits dataset.

    X = digits.data.astype("float32")
    # Gets the input pixel values and converts them to float32.

    y = digits.target
    # Gets the correct labels for the digit images.

    scaler = StandardScaler()
    # Creates a scaler object for normalization.

    X = scaler.fit_transform(X).astype("float32")
    # Standardizes the input features.
    # This helps the model perform better because inputs are scaled consistently.

    index = random.randint(0, len(X) - 1)
    # Picks a random sample from the dataset.

    sample = torch.tensor(X[index], dtype=torch.float32).unsqueeze(0).to(device)
    # Converts that sample into a PyTorch tensor.
    # unsqueeze(0) adds a batch dimension so shape becomes [1, 64].
    # .to(device) moves it to GPU or CPU.

    true_label = y[index]
    # Stores the correct label for comparison.

    model = DigitClassifier(input_size=64, num_classes=10).to(device)
    # Recreates the same network structure used during training.

    model.load_state_dict(torch.load("artifacts/digit_classifier.pt", map_location=device))
    # Loads the saved model weights from disk.
    # map_location=device ensures it loads onto the chosen device.

    model.eval()
    # Sets the model to evaluation mode.
    # Important because some layers behave differently during training vs inference.

    with torch.no_grad():
        # Disables gradient tracking to save memory and speed up inference.

        outputs = model(sample)
        # Runs the sample through the model.

        predicted_class = outputs.argmax(dim=1).item()
        # Finds the class with the highest score.
        # .item() converts the 1-value tensor into a normal Python number.

    print(f"True label: {true_label}")
    # Prints the correct answer.

    print(f"Predicted label: {predicted_class}")
    # Prints the model’s prediction.


if __name__ == "__main__":
    # Runs only when file is executed directly.
    main()