import torch
# PyTorch library.

from src.neural_net_app.model import DigitClassifier
# Imports the model to test.


def test_model_output_shape() -> None:
    # Simple test to make sure the model output shape is correct.

    model = DigitClassifier(input_size=64, num_classes=10)
    # Creates the model.

    x = torch.randn(32, 64)
    # Creates a fake batch of 32 samples.
    # Each sample has 64 input features.

    y = model(x)
    # Runs the fake batch through the model.

    assert y.shape == (32, 10)
    # Checks that output shape is correct:
    # 32 samples in the batch
    # 10 class scores per sample