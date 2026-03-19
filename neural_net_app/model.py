from __future__ import annotations
# Enables cleaner modern type hint support.

import torch
# Main PyTorch library.

from torch import nn
# Imports PyTorch neural network modules.


class DigitClassifier(nn.Module):
    # Defines a neural network class for digit classification.
    # Inherits from nn.Module, which is the base class for PyTorch models.

    def __init__(self, input_size: int = 64, num_classes: int = 10) -> None:
        # Constructor for the model.
        # input_size is 64 because each image is 8x8 pixels flattened.
        # num_classes is 10 because we classify digits 0 through 9.

        super().__init__()
        # Calls the parent class constructor.
        # Required when inheriting from nn.Module.

        self.network = nn.Sequential(
            # nn.Sequential lets us stack layers in order.

            nn.Linear(input_size, 128),
            # First fully connected layer.
            # Takes 64 inputs and outputs 128 values.

            nn.ReLU(),
            # ReLU activation introduces nonlinearity.
            # Without this, the model would just behave like a simple linear function.

            nn.Linear(128, 64),
            # Second fully connected layer.
            # Takes 128 inputs and outputs 64 values.

            nn.ReLU(),
            # Another ReLU activation.

            nn.Linear(64, num_classes),
            # Final output layer.
            # Takes 64 inputs and outputs 10 scores, one for each digit class.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Defines how data flows through the model during prediction/training.

        return self.network(x)
        # Passes input x through the full stack of layers and returns the output.