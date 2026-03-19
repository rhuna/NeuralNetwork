import torch

from src.neural_net_app.model import DigitClassifier


def test_model_output_shape() -> None:
    model = DigitClassifier(input_size=64, num_classes=10)
    batch = torch.randn(32, 64)
    output = model(batch)
    assert output.shape == (32, 10)
