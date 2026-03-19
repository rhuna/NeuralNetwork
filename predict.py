from pathlib import Path

import torch

from src.neural_net_app.data import load_digits_arrays
from src.neural_net_app.model import DigitClassifier
from src.neural_net_app.utils import get_device


def main() -> None:
    device = get_device()
    model_path = Path("artifacts/digit_classifier.pt")

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train.py` first."
        )

    checkpoint = torch.load(model_path, map_location=device)
    model = DigitClassifier(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, x_test, y_test = load_digits_arrays(test_size=0.2, random_state=42)

    sample_count = 10
    inputs = torch.tensor(x_test[:sample_count], dtype=torch.float32).to(device)
    labels = y_test[:sample_count]

    with torch.no_grad():
        logits = model(inputs)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    print("Sample predictions:")
    for idx, (pred, actual) in enumerate(zip(predictions, labels), start=1):
        print(f"  sample {idx:02d}: predicted={pred} actual={actual}")


if __name__ == "__main__":
    main()
