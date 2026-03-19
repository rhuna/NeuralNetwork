# Neural Network Repo (Python + PyTorch)

A complete, copy/paste-ready Python project that trains a simple feedforward neural network on the built-in handwritten digits dataset from `scikit-learn`.

## What this repo does

- Loads the digits dataset (8x8 images of handwritten numbers)
- Normalizes the input features
- Splits the data into train and test sets
- Trains a neural network with PyTorch
- Evaluates accuracy on the test set
- Saves the trained model
- Predicts classes for sample inputs

## Project structure

```text
neural_network_repo/
├── requirements.txt
├── README.md
├── train.py
├── predict.py
├── src/
│   └── neural_net_app/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       ├── engine.py
│       └── utils.py
└── tests/
    └── test_model_shapes.py
```

## 1. Create and activate a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Train the model

```bash
python train.py
```

You should see training loss/accuracy by epoch and then a final test accuracy.

## 4. Run prediction script

```bash
python predict.py
```

## Notes

- This is a **fully connected neural network (MLP)**.
- It is a great starter repo before moving into CNNs, RNNs, transformers, or custom datasets.
- The digits dataset is included with `scikit-learn`, so you do not need to manually download anything.

## Upgrade ideas

- Add dropout
- Add batch normalization
- Save metrics to CSV
- Add confusion matrix
- Swap digits dataset for MNIST or FashionMNIST
- Turn this into a Flask/FastAPI inference API

