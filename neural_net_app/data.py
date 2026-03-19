from __future__ import annotations
# Enables modern type hint usage.

from typing import Tuple
# Lets us type-hint a function returning multiple items.

import numpy as np
# Used for array handling.

import torch
# Main PyTorch library.

from sklearn.datasets import load_digits
# Loads the sklearn handwritten digits dataset.

from sklearn.model_selection import train_test_split
# Splits data into train/validation/test sets.

from sklearn.preprocessing import StandardScaler
# Standardizes features for better model training.

from torch.utils.data import DataLoader, TensorDataset
# TensorDataset wraps tensors into dataset objects.
# DataLoader creates batch iterators.


def get_data_loaders(
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Loads data, preprocesses it, splits it,
    # and returns train/validation/test DataLoaders.

    digits = load_digits()
    # Loads the full digits dataset.

    X = digits.data.astype(np.float32)
    # Input features: each image flattened into 64 pixel values.
    # Converted to float32 because neural networks commonly use float32.

    y = digits.target.astype(np.int64)
    # Labels: integers from 0 to 9.
    # Converted to int64 because PyTorch classification expects long/int64 labels.

    scaler = StandardScaler()
    # Creates a standardization object.

    X = scaler.fit_transform(X).astype(np.float32)
    # Standardizes the data:
    # each feature gets centered around 0 and scaled to unit variance.
    # This often helps training converge more smoothly.

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Splits data into:
    # - 70% training
    # - 30% temporary set
    # stratify=y keeps class distribution balanced.

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    # Splits the temporary 30% into:
    # - 15% validation
    # - 15% test
    # Again keeps class balance.

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    # Wraps training inputs and labels into a PyTorch dataset object.

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    # Wraps validation inputs and labels.

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    # Wraps test inputs and labels.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Creates an iterator for training data.
    # shuffle=True randomizes order each epoch, which helps training.

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Validation data loader.
    # No need to shuffle validation data.

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Test data loader.
    # No need to shuffle test data either.

    return train_loader, val_loader, test_loader
    # Returns all 3 loaders.