"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Class names for display purposes
MNIST_CLASSES = [str(i) for i in range(10)]

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def load_data(dataset: str = "fashion_mnist",
              val_split: float = 0.1,
              flatten: bool = True,
              normalize: bool = True):
    
    """ Load MNIST or Fashion-MNIST and split into train / validation / test. """
    
    # Load via keras
    from keras.datasets import mnist, fashion_mnist

    dataset = dataset.lower()
    if dataset == "mnist":
        (X_tr, y_tr), (X_te, y_te) = mnist.load_data()
        class_names = MNIST_CLASSES
    elif dataset == "fashion_mnist":
        (X_tr, y_tr), (X_te, y_te) = fashion_mnist.load_data()
        class_names = FASHION_CLASSES
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'mnist' or 'fashion_mnist'.")

    # Normalize
    if normalize:
        X_tr = X_tr.astype(np.float64) / 255.0
        X_te = X_te.astype(np.float64) / 255.0
    else:
        X_tr = X_tr.astype(np.float64)
        X_te = X_te.astype(np.float64)

    # Flatten images: (N, 28, 28) → (N, 784)
    if flatten:
        X_tr = X_tr.reshape(X_tr.shape[0], -1)
        X_te = X_te.reshape(X_te.shape[0], -1)

    # Train / validation split using scikit-learn
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, 
        test_size=val_split, 
        random_state=42, 
        stratify=y_tr
    )

    print(f"Dataset: {dataset}")
    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Val   : {X_val.shape[0]}   samples")
    print(f"  Test  : {X_te.shape[0]}   samples")
    print(f"  Input : {X_train.shape[1]} features")

    return (X_train, y_train), (X_val, y_val), (X_te, y_te), class_names