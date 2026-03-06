"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets without keras/tensorflow dependency.
Downloads directly from source URLs using urllib.
"""

import os
import gzip
import struct
import numpy as np
import urllib.request


# ---------------------------------------------------------------------------
# Download URLs
# ---------------------------------------------------------------------------
MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

FASHION_MNIST_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images":  "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels":  "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

MNIST_CLASSES        = [str(i) for i in range(10)]
FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Cache directory — store downloaded files next to this script
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dataset_cache")


# ---------------------------------------------------------------------------
# IDX file parsers
# ---------------------------------------------------------------------------
def _parse_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)


def _parse_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------
def _download(url: str, dest: str):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)


def _get_dataset_files(name: str) -> dict:
    urls      = MNIST_URLS if name == "mnist" else FASHION_MNIST_URLS
    cache_dir = os.path.join(_CACHE_DIR, name)
    paths     = {}
    for key, url in urls.items():
        fname = url.split("/")[-1]
        dest  = os.path.join(cache_dir, fname)
        _download(url, dest)
        paths[key] = dest
    return paths


# ---------------------------------------------------------------------------
# Try keras first (faster), fall back to direct download
# ---------------------------------------------------------------------------
def _load_via_keras(name: str):
    """Try to load using keras — returns None if keras/tf not available."""
    try:
        if name == "mnist":
            from keras.datasets import mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
        else:
            from keras.datasets import fashion_mnist
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        # Flatten and normalize
        X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
        X_test  = X_test.reshape(-1, 784).astype(np.float32) / 255.0
        return (X_train, y_train), (X_test, y_test)
    except Exception:
        return None


def _load_via_download(name: str):
    """Load by downloading IDX files directly."""
    paths   = _get_dataset_files(name)
    X_train = _parse_images(paths["train_images"]).astype(np.float32) / 255.0
    y_train = _parse_labels(paths["train_labels"])
    X_test  = _parse_images(paths["test_images"]).astype(np.float32) / 255.0
    y_test  = _parse_labels(paths["test_labels"])
    return (X_train, y_train), (X_test, y_test)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_data(dataset: str = "mnist", val_split: float = 0.1):
    """
    Load MNIST or Fashion-MNIST dataset.

    Parameters
    ----------
    dataset   : 'mnist' or 'fashion_mnist'
    val_split : fraction of training data to use as validation

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names
    """
    assert dataset in ("mnist", "fashion_mnist"), \
        f"Unknown dataset '{dataset}'. Choose 'mnist' or 'fashion_mnist'."

    # Try keras first, fall back to direct download
    result = _load_via_keras(dataset)
    if result is None:
        print(f"Keras unavailable — downloading {dataset} directly...")
        result = _load_via_download(dataset)

    (X_train_full, y_train_full), (X_test, y_test) = result

    # Train / validation split
    n_val   = int(len(X_train_full) * val_split)
    idx     = np.random.permutation(len(X_train_full))
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    X_train, y_train = X_train_full[trn_idx], y_train_full[trn_idx]
    X_val,   y_val   = X_train_full[val_idx], y_train_full[val_idx]

    class_names = MNIST_CLASSES if dataset == "mnist" else FASHION_MNIST_CLASSES

    print(f"Loaded {dataset}: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names