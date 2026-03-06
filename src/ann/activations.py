"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


class ReLU:
    def forward(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return (z > 0).astype(float)


class Sigmoid:
    def forward(self, z):
        # Clip to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z):
        s = self.forward(z)
        return s * (1 - s)


class Tanh:
    def forward(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1.0 - np.tanh(z) ** 2


class Softmax:
    def forward(self, z):
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def derivative(self, z):
        s = self.forward(z)
        return s * (1 - s)


class Identity:
    def forward(self, z):
        return z

    def derivative(self, z):
        return np.ones_like(z)

# ---------------------------------------------------------------------------

# Factory helper
ACTIVATION_MAP = {
    "relu":     ReLU,
    "sigmoid":  Sigmoid,
    "tanh":     Tanh,
    "softmax":  Softmax,
    "identity": Identity,
    "linear":   Identity,
}


def get_activation(name: str):
    """Return an instantiated activation object by name string."""
    name = name.lower()
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATION_MAP.keys())}")
    return ACTIVATION_MAP[name]()
