"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy (with softmax), Mean Squared Error (MSE)
"""

import numpy as np


# Helper: softmax 
def _softmax(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


# Helper: one-hot encoding
def _one_hot(y, num_classes):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y.astype(int)] = 1.0
    return oh


# Cross-Entropy Loss  (combined with softmax)
class CrossEntropyLoss:
    
    def forward(self, logits, y_true):
        self.probs  = _softmax(logits)               # (b, C)
        self.y_true = y_true.astype(int)
        self.num_classes = logits.shape[1]

        # Clip for numerical stability
        probs_clipped = np.clip(self.probs, 1e-12, 1.0)
        batch_size = logits.shape[0]

        # Select the probability for the true class
        correct_log_probs = -np.log(probs_clipped[np.arange(batch_size), self.y_true])
        return float(np.mean(correct_log_probs))

    def backward(self):
        batch_size = self.probs.shape[0]
        y_oh = _one_hot(self.y_true, self.num_classes)
        # Return gradient w.r.t. logits (not divided by batch; layer.backward divides it)
        return (self.probs - y_oh)


# Mean Squared Error  (MSE)
class MSELoss:

    def forward(self, logits, y_true):
        self.probs  = _softmax(logits)
        self.y_true = y_true.astype(int)
        self.num_classes = logits.shape[1]

        y_oh = _one_hot(self.y_true, self.num_classes)   # (b, C)
        self.diff = self.probs - y_oh                     # (b, C)
        return float(np.mean(self.diff ** 2))

    def backward(self) -> np.ndarray:
        C = self.num_classes
        d_probs = (2.0 / C) * self.diff
        dot = np.sum(d_probs * self.probs, axis=-1, keepdims=True)
        return self.probs * (d_probs - dot)                       # (b, C)


# ---------------------------------------------------------------------------

# Factory
LOSS_MAP = {
    "cross_entropy": CrossEntropyLoss,
    "mse":           MSELoss,
}


def get_loss(name: str):
    name = name.lower()
    if name not in LOSS_MAP:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSS_MAP.keys())}")
    return LOSS_MAP[name]()
