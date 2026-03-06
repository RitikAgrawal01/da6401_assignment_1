"""
Optimization Algorithms
Implements: SGD, Momentum, NAG (Nesterov), RMSProp
"""

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseOptimizer:
    def __init__(self, learning_rate: float = 0.01, weight_decay: float = 0.0):
        self.lr           = learning_rate
        self.weight_decay = weight_decay   # L2 regularization

    def update(self, layers):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SGD  (mini-batch gradient descent)
# ---------------------------------------------------------------------------
class SGD(BaseOptimizer):
    
    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
            layer.b -= self.lr * layer.grad_b


# ---------------------------------------------------------------------------
# Momentum SGD
# ---------------------------------------------------------------------------
class Momentum(BaseOptimizer):
    
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.v_W  = None
        self.v_b  = None

    def _init_velocity(self, layers):
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        if self.v_W is None:
            self._init_velocity(layers)

        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W + self.weight_decay * layer.W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b

            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]


# ---------------------------------------------------------------------------
# Nesterov Accelerated Gradient (NAG)
# ---------------------------------------------------------------------------
class NAG(BaseOptimizer):
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta  = beta
        self.v_W   = None
        self.v_b   = None

    def _init_velocity(self, layers):
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        if self.v_W is None:
            self._init_velocity(layers)

        for i, layer in enumerate(layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            v_W_prev = self.v_W[i].copy()
            v_b_prev = self.v_b[i].copy()

            self.v_W[i] = self.beta * self.v_W[i] + grad_W
            self.v_b[i] = self.beta * self.v_b[i] + grad_b

            # Nesterov update: use next velocity estimate
            layer.W -= self.lr * (self.beta * self.v_W[i] + grad_W)
            layer.b -= self.lr * (self.beta * self.v_b[i] + grad_b)

# ---------------------------------------------------------------------------
# RMSProp
# ---------------------------------------------------------------------------
class RMSProp(BaseOptimizer):
    
    def __init__(self, learning_rate=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.eps  = eps
        self.s_W  = None
        self.s_b  = None

    def _init_cache(self, layers):
        self.s_W = [np.zeros_like(l.W) for l in layers]
        self.s_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        if self.s_W is None:
            self._init_cache(layers)

        for i, layer in enumerate(layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * grad_W ** 2
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * grad_b ** 2

            layer.W -= self.lr / (np.sqrt(self.s_W[i]) + self.eps) * grad_W
            layer.b -= self.lr / (np.sqrt(self.s_b[i]) + self.eps) * grad_b



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
OPTIMIZER_MAP = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp,
}


def get_optimizer(name: str, learning_rate: float = 0.01, weight_decay: float = 0.0, **kwargs):
    """Return instantiated optimizer by name."""
    name = name.lower()
    if name not in OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(OPTIMIZER_MAP.keys())}")
    return OPTIMIZER_MAP[name](learning_rate=learning_rate, weight_decay=weight_decay, **kwargs)
