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
# Adam
# ---------------------------------------------------------------------------
class Adam(BaseOptimizer):
    """
    m_W  ←  b1 * m_W + (1 - b1) * grad_W          (first moment)
    v_W  ←  b2 * v_W + (1 - b2) * grad_W^2         (second moment)
    m̂_W  =  m_W / (1 - b1^t)                        (bias correction)
    v̂_W  =  v_W / (1 - b2^t)
    W    ←  W - lr * m̂_W / (sqrt(v̂_W) + eps)
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0      # time step
        self.m_W   = None
        self.v_W   = None
        self.m_b   = None
        self.v_b   = None

    def _init_moments(self, layers):
        self.m_W = [np.zeros_like(l.W) for l in layers]
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.m_b = [np.zeros_like(l.b) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        if self.m_W is None:
            self._init_moments(layers)

        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for i, layer in enumerate(layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            # Update moments
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grad_W
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * grad_W ** 2
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_b ** 2

            layer.W -= lr_t * self.m_W[i] / (np.sqrt(self.v_W[i]) + self.eps)
            layer.b -= lr_t * self.m_b[i] / (np.sqrt(self.v_b[i]) + self.eps)


# ---------------------------------------------------------------------------
# Nadam  (Nesterov + Adam)
# ---------------------------------------------------------------------------
class Nadam(BaseOptimizer):
    """
    Same as Adam but uses the Nesterov update for the first moment:
    W ← W - lr_t * (b1*m̂_W + (1-b1)*grad_W/(1-b1^t)) / (sqrt(v̂_W) + eps)
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m_W   = None
        self.v_W   = None
        self.m_b   = None
        self.v_b   = None

    def _init_moments(self, layers):
        self.m_W = [np.zeros_like(l.W) for l in layers]
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.m_b = [np.zeros_like(l.b) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        if self.m_W is None:
            self._init_moments(layers)

        self.t += 1
        b1, b2 = self.beta1, self.beta2

        for i, layer in enumerate(layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.m_W[i] = b1 * self.m_W[i] + (1 - b1) * grad_W
            self.v_W[i] = b2 * self.v_W[i] + (1 - b2) * grad_W ** 2
            self.m_b[i] = b1 * self.m_b[i] + (1 - b1) * grad_b
            self.v_b[i] = b2 * self.v_b[i] + (1 - b2) * grad_b ** 2

            # Bias-corrected
            m_hat_W = self.m_W[i] / (1 - b1 ** self.t)
            v_hat_W = self.v_W[i] / (1 - b2 ** self.t)
            m_hat_b = self.m_b[i] / (1 - b1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - b2 ** self.t)

            # Nesterov correction
            lr_t = self.lr / (1 - b1 ** self.t)
            nesterov_W = b1 * m_hat_W + (1 - b1) * grad_W / (1 - b1 ** self.t)
            nesterov_b = b1 * m_hat_b + (1 - b1) * grad_b / (1 - b1 ** self.t)

            layer.W -= self.lr * nesterov_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= self.lr * nesterov_b / (np.sqrt(v_hat_b) + self.eps)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
OPTIMIZER_MAP = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp,
    "adam":     Adam,
    "nadam":    Nadam,
}


def get_optimizer(name: str, learning_rate: float = 0.01, weight_decay: float = 0.0, **kwargs):
    """Return instantiated optimizer by name."""
    name = name.lower()
    if name not in OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(OPTIMIZER_MAP.keys())}")
    return OPTIMIZER_MAP[name](learning_rate=learning_rate, weight_decay=weight_decay, **kwargs)
