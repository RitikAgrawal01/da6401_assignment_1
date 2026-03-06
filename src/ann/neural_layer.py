"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""


import numpy as np
from ann.activations import get_activation


class NeuralLayer:
    """
    A single fully-connected layer.

    Parameters
    ----------
    input_size  : int   - number of input features (D_in)
    output_size : int   - number of output neurons  (D_out)
    activation  : str   - activation function name ('relu', 'sigmoid', 'tanh', etc.)
    weight_init : str   - 'random' or 'xavier'
    """

    def __init__(self, input_size: int, output_size: int,
                 activation: str = "relu", weight_init: str = "xavier"):
        self.input_size  = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.activation_fn   = get_activation(activation)

        # Initialize weights and biases
        self.W, self.b = self._initialize_weights(weight_init)

        # Cache for forward pass values (needed by backprop)
        self.X = None   # input to this layer  shape (b, D_in)
        self.z = None   # pre-activation       shape (b, D_out)
        self.a = None   # post-activation      shape (b, D_out)

        # Gradients (set during backward pass)
        self.grad_W = None   # shape (D_in, D_out)
        self.grad_b = None   # shape (1, D_out)

    # ------------------------------------------------------------------
    # Weight Initialization
    # ------------------------------------------------------------------
    def _initialize_weights(self, method: str):
        """
        Initialize W and b.

        random : W ~ N(0, 0.01),  b = 0
        xavier : W ~ N(0, sqrt(2 / (fan_in + fan_out))), b = 0
                 (also known as Glorot normal initialization)
        zeros  : W = 0, b = 0  (useful to demonstrate symmetry breaking)
        """
        method = method.lower()

        if method == "random":
            W = np.random.randn(self.input_size, self.output_size) * 0.01

        elif method == "xavier":
            std = np.sqrt(2.0 / (self.input_size + self.output_size))
            W   = np.random.randn(self.input_size, self.output_size) * std

        elif method == "zeros":
            W = np.zeros((self.input_size, self.output_size))

        else:
            raise ValueError(f"Unknown weight init '{method}'. Use 'random', 'xavier', or 'zeros'.")

        b = np.zeros((1, self.output_size))
        return W, b

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute z = X @ W + b, then a = activation(z).

        Parameters
        ----------
        X : ndarray of shape (batch_size, input_size)

        Returns
        -------
        a : ndarray of shape (batch_size, output_size)
        """
        self.X = X                              # cache input for backprop
        self.z = X @ self.W + self.b            # linear transform
        self.a = self.activation_fn.forward(self.z)  # apply activation
        return self.a

    # ------------------------------------------------------------------
    # Backward Pass
    # ------------------------------------------------------------------
    def backward(self, delta: np.ndarray, weight_decay: float = 0.0) -> np.ndarray:
        """
        Compute gradients for W, b and propagate delta to previous layer.

        Parameters
        ----------
        delta       : ndarray of shape (batch_size, output_size)
                      gradient of loss w.r.t. this layer's PRE-activation z
                      (i.e. dL/dz for this layer)
        weight_decay: L2 regularization coefficient (lambda)

        Returns
        -------
        delta_prev  : ndarray of shape (batch_size, input_size)
                      gradient to pass to the previous layer (dL/dz_prev)
        """
        batch_size = self.X.shape[0]

        # Gradient w.r.t. W: (D_in, b) @ (b, D_out) / b  +  L2 term
        self.grad_W = (self.X.T @ delta) / batch_size + weight_decay * self.W

        # Gradient w.r.t. b: mean over batch
        self.grad_b = np.mean(delta, axis=0, keepdims=True)

        # Gradient to pass backwards: (b, D_out) @ (D_out, D_in) = (b, D_in)
        delta_prev = delta @ self.W.T
        return delta_prev
