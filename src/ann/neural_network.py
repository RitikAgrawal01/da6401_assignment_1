"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer, NAG

class NeuralNetwork:

    def __init__(self, cli_args):
        # Support both Namespace objects and dicts
        if isinstance(cli_args, dict):
            args = type("Args", (), cli_args)()
        else:
            args = cli_args

        self.args = args

        # Build layer list
        self.layers = self._build_layers(args)

        # Loss and optimizer
        self.loss_fn   = get_loss(getattr(args, "loss", "cross_entropy"))
        self.optimizer = get_optimizer(
            name          = getattr(args, "optimizer", "rmsprop"),
            learning_rate = getattr(args, "learning_rate", 0.001),
            weight_decay  = getattr(args, "weight_decay", 0.0),
        )

        # Gradient arrays
        self.grad_W = None
        self.grad_b = None
        
    def _build_layers(self, args):
        
        input_size   = getattr(args, "input_size",   784)
        output_size  = getattr(args, "output_size",  10)
        num_layers   = getattr(args, "num_layers",   3)
        hidden_size  = getattr(args, "hidden_size",  128)
        activation   = getattr(args, "activation",   "relu")
        weight_init  = getattr(args, "weight_init",  "xavier")

        layers = []
        prev_size = input_size

        for _ in range(num_layers):
            layers.append(NeuralLayer(prev_size, hidden_size, activation, weight_init))
            prev_size = hidden_size

        # Output layer — always linear (logits); softmax is in the loss
        layers.append(NeuralLayer(prev_size, output_size, "identity", weight_init))
        return layers

    def forward(self, X):
        
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out   # logits

    def backward(self, y_true, y_pred):
        
        # Compute loss and its gradient w.r.t. logits
        _loss = self.loss_fn.forward(y_pred, y_true)
        delta = self.loss_fn.backward()

        weight_decay = getattr(self.args, "weight_decay", 0.0)
        
        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse
        for layer in reversed(self.layers):
            if layer.activation_name not in ("identity", "linear", "softmax"):
                delta = delta * layer.activation_fn.derivative(layer.z)

            # layer.backward returns dL/da for the previous layer
            delta = layer.backward(delta, weight_decay)

            grad_W_list.append(layer.grad_W.copy())
            grad_b_list.append(layer.grad_b.copy())

        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.update(self.layers)

    def train(self, X_train, y_train, X_val = None, y_val = None, epochs = 10, batch_size = 32,
              wandb_log = False, wandb_run = None, log_gradient_norms = False):
        
        n = X_train.shape[0]
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        is_nag = isinstance(self.optimizer, NAG)

        for epoch in range(1, epochs + 1):
            # Shuffle training data
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                X_b = X_shuf[start: start + batch_size]
                y_b = y_shuf[start: start + batch_size]

                # Forward
                logits = self.forward(X_b)
                loss   = self.loss_fn.forward(logits, y_b)

                # Backward
                self.backward(y_b, logits)

                # Update
                self.update_weights()

                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            train_acc, _ = self.evaluate(X_train, y_train)
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(train_acc)

            log_dict = {"epoch": epoch, "train_loss": avg_loss, "train_acc": train_acc}

            # Validation
            if X_val is not None and y_val is not None:
                val_acc, val_loss = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                log_dict.update({"val_loss": val_loss, "val_acc": val_acc})
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

            # Gradient norm logging
            if log_gradient_norms and self.grad_W is not None:
                for li, gw in enumerate(reversed(list(self.grad_W))):
                    # li=0 is first hidden layer, li=num_layers-1 is output layer
                    norm = float(np.linalg.norm(gw))
                    log_dict[f"grad_norm_layer_{li}"] = norm

            # W&B logging
            if wandb_log:
                try:
                    import wandb
                    wandb.log(log_dict)
                except Exception as e:
                    print(f"W&B log error: {e}")

        return history

    def evaluate(self, X, y):
        logits = self.forward(X)
        loss   = self.loss_fn.forward(logits, y)
        preds  = np.argmax(logits, axis=1)
        acc    = float(np.mean(preds == y.astype(int)))
        return acc, float(loss)

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

