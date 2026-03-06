"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    # ---- Dataset ----
    parser.add_argument("-d", "--dataset",
                        type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    # ---- Training (kept so CLI is identical to train.py) ----
    parser.add_argument("-e", "--epochs",
                        type=int, default=15)

    parser.add_argument("-b", "--batch_size",
                        type=int, default=32)

    parser.add_argument("-lr", "--learning_rate",
                        type=float, default=0.01)

    parser.add_argument("-o", "--optimizer",
                        type=str, default="momentum",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])

    parser.add_argument("-wd", "--weight_decay",
                        type=float, default=0.0005)

    parser.add_argument("-nhl", "--num_layers",
                        type=int, default=5)

    parser.add_argument("-sz", "--hidden_size",
                    type=int, default=128, nargs="+")

    parser.add_argument("-a", "--activation",
                        type=str, default="relu",
                        choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("-l", "--loss",
                        type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"])

    parser.add_argument("-w_i", "--weight_init",
                        type=str, default="xavier",
                        choices=["random", "xavier", "zeros"])

    parser.add_argument("-w_p", "--wandb_project",
                        type=str, default=None)

    # ---- Inference-specific ----
    parser.add_argument("--model_path",
                        type=str, default="best_model.npy",
                        help="Relative path to saved model weights (.npy)")

    parser.add_argument("--config_path",
                        type=str, default="best_config.json",
                        help="Optional path to config JSON; overrides CLI args if provided")

    parser.add_argument("--input_size",  type=int, default=784)
    parser.add_argument("--output_size", type=int, default=10)
    parser.add_argument("--val_split",   type=float, default=0.1)
    
    args, _ = parser.parse_known_args()
    if isinstance(args.hidden_size, list) and len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]
    return args


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score)
    from ann.objective_functions import get_loss

    # Forward pass — returns logits
    logits = model.forward(X_test)

    # Loss
    loss_fn = get_loss(model.args.loss if hasattr(model.args, "loss") else "cross_entropy")
    loss    = loss_fn.forward(logits, y_test)

    # Predicted classes
    preds = np.argmax(logits, axis=1)
    y_int = y_test.astype(int)

    accuracy  = accuracy_score(y_int, preds)
    f1        = f1_score(y_int, preds, average="macro", zero_division=0)
    precision = precision_score(y_int, preds, average="macro", zero_division=0)
    recall    = recall_score(y_int, preds, average="macro", zero_division=0)

    return {
        "logits":    logits,
        "loss":      float(loss),
        "accuracy":  float(accuracy),
        "f1":        float(f1),
        "precision": float(precision),
        "recall":    float(recall),
    }


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    
    # --- Optionally override args from saved config ---
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(args, k, v)   # always override with saved config
        print(f"Loaded config from {args.config_path}")

    # --- Load data (test set only) ---
    _, _, (X_test, y_test), class_names = load_data(
        args.dataset, val_split=args.val_split
    )

    # --- Rebuild model with correct architecture ---
    model = NeuralNetwork(args)

    # --- Load weights ---
    weights = load_model(args.model_path)
    model.set_weights(weights)
    print(f"Loaded model weights from {args.model_path}")

    # --- Evaluate ---
    results = evaluate_model(model, X_test, y_test)

    print("\n========== Evaluation Results ==========")
    print(f"  Loss      : {results['loss']:.4f}")
    print(f"  Accuracy  : {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print("=========================================\n")

    print("Evaluation complete!")
    return results


if __name__ == '__main__':
    main()
