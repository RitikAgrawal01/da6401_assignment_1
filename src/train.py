"""
Main Training Script
Entry point for training neural networks with command-line arguments
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
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    # ---- Dataset ----
    parser.add_argument("-d", "--dataset",
                        type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to use (default: mnist)")

    # ---- Training ----
    parser.add_argument("-e", "--epochs",
                        type=int, default=10,
                        help="Number of training epochs (default: 10)")

    parser.add_argument("-b", "--batch_size",
                        type=int, default=32,
                        help="Mini-batch size (default: 32)")

    parser.add_argument("-lr", "--learning_rate",
                        type=float, default=0.01,
                        help="Learning rate (default: 0.001)")

    # ---- Optimizer ----
    parser.add_argument("-o", "--optimizer",
                        type=str, default="adam",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help="Optimizer (default: rmsprop)")

    parser.add_argument("-wd", "--weight_decay",
                        type=float, default=0.0005,
                        help="L2 weight decay / regularization (default: 0.0)")

    # ---- Architecture ----
    parser.add_argument("-nhl", "--num_layers",
                        type=int, default=5,
                        help="Number of hidden layers (default: 3)")

    parser.add_argument("-sz", "--hidden_size",
                    type=int, default=128, nargs="+",
                    help="Number of neurons per hidden layer (default: 128)")

    # ---- Activation ----
    parser.add_argument("-a", "--activation",
                        type=str, default="relu",
                        choices=["relu", "sigmoid", "tanh"],
                        help="Hidden layer activation (default: relu)")

    # ---- Loss ----
    parser.add_argument("-l", "--loss",
                        type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"],
                        help="Loss function (default: cross_entropy)")

    # ---- Weight init ----
    parser.add_argument("-w_i", "--weight_init",
                        type=str, default="xavier",
                        choices=["random", "xavier", "zeros"],
                        help="Weight initialization (default: xavier)")

    # ---- W&B ----
    parser.add_argument("-w_p", "--wandb_project",
                        type=str, default=None,
                        help="Weights & Biases project name (omit to skip W&B logging)")

    parser.add_argument("--wandb_entity",
                        type=str, default=None,
                        help="W&B entity/username (optional)")

    # ---- Model saving ----
    parser.add_argument("--model_save_path",
                        type=str, default="best_model.npy",
                        help="Relative path to save best model weights (default: best_model.npy)")

    parser.add_argument("--config_save_path",
                        type=str, default="best_config.json",
                        help="Relative path to save best config JSON (default: best_config.json)")

    # ---- Data ----
    parser.add_argument("--val_split",
                        type=float, default=0.001,
                        help="Fraction of training data for validation (default: 0.1)")

    # ---- Fixed architecture I/O (usually don't change) ----
    parser.add_argument("--input_size",  type=int, default=784)
    parser.add_argument("--output_size", type=int, default=10)
    
    args, _ = parser.parse_known_args()
    if isinstance(args.hidden_size, list) and len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]
    return args


def save_model(model: NeuralNetwork, save_path: str, config_path: str, args, best_f1: float = 0.0):
    """Save weights as .npy and config as .json."""
    weights = model.get_weights()
    np.save(save_path, weights)
    print(f"  → Model weights saved to {save_path}")

    config = {
        "dataset":       args.dataset,
        "num_layers":    args.num_layers,
        "hidden_size":   args.hidden_size,
        "activation":    args.activation,
        "optimizer":     args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay":  args.weight_decay,
        "batch_size":    args.batch_size,
        "epochs":        args.epochs,
        "loss":          args.loss,
        "weight_init":   args.weight_init,
        "input_size":    args.input_size,
        "output_size":   args.output_size,
        "best_f1":       best_f1,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  → Config saved to {config_path}")



def main():
    
    args = parse_arguments()

    # ---- W&B init ----
    use_wandb = args.wandb_project is not None
    run = None
    if use_wandb:
        try:
            import wandb
            run = wandb.init(
                project = args.wandb_project,
                entity  = args.wandb_entity,
                config  = vars(args),
            )
            print(f"W&B run initialised: {run.name}")
        except Exception as e:
            print(f"W&B initialisation failed: {e}. Continuing without W&B.")
            use_wandb = False

    # ---- Load data ----
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = load_data(args.dataset, val_split=args.val_split)

    # ---- Build model ----
    model = NeuralNetwork(args)
    print(f"\nModel architecture:")
    print(f"  Input: {args.input_size}")
    for i in range(args.num_layers):
        print(f"  Hidden {i+1}: {args.hidden_size} neurons ({args.activation})")
    print(f"  Output: {args.output_size} (linear/logits)")
    print(f"  Optimizer: {args.optimizer}  LR: {args.learning_rate}  WD: {args.weight_decay}")
    print(f"  Loss: {args.loss}  Init: {args.weight_init}\n")

    # ---- Train ----
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        wandb_log=use_wandb,
        wandb_run=run,
    )

    # ---- Final test evaluation ----
    from sklearn.metrics import f1_score as sk_f1
    test_acc, test_loss = model.evaluate(X_test, y_test)
    test_logits = model.forward(X_test)
    test_preds  = np.argmax(test_logits, axis=1)
    test_f1     = sk_f1(y_test.astype(int), test_preds, average='macro', zero_division=0)
    print(f"\nFinal Test  | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

    if use_wandb:
        try:
            import wandb
            wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1})
        except Exception:
            pass

    # ---- Save model only if best F1 so far ----
    best_f1_so_far = 0.0
    if os.path.exists(args.config_save_path):
        with open(args.config_save_path, "r") as f:
            prev_config = json.load(f)
            best_f1_so_far = prev_config.get("best_f1", 0.0)

    if test_f1 > best_f1_so_far:
        save_model(model, args.model_save_path, args.config_save_path, args, test_f1)
        print(f"  → New best F1: {test_f1:.4f} (previous: {best_f1_so_far:.4f})")
    else:
        print(f"  → F1 {test_f1:.4f} did not beat best {best_f1_so_far:.4f} — model not saved.")


    if use_wandb and run is not None:
        try:
            run.finish()
        except Exception:
            pass

    print("\nTraining complete!")
    return history


if __name__ == '__main__':
    main()
