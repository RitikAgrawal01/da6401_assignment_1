# Assignment 1: Multi-Layer Perceptron for Image Classification

**Name:** Ritik Agrawal | **Roll No:** DA25M026

## Links
🔗 **W&B Report**: https://api.wandb.ai/links/agrawalritik2001-/q3wd7ey9  
🔗 **GitHub**: https://github.com/RitikAgrawal01/da6401_assignment_1

---

## Overview
A complete NumPy-only implementation of a configurable Multi-Layer Perceptron (MLP) trained on MNIST and Fashion-MNIST. No PyTorch, TensorFlow or JAX used — only NumPy for all forward/backward computations.

**Best Results:**
- MNIST: ~96.4% test accuracy
- Fashion-MNIST: ~87.9% test accuracy

---

## Project Structure
```
da6401_assignment_1/
├── notebooks/
│   └── wandb_experiments.ipynb   ← All W&B experiments (sections 2.1–2.10)
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py         ← ReLU, Sigmoid, Tanh, Softmax, Identity
│   │   ├── neural_layer.py        ← Single FC layer (forward + backward)
│   │   ├── neural_network.py      ← Main model class
│   │   ├── objective_functions.py ← CrossEntropy, MSE
│   │   └── optimizers.py          ← SGD, Momentum, NAG, RMSProp
│   ├── utils/
│   │   |── __init__.py
│   |   └── data_loader.py         ← MNIST / Fashion-MNIST loading
│   ├── train.py                   ← Training entry point (CLI)
│   ├── inference.py               ← Evaluation entry point (CLI)
│   ├── best_model.npy             ← Best model weights (MNIST)
│   └── best_config.json           ← Best hyperparameter config
├── requirements.txt
└── README.md
```

---

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/da6401_assignment_1.git
cd da6401_assignment_1
pip install -r requirements.txt
```

---

## Training

```bash
cd src

# Train with default best config
python train.py

# Full example with all arguments
python train.py \
  -d mnist \
  -e 20 \
  -b 32 \
  -lr 0.001 \
  -o rmsprop \
  -nhl 3 \
  -sz 128 \
  -a relu \
  -l cross_entropy \
  -w_i xavier \
  -wd 0.0005 \
  -w_p your_wandb_project
```

### CLI Arguments

| Flag | Long | Default | Description |
|------|------|---------|-------------|
| `-d` | `--dataset` | `mnist` | `mnist` or `fashion_mnist` |
| `-e` | `--epochs` | `20` | Number of training epochs |
| `-b` | `--batch_size` | `32` | Mini-batch size |
| `-lr` | `--learning_rate` | `0.001` | Learning rate |
| `-o` | `--optimizer` | `rmsprop` | `sgd`, `momentum`, `nag`, `rmsprop` |
| `-wd` | `--weight_decay` | `0.0005` | L2 regularization coefficient |
| `-nhl` | `--num_layers` | `3` | Number of hidden layers |
| `-sz` | `--hidden_size` | `128` | Neurons per hidden layer |
| `-a` | `--activation` | `relu` | `relu`, `sigmoid`, `tanh` |
| `-l` | `--loss` | `cross_entropy` | `cross_entropy`, `mse` |
| `-w_i` | `--weight_init` | `xavier` | `random`, `xavier` |
| `-w_p` | `--wandb_project` | `None` | W&B project name (omit to skip logging) |

---

## Inference / Evaluation

```bash
cd src

# Evaluate best saved model (reads config from best_config.json automatically)
python inference.py --model_path best_model.npy

# Specify dataset explicitly
python inference.py --model_path best_model.npy -d mnist
```

**Output:**
```
========== Evaluation Results ==========
  Loss      : 0.1523
  Accuracy  : 0.9642  (96.42%)
  F1-Score  : 0.9641
  Precision : 0.9644
  Recall    : 0.9642
=========================================
```

---

## Best Configuration

| Hyperparameter | Value |
|---------------|-------|
| Dataset | MNIST |
| Optimizer | momentum |
| Learning Rate | 0.01 |
| Hidden Layers | 5 |
| Neurons/Layer | 128 |
| Activation | ReLU |
| Weight Init | Xavier |
| Batch Size | 32 |
| Epochs | 20 |
| Weight Decay | 0.0005 |
| Loss | Cross Entropy |

---

## Key Implementation Details

### Model Output
The network returns **raw logits** (no softmax applied). Softmax is computed inside the loss function for numerical stability.

### Gradient Ordering
`grad_W[0]` = last layer gradients  
`grad_W[-1]` = first layer gradients  
(Reversed order as required by the grader)

### Weight Initialization
- **Xavier**: `W ~ N(0, sqrt(2 / (fan_in + fan_out)))` — default, recommended
- **Random**: `W ~ N(0, 0.01)`

### Optimizers Implemented
- **SGD**: `W ← W - lr * grad_W`
- **Momentum**: `v ← β*v + grad_W`, `W ← W - lr*v`
- **NAG**: Nesterov look-ahead approximation
- **RMSProp**: Adaptive learning rate using squared gradient moving average

### Best Model Saving
Model is saved automatically during training — only saves if the current run's test F1 score beats the previous best stored in `best_config.json`:

```python
# Automatically handled in train.py
if test_f1 > best_f1_so_far:
    save_model(model, args.model_save_path, args.config_save_path, args, test_f1)
```

### Model Loading
```python
import numpy as np
from ann.neural_network import NeuralNetwork

data  = np.load("best_model.npy", allow_pickle=True).item()
model = NeuralNetwork(args)
model.set_weights(data)
```

---

## W&B Experiments

The notebook `notebooks/wandb_experiments.ipynb` contains all experiments:

| Section | Topic |
|---------|-------|
| 2.1 | Data Exploration — 5 samples per class |
| 2.2 | Hyperparameter Sweep — 100+ runs, Bayesian optimization |
| 2.3 | Optimizer Comparison — SGD vs Momentum vs NAG vs RMSProp |
| 2.4 | Vanishing Gradients — Sigmoid vs ReLU (6 layers) |
| 2.5 | Dead Neurons — ReLU vs Tanh at high learning rate |
| 2.6 | Loss Functions — Cross Entropy vs MSE |
| 2.7 | Global Performance — Train vs Val accuracy overlay |
| 2.8 | Error Analysis — Confusion matrix + per-class accuracy |
| 2.9 | Weight Initialization — Zeros vs Xavier symmetry |
| 2.10 | Fashion-MNIST Transfer — 3 configs from MNIST |

---

## Requirements
```
numpy
keras
tensorflow
scikit-learn
matplotlib
seaborn
wandb
```

---

## Author
**Ritik Agrawal**  
DA25M026 — DA6401 Deep Learning, IIT Madras
