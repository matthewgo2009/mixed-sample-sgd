# Mixed-Sample SGD (PyTorch Implementation)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-Compatible-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of **Mixed-Sample SGD**, an optimization algorithm designed for efficient transfer learning and domain adaptation.

It dynamically reweights samples from a source domain ($S_Q$) and a target domain ($S_P$) using a dual-variable mechanism, ensuring robust convergence even with limited target data.

## Installation

You can install this package directly from GitHub:

```bash
pip install git+https://github.com/matthewgo2009/mixed-sample-sgd.git
```

For development or local testing:

```bash
git clone https://github.com/matthewgo2009/mixed-sample-sgd.git
cd mixed-sample-sgd
pip install -e .
```

## Quick Start

Usage is designed to be compatible with standard PyTorch optimizers, with an additional step for updating the verifier parameters ($\lambda$ and $\theta_Q$).

```python
import torch
import torch.nn as nn
from mixed_sample_sgd import MixedSampleSGD

# 1. Define Main Model (theta) and Anchor Model (theta_q)
model = nn.Linear(10, 1)
model_q = nn.Linear(10, 1) # Must have same structure as model

# 2. Initialize Optimizer
optimizer = MixedSampleSGD(
    model.parameters(), 
    model_q.parameters(),
    lr=1e-3,        # eta (Main learning rate)
    alpha=1e-3,     # alpha_t (Anchor learning rate)
    gamma=1e-4,     # lambda decay
    epsilon_q=0.1   # tolerance
)

# 3. Training Loop
# ... (Data loading logic) ...

# Step A: Standard Optimization Step (Updates Theta)
# The optimizer decides whether to sample from Target or Source based on lambda
use_target = optimizer.get_sampling_probability()
# ... (Load batch based on use_target) ...

# loss = ...
# loss.backward()
optimizer.step() 

# Step B: Update Verifier (Updates Lambda & Theta_Q)
# This step requires a batch from the Source Domain (Sq)
# ... (Calculate loss_theta and loss_q on Source data) ...
optimizer.update_verifier(loss_theta, loss_q)
```

## Experiments

We provide a complete example to reproduce the experiments on CIFAR-like data (using ResNet-18 features).

### Running the Demo

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the experiment script:
   ```bash
   python examples/cifar_experiment.py
   ```

**Note:** The script will automatically download the pre-processed feature dataset (`cat_dog_resnet18.npz`) from the [GitHub Releases](https://github.com/matthewgo2009/mixed-sample-sgd/releases) if it is not found locally.

The script will:
* Train models using **Mixed-Sample SGD** vs. Standard SGD (Source-Only & Target-Only).
* Save performance plots and metrics to the `results/` directory.

## Parameters

| Argument | Symbol | Description | Default |
| :--- | :---: | :--- | :---: |
| `params` | $\theta$ | Parameters of the main target model. | - |
| `params_q` | $\theta_Q$ | Parameters of the reference/anchor model. | - |
| `lr` | $\eta$ | Learning rate for $\theta$ and $\lambda$. | 1e-3 |
| `alpha` | $\alpha_t$ | Learning rate for $\theta_Q$. | 1e-3 |
| `gamma` | $\gamma$ | Decay factor for $\lambda$. | 1e-4 |
| `epsilon_q` | $\epsilon_Q$ | Error tolerance threshold. | 0.1 |
| `weight_decay` | - | L2 penalty for $\theta$. | 0 |

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{dengmixed,
  title={Mixed-Sample SGD: an End-to-end Analysis of Supervised Transfer Learning},
  author={Deng, Yuyang and Kpotufe, Samory},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems} 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
