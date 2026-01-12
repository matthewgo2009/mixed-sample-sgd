 **Mixed-Sample SGD (PyTorch Implementation)**

This repository contains the official PyTorch implementation of Mixed-Sample SGD from the paper:
MIXED-SAMPLE SGD: AN END-TO-END ANALYSIS OF SUPERVISED TRANSFER LEARNING
Yuyang Deng, Samory Kpotufe
NeurIPS 2025

 Mixed-Sample SGD is an optimization algorithm designed for [briefly describe the goal, e.g., robust transfer learning, domain adaptation, etc.]. It dynamically reweights samples from a source domain ($S_Q$) and a target domain ($S_P$) using a dual-variable mechanism.

 **Installation**

You can install this package directly from GitHub:Bashpip install git+https://github.com/[YOUR_USERNAME]/mixed-sample-sgd.git
For development or local testing:Bashgit clone https://github.com/[YOUR_USERNAME]/mixed-sample-sgd.git
cd mixed-sample-sgd
pip install -e .
Quick StartUsage is designed to be compatible with standard PyTorch optimizers, with an additional step for updating the verifier parameters ($\lambda$ and $\theta_Q$).Pythonimport torch
import torch.nn as nn
from mixed_sample_sgd import MixedSampleSGD
