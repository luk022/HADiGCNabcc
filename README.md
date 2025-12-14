# HADiGCN: Code for the Hierarchy-Aware Differential GCN (HADiGCN)

This repository provides the official implementation of the Hierarchy-Aware Differential Graph Convolutional Network (HADiGCN) developed in the thesis, including experiments for simulation, single-cell RNA-seq classification, and spatial transcriptomics annotation.

## Repository Structure

The core codebase is organized as a Python package, `hadigcn`, with dedicated scripts for running each major experiment. This ensures code reuse and clarity for your thesis submission.

| Directory / File | Description |
| :--- | :--- |
| **`hadigcn/`** | The main Python package containing all reusable components. |
| ├── `__init__.py` | **Public API:** Exports core classes, models, and utilities (`HADiGCN`, `train`, `set_seed`, etc.). |
| ├── `models.py` | Defines the **`HADiGCN`** model and MLP baseline models. |
| ├── `train.py` | Contains the **reusable training loop** logic (`train`, `train_one_epoch`, `validate_one_epoch`). |
| ├── `loss_functions.py`| Defines `TopKCrossEntropyLoss` and `CrossEntropyLossWrapper`. |
| └── `utils.py` | General utilities, including **data preprocessing helpers** (`high_var_npdata`, `sparse_mx_to_torch_sparse_tensor`, etc.). |
| **`examples/`** | Scripts for running the full set of thesis experiments. |
| ├── `example_pancreas_full.py` | **Pancreas (Baron $\rightarrow$ Segerstolpe)** cross-dataset experiment. |
| ├── `example_dlpfc_full.py` | **DLPFC (Layer Classification)** spatial transcriptomics experiment. |
| ├── `simulation_example.py` | **Synthetic/Simulation** experiment runner (CE vs. Top-K Loss). |
| └── `example_tuning_alpha.py`| Hyperparameter tuning experiment for the Top-K loss $\alpha$ parameter. |
| **`data/`** | **[Placeholder]** Directory for storing raw and processed input datasets. |
| **`results/`** | **[Placeholder]** Directory for saving model checkpoints, training logs, and final metrics. |

## Installation and Setup

### Prerequisites

You will need Python 3.8+ and a standard scientific computing environment.

### 1. Clone the Repository

```bash
git clone [YOUR_REPO_URL]
cd HADiGCN
2. Set up Environment (Recommended)
Bash

conda create -n hadigcn python=3.9
conda activate hadigcn
3. Install Dependencies
This project relies on PyTorch and PyTorch Geometric. Note: Adjust the torch_geometric installation based on your specific CUDA version (or omit cuda if using CPU-only).

Bash

# Core dependencies
pip install torch==1.13.1 torchaudio==0.13.1 torchvision==0.14.1

# Scientific and data dependencies
pip install numpy pandas scipy scikit-learn anndata matplotlib

# Top-k loss Dependency
pip install difftopk

# PyTorch Geometric (Example command for CUDA 11.7)
pip install torch_geometric 

# Final dependency: Install the local package in editable mode
pip install -e .

Running the Experiments
The primary experiments corresponding to the thesis chapters are run using the scripts in the examples/ directory.

1. Pancreas Experiment (Cross-Dataset)
Bash

python examples/example_pancreas_full.py
2. DLPFC Experiment (Spatial Classification)
Bash

python examples/example_dlpfc_full.py
3. Simulation Experiments
Bash

python examples/simulation_example.py
4. Top-k Alpha Tuning
Bash

python examples/example_tuning_alpha.py
Debug Mode
The core experiment scripts (pancreas, dlpfc, simulation) support a --debug flag for rapid testing, which limits the run to a single seed and a small number of epochs (e.g., 3).

Bash

# Example: Quick test of the Pancreas experiment
python examples/example_pancreas_full.py --debug

## Acknowledgement

The core logic for the top-k loss is from the **difftopk package dependency** that users should install themselves. The original repository can be found here: https://github.com/Felix-Petersen/difftopk. 