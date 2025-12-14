#!/usr/bin/env python3
"""
example_dlpfc_full.py

Self-contained reproduction of your DLPFC experiments.
Assumes you will provide preprocessed files:
  - HUMAN_DLPFC_<SLICE>_processed.h5ad  (AnnData with .X and .obs['layer_label_int'] or similar)
  - HUMAN_DLPFC_<SLICE>_adj_csr.npz     (gene-gene adjacency, CSR)
  - optionally: gene names inside the .h5ad.var_names or a separate .npy referenced below

Usage:
    python examples/dlpfc/example_dlpfc_full.py            # full run
    python examples/dlpfc/example_dlpfc_full.py --debug    # quick debug (3 epochs, single sorter/seed)

Notes:
- The create_dataset_for_gcn function expects labels stored in .obs with key 'layer_label_int' by default.
  If your preprocessed h5ad uses a different obs key, update LABEL_OBS_KEY below.
- This script mirrors the pancreas example style.
"""

import os
import sys 
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import anndata as ad
import scipy.sparse as sp

import torch
#import torch.nn as nn
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import json 
import h5py
import torch_geometric.utils

#--- add path where hadigcn pacakge is located---
script_dir = os.path.dirname(os.path.abspath(__file__))
package_parent_dir = os.path.dirname(os.path.dirname(script_dir))

# Add the parent directory to Python's search path if it's not already there
if package_parent_dir not in sys.path:
    sys.path.append(package_parent_dir)
# --- END PATH FIX ---

# Package imports (your package; ensure package folder is named `hadigcn_thesis`)
from hadigcn.loss_functions import TopKCrossEntropyLoss, CrossEntropyLossWrapper
from hadigcn.models import HADiGCN#, MLP1Layer, MLP2Layer  # adjust names if needed
from hadigcn.utils import set_seed, get_device
from hadigcn.train import train_one_epoch

# -------------------------
# User adjustable defaults
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DLPFC_DATA_DIR = os.path.join(THIS_DIR, "dataset")   # examples/dlpfc/dataset/
RESULTS_ROOT = os.path.join(THIS_DIR, "results")
os.makedirs(RESULTS_ROOT, exist_ok=True)

# Label obs key inside the .h5ad file (change if your preprocessed h5ad uses different column)
LABEL_OBS_KEY = "layer_label_int"

# Experiment defaults
num_features = 1                # each gene node feature dimension (expression scalar)
batch_size = 32
initial_lr = 1e-3
num_epochs = 300
NUM_GENES_TO_KEEP = 100
NUM_SPOTS_TO_USE = None         # None -> use all spots


TRAIN_SLICE_ID = "151673" 

# Sorter -> Pk mapping (your requested config)
SORTER_TO_PK = {
    "celoss": [None],
    "odd_even": [
        [0.3, 0.4, 0.3],
        [0.4, 0.0, 0.6],
        [0.7, 0.2, 0.1],
    ],
    "softsort": [
        [0.4, 0.0, 0.6],
        [0.8, 0.0, 0.2],
    ],
}
DEFAULT_SORTER_LIST = ["odd_even", "softsort"]  # default (you can include 'celoss' here to run CE baseline)

# Debug overrides: single sorter/seed/Pk and few epochs for quick check
DEBUG_CONFIG = {
    "num_epochs": 3,
    "global_seeds_list": [14,15],#32
    "default_sorter_list": ["celoss","softsort"],   # run only softsort in debug by default
    "softsort_pk_debug": [[0.4, 0.0, 0.6]]
}

# -------------------------
# Helper utilities
# -------------------------
def build_weighted_sampler(train_labels_for_weights, num_classes):
    class_counts = Counter(train_labels_for_weights)
    for i in range(num_classes):
        if i not in class_counts:
            class_counts[i] = 0
    class_weights_for_sampler = torch.tensor(
        [1.0 / (class_counts[i] + 1e-5) for i in range(num_classes)], dtype=torch.float32
    )
    sample_weights = torch.tensor([class_weights_for_sampler[int(lbl)] for lbl in train_labels_for_weights], dtype=torch.float32)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler

def pk_to_string(Pk):
    if Pk is None:
        return "celoss"
    return "".join([c for num in Pk for c in str(num).replace(".", "")])

# -------------------------
# Dataset loader for preprocessed DLPFC .h5ad + adjacency
# -------------------------

# --- Function to load preprocessed data ---
def load_preprocessed_dlpfc_data(h5ad_path, metadata_json_path, npz_adj_path, base_dlpfc_dir, train_slice_id, device):
    """
    Loads preprocessed AnnData objects and global metadata, then prepares PyTorch Geometric Data.
    
    Args:
        h5ad_path (str): Path to the preprocessed AnnData (.h5ad) file.
        metadata_json_path (str): Path to the global metadata JSON file.
        npz_adj_path (str): Path to the global gene-gene adjacency matrix NPZ file.
        base_dlpfc_dir (str): Base directory for raw DLPFC data (to get full gene list from H5).
        train_slice_id (str): ID of the training slice (used to find the original H5 for NPZ gene list).
        device (torch.device): Device to load data onto.
        
    Returns:
        tuple: (dataset_pyg, num_unique_classes, mapped_labels_np, gene_names_in_adata)
    """
    print(f"\n--- Loading preprocessed data from {h5ad_path} ---")
    
    try:
        adata = ad.read_h5ad(h5ad_path)
        print(f"  Loaded AnnData object. Shape: {adata.shape} (spots x genes)")
    except Exception as e:
        print(f"Error loading AnnData from {h5ad_path}: {e}")
        raise

    try:
        with open(metadata_json_path, 'r') as f:
            global_metadata = json.load(f)
        print(f"  Loaded global metadata from {metadata_json_path}.")
    except Exception as e:
        print(f"Error loading global metadata from {metadata_json_path}: {e}")
        raise

    features_np = adata.X.toarray().astype(np.float32)
    mapped_labels_np = adata.obs['layer_label_int'].values
    gene_names_in_adata = adata.var_names.tolist()
    
    num_unique_classes = global_metadata['num_unique_classes']
    train_selected_gene_names = global_metadata['train_selected_gene_names']
    
    print("Loading full gene-gene adjacency matrix (NPZ)...")
    adjall_sparse = sp.load_npz(npz_adj_path)
    adjall_sparse = adjall_sparse.tocsr()
    print(f"  Full adjacency matrix shape: {adjall_sparse.shape}")

    full_gene_list_for_adj = []
    try:
        original_h5_path_for_adj_genes = os.path.join(DLPFC_DATA_DIR, f'{TRAIN_SLICE_ID}_filtered_feature_bc_matrix.h5')
        with h5py.File(original_h5_path_for_adj_genes, 'r') as f:
            features_group = f['matrix']['features']
            full_gene_list_for_adj = [name.decode('utf-8') for name in features_group['name'][:]]
    except Exception as e:
        print(f"Warning: Could not load full gene list for adjacency matrix from H5: {e}. Assuming adjacency matrix gene order matches selected genes directly.")
        full_gene_list_for_adj = train_selected_gene_names
        
    gene_indices_in_full_adj = [full_gene_list_for_adj.index(gene) for gene in train_selected_gene_names if gene in full_gene_list_for_adj]
    
    if len(gene_indices_in_full_adj) != len(train_selected_gene_names):
        print("Warning: Mismatch between selected genes and genes found in full adjacency matrix. Proceeding with the intersection.")
    
    final_adj_sparse = adjall_sparse[gene_indices_in_full_adj, :][:, gene_indices_in_full_adj]
    
    print(f"  Final adjacency matrix shape (after subsetting to selected genes): {final_adj_sparse.shape}")

    edge_index_pyg, edge_attr_pyg = torch_geometric.utils.from_scipy_sparse_matrix(final_adj_sparse)
    edge_index_pyg = edge_index_pyg.to(device)
    if edge_attr_pyg is not None:
        edge_attr_pyg = edge_attr_pyg.to(device)

    print(f"  Converted adjacency matrix to PyG edge_index. Shape: {edge_index_pyg.shape}")

    dataset_pyg = []
    print(f"Creating {features_np.shape[0]} PyTorch Geometric Data objects...")
    for spot_idx in range(features_np.shape[0]):
        x_spot_features = torch.tensor(features_np[spot_idx, :], dtype=torch.float32).unsqueeze(1).to(device)
        y_spot_label = torch.tensor([mapped_labels_np[spot_idx]], dtype=torch.long).to(device)
        dataset_pyg.append(Data(x=x_spot_features, edge_index=edge_index_pyg, edge_attr=edge_attr_pyg, y=y_spot_label))
    print(f"Successfully generated {len(dataset_pyg)} PyTorch Geometric Data objects.")
        
    return dataset_pyg, num_unique_classes, mapped_labels_np, gene_names_in_adata
# Evaluation (top1, top3, micro/macro f1)
# -------------------------
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    top1_correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr) if hasattr(model, "conv1") else model(batch.x)
            # out: (batch_size, num_classes)
            #probs = torch.softmax(out, dim=1)
            preds_top1 = out.argmax(dim=1)
            _, topk = out.topk(3, dim=1)

            labels = batch.y.squeeze()
            top1_correct += (preds_top1 == labels).sum().item()
            top3_correct += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()

            all_preds.extend(preds_top1.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)

    top1 = top1_correct / total if total > 0 else 0.0
    top3 = top3_correct / total if total > 0 else 0.0
    micro_f1 = float(f1_score(all_labels, all_preds, average="micro", zero_division=0))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return top1, top3, micro_f1, macro_f1



GLOBAL_METADATA_PATH = os.path.join(DLPFC_DATA_DIR, "global_preprocessing_metadata.json")
# Main experiment runner
# -------------------------
# -------------------------
def main_experiment_runner(args):
    # runtime settings
    debug = args.debug

    if debug:
        local_num_epochs = DEBUG_CONFIG["num_epochs"]
        global_seeds_list = DEBUG_CONFIG["global_seeds_list"]
        sorter_list = DEBUG_CONFIG["default_sorter_list"]
        # override Pk for debug softsort if present
        SORTER_TO_PK["softsort"] = DEBUG_CONFIG["softsort_pk_debug"]
    else:
        local_num_epochs = num_epochs
        global_seeds_list = [32, 42, 52]
        sorter_list = DEFAULT_SORTER_LIST

    # dataset paths (update filenames if you named differently)
    DATASET_CONFIG = {
        "train": {
            "h5ad": os.path.join(DLPFC_DATA_DIR, "HUMAN_DLPFC_151673_processed.h5ad"),
            "adj": os.path.join(DLPFC_DATA_DIR, "HUMAN_DLPFC_151673_adj_csr.npz"),
        },
        "test_scenario_1": {
            "h5ad": os.path.join(DLPFC_DATA_DIR, "HUMAN_DLPFC_151676_processed.h5ad"),
            "adj": os.path.join(DLPFC_DATA_DIR, "HUMAN_DLPFC_151673_adj_csr.npz"),  # same adj base
        },
        "test_scenario_2": {
            "h5ad": os.path.join(DLPFC_DATA_DIR, "HUMAN_DLPFC_151507_processed.h5ad"),
            "adj": os.path.join(DLPFC_DATA_DIR, "HUMAN_DLPFC_151673_adj_csr.npz"),
        },
    }

    device = get_device()

    
    # New function call with 6 arguments and 4 return values (mapping to train_dataset, num_classes, train_labels_for_weights, train_gene_names)
    train_dataset, num_classes, train_labels_for_weights, train_gene_names = load_preprocessed_dlpfc_data(
        h5ad_path=DATASET_CONFIG["train"]["h5ad"],
        metadata_json_path=GLOBAL_METADATA_PATH,  # New argument
        npz_adj_path=DATASET_CONFIG["train"]["adj"],
        base_dlpfc_dir=DLPFC_DATA_DIR,            # New argument
        train_slice_id=TRAIN_SLICE_ID,            # New argument
        device=device
    )

    # load both test scenarios
    print("Preparing test dataset (scenario 1: slice 151676)...")
    # New function call with 6 arguments and 4 return values (using _ to ignore the last 3)
    test1_dataset, _, _, _ = load_preprocessed_dlpfc_data(
        h5ad_path=DATASET_CONFIG["test_scenario_1"]["h5ad"],
        metadata_json_path=GLOBAL_METADATA_PATH, # New argument
        npz_adj_path=DATASET_CONFIG["test_scenario_1"]["adj"],
        base_dlpfc_dir=DLPFC_DATA_DIR,           # New argument
        train_slice_id=TRAIN_SLICE_ID,           # MUST use the train slice ID for adjacency consistency
        device=device
    )

    print("Preparing test dataset (scenario 2: slice 151507)...")
    # New function call with 6 arguments and 4 return values
    test2_dataset, _, _, _ = load_preprocessed_dlpfc_data(
        h5ad_path=DATASET_CONFIG["test_scenario_2"]["h5ad"],
        metadata_json_path=GLOBAL_METADATA_PATH, # New argument
        npz_adj_path=DATASET_CONFIG["test_scenario_2"]["adj"],
        base_dlpfc_dir=DLPFC_DATA_DIR,           # New argument
        train_slice_id=TRAIN_SLICE_ID,           # MUST use the train slice ID for adjacency consistency
        device=device
    )
    

    num_nodes = len(train_gene_names)

    # iterate sorters and their Pk lists
    for sorter in sorter_list:
        pk_list = SORTER_TO_PK.get(sorter, [])
        if not pk_list:
            print(f"[WARNING] sorter '{sorter}' not in SORTER_TO_PK mapping - skipping.")
            continue

        output_dir_base = os.path.join(RESULTS_ROOT, f"{sorter}_dlfpc_runs")
        for Pk in pk_list:
            pkstr = pk_to_string(Pk)
            experiment_dir = os.path.join(output_dir_base, "multseed_train151673_test_scenarios", pkstr, f"{batch_size}bs_{local_num_epochs}epochs")
            os.makedirs(experiment_dir, exist_ok=True)

            seed_metrics_for_pk = []

            for globalseed in global_seeds_list:
                set_seed(globalseed)
                print(f"\n[START] sorter={sorter} Pk={Pk} seed={globalseed}")

                sampler = build_weighted_sampler(train_labels_for_weights, num_classes)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, generator=torch.Generator().manual_seed(globalseed))
                test1_loader = DataLoader(test1_dataset, batch_size=batch_size, shuffle=False)
                test2_loader = DataLoader(test2_dataset, batch_size=batch_size, shuffle=False)

                # model (use HADiGCN default architecture; adjust hidden sizes as needed)
                model = HADiGCN(
                    in_channels=num_features,
                    hidden_channels=16,
                    final_channels=32,
                    num_nodes=num_nodes,
                    num_classes=num_classes
                ).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

                # choose criterion
                if sorter == "celoss":
                    criterion = CrossEntropyLossWrapper().to(device)
                else:
                    criterion = TopKCrossEntropyLoss(
                        diffsort_method=sorter,
                        inverse_temperature=2.0,
                        p_k=Pk,
                        n=num_classes,
                        m=(len(Pk) + 1) if Pk is not None else 1,
                        distribution="cauchy",
                        device=device,
                    ).to(device)

                # containers for logging
                metrics = {"train_loss": [], "test1_top1": [], "test1_top3": [], "test1_micro_f1": [], "test1_macro_f1": [],
                           "test2_top1": [], "test2_top3": [], "test2_micro_f1": [], "test2_macro_f1": []}
                log_lines = ["Epoch\tTrainLoss\tTest1Top1\tTest1Top3\tTest1MicroF1\tTest1MacroF1\tTest2Top1\tTest2Top3\tTest2MicroF1\tTest2MacroF1\tLR"]

                # training loop
                for epoch in range(local_num_epochs):
                    train_loss = train_one_epoch(
                        model=model,
                        dataloader=train_loader,
                        criterion=criterion,
                        optimizer=optimizer,
                        device=device
                    )
                    

                    # evaluate
                    t1_top1, t1_top3, t1_micro, t1_macro = evaluate(model, test1_loader, device)
                    t2_top1, t2_top3, t2_micro, t2_macro = evaluate(model, test2_loader, device)

                    metrics["train_loss"].append(train_loss)
                    metrics["test1_top1"].append(t1_top1)
                    metrics["test1_top3"].append(t1_top3)
                    metrics["test1_micro_f1"].append(t1_micro)
                    metrics["test1_macro_f1"].append(t1_macro)
                    metrics["test2_top1"].append(t2_top1)
                    metrics["test2_top3"].append(t2_top3)
                    metrics["test2_micro_f1"].append(t2_micro)
                    metrics["test2_macro_f1"].append(t2_macro)

                    current_lr = optimizer.param_groups[0]["lr"]
                    log_lines.append(f"{epoch+1}\t{train_loss:.4f}\t{t1_top1:.4f}\t{t1_top3:.4f}\t{t1_micro:.4f}\t{t1_macro:.4f}\t{t2_top1:.4f}\t{t2_top3:.4f}\t{t2_micro:.4f}\t{t2_macro:.4f}\t{current_lr:.6f}")

                    if (epoch + 1) % 10 == 0 or epoch in {0, local_num_epochs - 1}:
                        print(f"Epoch {epoch+1}/{local_num_epochs} | train_loss={train_loss:.4f} | test1_top1={t1_top1:.4f} test1_top3={t1_top3:.4f} | test2_top1={t2_top1:.4f}")

                # save per-seed logs
                timestamp = datetime.now().strftime("%m%d_%H%M")
                seed_dir = os.path.join(experiment_dir, f"seed_{globalseed}")
                os.makedirs(seed_dir, exist_ok=True)
                log_path = os.path.join(seed_dir, f"training_log_dlpfc_{pkstr}_{globalseed}_{timestamp}.txt")
                with open(log_path, "w") as f:
                    f.write("\n".join(log_lines))
                print(f"Saved training log to: {log_path}")

                # append seed-level summary
                seed_metrics_for_pk.append({
                    "seed": globalseed,
                    "test1_top1": metrics["test1_top1"][-1],
                    "test1_top3": metrics["test1_top3"][-1],
                    "test1_micro_f1": metrics["test1_micro_f1"][-1],
                    "test1_macro_f1": metrics["test1_macro_f1"][-1],
                    "test2_top1": metrics["test2_top1"][-1],
                    "test2_top3": metrics["test2_top3"][-1],
                    "test2_micro_f1": metrics["test2_micro_f1"][-1],
                    "test2_macro_f1": metrics["test2_macro_f1"][-1],
                })

                # optional plotting
                try:
                    if args.plot:
                        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        axes[0].plot(metrics["train_loss"], label="train_loss")
                        axes[0].set_title("Train Loss")
                        axes[1].plot(metrics["test1_top1"], label="test1_top1")
                        axes[1].plot(metrics["test1_top3"], label="test1_top3")
                        axes[1].legend()
                        plt.tight_layout()
                        plot_path = os.path.join(seed_dir, f"plots_{pkstr}_{globalseed}_{timestamp}.png")
                        plt.savefig(plot_path)
                        plt.close(fig)
                        print(f"Saved plots to: {plot_path}")
                except Exception as e:
                    print(f"[WARNING] plotting failed: {e}")
                    
            # after all seeds for this Pk -> write aggregate metrics
            metrics_dir = os.path.join(experiment_dir, "final_metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            final_metrics_path = os.path.join(metrics_dir, f"final_metrics_{pkstr}.txt")
            
            # --- APPEND LOGIC ---
            # 1. Check if the file already exists to decide on writing the header.
            file_exists = os.path.exists(final_metrics_path)

            # 2. Open the file in append mode ("a").
            # The context manager will automatically close the file afterwards.
            with open(final_metrics_path, "a") as f:
                
                # 3. Write the configuration header only if the file did not exist before this run.
                # This ensures the header is only at the top of the file.
                if not file_exists:
                    f.write(f"--- sorter: {sorter}  Pk: {pkstr} ---\n")
                    
                # 4. Write the results for the new seed(s) run in the current execution.
                for m in seed_metrics_for_pk:
                    f.write(f"Seed: {m['seed']}  Test1 Top1: {m['test1_top1']:.4f} Top3: {m['test1_top3']:.4f} MicroF1: {m['test1_micro_f1']:.4f} MacroF1: {m['test1_macro_f1']:.4f}\n")
                    f.write(f"        Test2 Top1: {m['test2_top1']:.4f} Top3: {m['test2_top3']:.4f} MicroF1: {m['test2_micro_f1']:.4f} MacroF1: {m['test2_macro_f1']:.4f}\n\n")

            print(f"Appended metrics for {len(seed_metrics_for_pk)} seed(s) to: {final_metrics_path}")
            
            '''
            # after all seeds for this Pk -> write aggregate metrics
            metrics_dir = os.path.join(experiment_dir, "final_metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            final_metrics_path = os.path.join(metrics_dir, f"final_metrics_{pkstr}.txt")
            with open(final_metrics_path, "w") as f:
                f.write(f"--- sorter: {sorter}  Pk: {pkstr} ---\n")
                for m in seed_metrics_for_pk:
                    f.write(f"Seed: {m['seed']}  Test1 Top1: {m['test1_top1']:.4f} Top3: {m['test1_top3']:.4f} MicroF1: {m['test1_micro_f1']:.4f} MacroF1: {m['test1_macro_f1']:.4f}\n")
                    f.write(f"        Test2 Top1: {m['test2_top1']:.4f} Top3: {m['test2_top3']:.4f} MicroF1: {m['test2_micro_f1']:.4f} MacroF1: {m['test2_macro_f1']:.4f}\n\n")
            print(f"Saved aggregated metrics to: {final_metrics_path}")
            
    print("\nAll DLPFC experiments completed.")
            '''
# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example DLPFC experiment runner")
    parser.add_argument("--debug", action="store_true", help="Run debug quick mode (3 epochs, single seed/sorter)")
    parser.add_argument("--plot", action="store_true", help="Enable plotting for each run")
    args = parser.parse_args()
    main_experiment_runner(args)
    #for debugging run this in console:
    #%run "D:/Github repos/local cloned/HADiGCN_test-new/HADiGCN_test new/examples/dlpfc/example_dlpfc_full_train_import.py" --debug
