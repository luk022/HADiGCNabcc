#!/usr/bin/env python3
"""
example_pancreas_full.py

Self-contained reproduction validation on Segerstolpe dataset trained on Baron dataset.
Contains dataset loading (create_dataset_for_gcn) and evaluation (evaluate)
copied from your uploaded code (kept unchanged except for relative paths).
"""

import os
import sys

from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt



#--- add path where hadigcn pacakge is located---
script_dir = os.path.dirname(os.path.abspath(__file__))
package_parent_dir = os.path.dirname(os.path.dirname(script_dir))

# Add the parent directory to Python's search path if it's not already there
if package_parent_dir not in sys.path:
    sys.path.append(package_parent_dir)
# --- END PATH FIX ---

# Package imports (ensure package folder is named `hadigcn_thesis`)
from hadigcn.loss_functions import TopKCrossEntropyLoss, CrossEntropyLossWrapper
from hadigcn import HADiGCN# <-- Make sure 'train' is imported
##from hadigcn.utils import set_seed, get_device
from hadigcn.train import train_one_epoch
from hadigcn.utils import (
    set_seed, 
    get_device,
    findDuplicated, 
    high_var_npdata, 
    #sparse_mx_to_torch_sparse_tensor, 
    process_features_and_adj
)
# -----------------------------------------------------------------------------
# Directory and data layout 
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PANCREAS_DATA_DIR = os.path.join(THIS_DIR, "dataset")   # data stored in examples/pancreas/dataset
RESULTS_ROOT = os.path.join(THIS_DIR, "results")        # outputs -> examples/pancreas/results
os.makedirs(RESULTS_ROOT, exist_ok=True)

# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------
ENABLE_PLOTTING = False   # set True to generate plots
num_gene_list = [100]#0]
global_seeds_list = [42]#, 42, 52] # ← set to [32] for debug run
NUM_CELLS_TO_USE = None

num_features = 1
batch_size = 32
initial_lr = 0.001
num_epochs = 3#300  #← set this to 3 for debug run

device = get_device()

# Sorter -> Pk mapping (exact values you wanted)
SORTER_TO_PK = {    #  one sorter with one Pk for debug run
    "celoss": [None],
    "odd_even": [
        [0.1, 0.5, 0.0, 0.4],
       # [0.4, 0.4, 0.1, 0.1],
      #  [0.5, 0.0, 0.0, 0.5],
    ]#,
    #"softsort": [
    #    [0.4, 0.4, 0.1, 0.1],
   #     [0.8, 0.0, 0.0, 0.2],
   # ],
}
DEFAULT_SORTER_LIST = ["celoss", "odd_even", "softsort"]


# -----------------------------------------------------------------------------
# create_dataset_for_gcn 
# -----------------------------------------------------------------------------
def create_dataset_for_gcn(config, num_genes_to_keep, device, num_cells_to_use, gene_names_to_use_for_consistency=None):
    """
    Loads and preprocesses a single real dataset, creating PyG Data objects for GNN training.
    Returns: data_list, final_gene_names, num_classes, labels_for_weights, map_class_to_lineage_func
    """
    print(f"[INFO] Loading data from: {config['csv_path']} and {config['labels_path']}")
    csv_path = config['csv_path']
    labels_path = config['labels_path']
    npz_path = config['adj_path']
    gene_names_path = config['gene_names_path']

    # 1. Load initial features (cells x genes)
    original_features_df = pd.read_csv(csv_path, sep=',', index_col=0, header=0)
    print(f"  Input features shape: {original_features_df.shape}")

    # 2. Load cell labels from CSV
    print("2. Loading cell labels (CSV)...")
    labels_df_temp = pd.read_csv(labels_path, sep=',', index_col=0, header=0)

    # Align dataframes by index (cell names).
    common_cells = original_features_df.index.intersection(labels_df_temp.index)
    original_features_df = original_features_df.loc[common_cells]
    labels_df_temp = labels_df_temp.loc[common_cells]

    cell_type_le = LabelEncoder()
    lineage_le = LabelEncoder()

    cell_type_labels_encoded = cell_type_le.fit_transform(labels_df_temp.iloc[:, 0].values)
    lineage_labels_encoded = lineage_le.fit_transform(labels_df_temp.iloc[:, 1].values)

    num_classes = len(cell_type_le.classes_)

    print(f"  Loaded labels shape: {cell_type_labels_encoded.shape}, Number of classes: {num_classes}")
    print(f"  Class Distribution: {dict(Counter(cell_type_le.inverse_transform(cell_type_labels_encoded)))}")

    if original_features_df.shape[0] != cell_type_labels_encoded.shape[0]:
        raise ValueError(f"Number of cells in features ({original_features_df.shape[0]}) "
                         f"does not match number of cells in labels ({cell_type_labels_encoded.shape[0]}).")

    if num_cells_to_use is not None and num_cells_to_use < original_features_df.shape[0]:
        print(f"  Subsetting to {num_cells_to_use} cells for faster processing...")
        np.random.seed(42)
        cell_indices = np.random.choice(original_features_df.shape[0], num_cells_to_use, replace=False)
        original_features_df = original_features_df.iloc[cell_indices].copy()
        cell_type_labels_encoded = cell_type_labels_encoded[cell_indices].copy()
        lineage_labels_encoded = lineage_labels_encoded[cell_indices].copy()
        print(f"  Features shape after cell subsetting: {original_features_df.shape}")
        print(f"  Labels shape after cell subsetting: {cell_type_labels_encoded.shape}")
        print(f"  Class Distribution of Subsampled Data (numerical): {dict(Counter(cell_type_labels_encoded))}")

    # 3. Duplicate gene removal
    print("Removing duplicate gene names from features...")
    features_no_duplicates_df = findDuplicated(original_features_df)
    print(f"  Features shape after duplicate removal: {features_no_duplicates_df.shape}")

    # 4. Load full adjacency matrix from NPZ
    print("Loading full gene-gene adjacency matrix from NPZ using sp.load_npz...")
    adj_data_npz = sp.load_npz(npz_path)
    full_adj_sparse = adj_data_npz.tocsr()
    adj_gene_names = np.load(gene_names_path, allow_pickle=True)

    print(f"  Number of genes from adjacency gene list (.npy): {len(adj_gene_names)}")
    print(f"  Full adjacency matrix shape: {full_adj_sparse.shape}")
    print(f"  Full adjacency matrix NNZ: {full_adj_sparse.nnz}")

    # 5. Select high-variance genes or use pre-selected names
    if gene_names_to_use_for_consistency is None:
        print(f"Selecting {num_genes_to_keep} highly variable genes from this dataset...")
        _, gene_names_selected = high_var_npdata(
            features_no_duplicates_df.values, num_genes_to_keep, features_no_duplicates_df.columns, return_indices=True
        )
        final_gene_names = gene_names_selected
    else:
        print(f"Using pre-selected gene names from training data for consistency.")
        final_gene_names = gene_names_to_use_for_consistency

    # 6. Ensure genes are in the adjacency matrix and feature matrix
    final_gene_indices_adj = np.where(np.isin(adj_gene_names, final_gene_names))[0]
    final_gene_indices_features = np.where(np.isin(features_no_duplicates_df.columns, final_gene_names))[0]

    # 7. Get max value for scaling from features with selected genes
    max_val_for_scaling = np.log1p(features_no_duplicates_df.iloc[:, final_gene_indices_features].values).max()

    # 8. Process features and adjacency matrix
    processed_features, processed_adj = process_features_and_adj(
        features_no_duplicates_df,
        full_adj_sparse,
        final_gene_indices_adj,
        max_val_for_scaling
    )

    print(f"  Final processed adjacency matrix shape: {processed_adj.shape}")

    # 9. Create CLASS_TO_LINEAGE_MAP (mapping primary classes to lineage)
    CLASS_TO_LINEAGE_MAP = {
        cell_type_le.transform([cell_type])[0]: lineage_le.transform([lineage])[0]
        for cell_type, lineage in zip(labels_df_temp.iloc[:, 0].values, labels_df_temp.iloc[:, 1].values)
    }

    def map_class_to_lineage_func(class_labels):
        """Maps an array of class labels to lineage labels."""
        return np.array([CLASS_TO_LINEAGE_MAP.get(label, -1) for label in class_labels])

    # 10. Create PyG Data objects
    coo = processed_adj.tocoo()
    edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)
    edge_attr = torch.tensor(coo.data, dtype=torch.float)

    data_list = [
        Data(
            x=torch.tensor(processed_features[i, :], dtype=torch.float).view(-1, 1),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(cell_type_labels_encoded[i], dtype=torch.long).unsqueeze(0),
            y_lineage=torch.tensor(lineage_labels_encoded[i], dtype=torch.long).unsqueeze(0)
        ) for i in range(processed_features.shape[0])
    ]

    labels_for_weights = [data.y.item() for data in data_list]

    return data_list, final_gene_names, num_classes, labels_for_weights, map_class_to_lineage_func


# -----------------------------------------------------------------------------
# evaluate (copied from uploaded code)
# -----------------------------------------------------------------------------
def evaluate(model, data_loader, map_class_to_lineage_func):
    """
    Evaluates the model and returns:
      top1_acc, top2_acc, top4_acc, micro_f1, macro_f1, lineage_acc
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_lineage_preds = []
    all_lineage_labels = []

    top1_correct = 0
    top2_correct = 0
    top4_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)

            preds = out.argmax(dim=1)
            top1_correct += (preds == batch.y.squeeze()).sum().item()

            _, top2_preds = out.topk(k=2, dim=1)
            _, top4_preds = out.topk(k=4, dim=1)

            top4_correct += (batch.y.squeeze().view(-1, 1) == top4_preds).any(dim=1).sum().item()
            top2_correct += (batch.y.squeeze().view(-1, 1) == top2_preds).any(dim=1).sum().item()

            total_samples += batch.y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.squeeze().cpu().numpy())

            lineage_preds = map_class_to_lineage_func(preds.cpu().numpy())
            lineage_labels = batch.y_lineage.squeeze().cpu().numpy()
            all_lineage_preds.extend(lineage_preds)
            all_lineage_labels.extend(lineage_labels)

    top1_acc = top1_correct / total_samples if total_samples > 0 else 0.0
    top2_acc = top2_correct / total_samples if total_samples > 0 else 0.0
    top4_acc = top4_correct / total_samples if total_samples > 0 else 0.0

    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    lineage_acc = accuracy_score(all_lineage_labels, all_lineage_preds)

    return top1_acc, top2_acc, top4_acc, micro_f1, macro_f1, lineage_acc


# -----------------------------------------------------------------------------
# Main experiment runner (self-contained, uses the functions above)
# -----------------------------------------------------------------------------
def main_experiment_runner(sorter_list=None):
    # dataset config using relative dataset/ folder
    DATASET_CONFIG = {
        "baron": {
            "csv_path": os.path.join(PANCREAS_DATA_DIR, "baron_shared_expression_cleaned.csv"),
            "labels_path": os.path.join(PANCREAS_DATA_DIR, "Baron_processed_labels.csv"),
            "adj_path": os.path.join(PANCREAS_DATA_DIR, "processed_Baron_adjacency_scipy_format.npz"),
            "gene_names_path": os.path.join(PANCREAS_DATA_DIR, "processed_Baron_adjacency_scipy_format_gene_names.npy"),
        },
        "segerstolpe": {
            "csv_path": os.path.join(PANCREAS_DATA_DIR, "segerstolpe_shared_expression_cleaned.csv"),
            "labels_path": os.path.join(PANCREAS_DATA_DIR, "Segerstolpe_processed_labels.csv"),
            "adj_path": os.path.join(PANCREAS_DATA_DIR, "processed_Baron_adjacency_scipy_format.npz"),
            "gene_names_path": os.path.join(PANCREAS_DATA_DIR, "processed_Baron_adjacency_scipy_format_gene_names.npy"),
        },
    }

    sorter_list = sorter_list if sorter_list is not None else DEFAULT_SORTER_LIST

    for num_gene in num_gene_list:
        print("Preparing training dataset (Baron)...")
        train_dataset, train_gene_names, num_classes, train_labels_for_weights, map_class_to_lineage_func = create_dataset_for_gcn(
            config=DATASET_CONFIG["baron"],
            num_genes_to_keep=num_gene,
            device=device,
            num_cells_to_use=NUM_CELLS_TO_USE
        )

        print("Preparing testing dataset (Segerstolpe)...")
        test_dataset, _, _, _, _ = create_dataset_for_gcn(
            config=DATASET_CONFIG["segerstolpe"],
            num_genes_to_keep=num_gene,
            device=device,
            num_cells_to_use=NUM_CELLS_TO_USE,
            gene_names_to_use_for_consistency=train_gene_names
        )

        for sorter in sorter_list:
            pk_list = SORTER_TO_PK.get(sorter, [])
            if not pk_list:
                print(f"[WARNING] Sorter '{sorter}' has no Pk list. Skipping.")
                continue

            torch.cuda.empty_cache()
            num_nodes = num_gene
            output_dir_base = os.path.join(RESULTS_ROOT, f"{sorter} gcn topk")

            for Pk in pk_list:
                pkstr = "".join([c for num in Pk for c in str(num).replace(".", "")]) if Pk is not None else "celoss"
                experiment_dir = os.path.join(
                    output_dir_base,
                    "multseed baron_train_seg_test",
                    pkstr,
                    f"{batch_size} batch_size {num_epochs} epochs"
                )

                seed_metrics_for_pk = []

                for globalseed in global_seeds_list:
                    set_seed(globalseed)
                    print(f"\n[EXPERIMENT] sorter={sorter}, Pk={Pk}, seed={globalseed}")

                    class_counts = Counter(train_labels_for_weights)
                    for i in range(num_classes):
                        if i not in class_counts:
                            class_counts[i] = 0

                    class_weights_for_sampler = torch.tensor([1.0 / (class_counts[i] + 1e-5) for i in range(num_classes)], dtype=torch.float32)
                    sample_weights = torch.tensor([class_weights_for_sampler[label] for label in train_labels_for_weights], dtype=torch.float32)

                    sampler = torch.utils.data.WeightedRandomSampler(
                        weights=sample_weights,
                        num_samples=len(sample_weights),
                        replacement=True
                    )

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, generator=torch.Generator().manual_seed(globalseed))
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    print(f"Dataset split: {len(train_dataset)} training graphs, {len(test_dataset)} testing graphs.")

                    seed_dir = os.path.join(experiment_dir, f"exp_{num_gene}genes_{num_classes}classes_{globalseed}")
                    os.makedirs(seed_dir, exist_ok=True)


                    # choose model (use the HADiGCN class from the package)
                    model = HADiGCN(
                        in_channels=num_features,
                        hidden_channels=16,
                        final_channels=32,
                        num_nodes=num_nodes,
                        num_classes=num_classes # Argument corrected to 'num_classes'
                    ).to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

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
                            device=device
                        ).to(device)

                    metrics = {
                        'train_loss': [], 'train_top1': [], 'train_top2': [], 'train_top4': [],
                        'test_top1': [], 'test_top2': [], 'test_top4': [],
                        'train_lineage_acc': [], 'test_lineage_acc': []
                    }

                    log_lines = ["Epoch\tTrainLoss\tTrainTop1\tTrainTop2\tTrainTop4\tTrainLineageAcc\tTestTop1\tTestTop2\tTestTop4\tTestLineageAcc\tTestMicroF1\tTestMacroF1\tLR"]

                    for epoch in range(num_epochs):
                        train_loss = train_one_epoch(
                            model=model,
                            dataloader=train_loader,
                            criterion=criterion,
                            optimizer=optimizer,
                            device=device
                        )


                        train_top1, train_top2, train_top4, _, _, train_lineage_acc = evaluate(model, train_loader, map_class_to_lineage_func)
                        test_top1, test_top2, test_top4, test_micro_f1, test_macro_f1, test_lineage_acc = evaluate(model, test_loader, map_class_to_lineage_func)

                        metrics['train_loss'].append(train_loss)
                        metrics['train_top1'].append(train_top1)
                        metrics['train_top2'].append(train_top2)
                        metrics['train_top4'].append(train_top4)
                        metrics['test_top1'].append(test_top1)
                        metrics['test_top2'].append(test_top2)
                        metrics['test_top4'].append(test_top4)
                        metrics['train_lineage_acc'].append(train_lineage_acc)
                        metrics['test_lineage_acc'].append(test_lineage_acc)

                        current_lr = optimizer.param_groups[0]['lr']
                        log_str = (f"{epoch+1}\t{train_loss:.4f}\t"
                                   f"{train_top1:.4f}\t{train_top2:.4f}\t{train_top4:.4f}\t{train_lineage_acc:.4f}\t"
                                   f"{test_top1:.4f}\t{test_top2:.4f}\t{test_top4:.4f}\t{test_lineage_acc:.4f}\t"
                                   f"{test_micro_f1:.4f}\t{test_macro_f1:.4f}\t{current_lr:.6f}")
                        log_lines.append(log_str)

                        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
                            print(f"Epoch {epoch+1:03d}: "
                                  f"Train Loss: {train_loss:.4f}, "
                                  f"Train Top1: {train_top1:.4f}, "
                                  f"Train Top2: {train_top2:.4f}, "
                                  f"Train Top4: {train_top4:.4f}, "
                                  f"Train Lineage: {train_lineage_acc:.4f}, "
                                  f"Test Top1: {test_top1:.4f}, "
                                  f"Test Top2: {test_top2:.4f}, "
                                  f"Test Top4: {test_top4:.4f}, "
                                  f"Test MicroF1: {test_micro_f1:.4f}, "
                                  f"Test MacroF1: {test_macro_f1:.4f}")

                    # save log
                    timestamp = datetime.now().strftime("%m%d_%H%M")
                    filename = f"training_log_baron_train_segerstolpe_test_{num_gene}genes_{num_classes}classes_{initial_lr}LR_{timestamp}.txt"
                    full_path = os.path.join(seed_dir, filename)
                    with open(full_path, "w") as f:
                        f.write("\n".join(log_lines))
                    print(f"\nTraining log saved to: {full_path}")

                    seed_metrics_for_pk.append({
                        'seed': globalseed,
                        'Top1 Acc': test_top1,
                        'Top2 Acc': test_top2,
                        'Top4 Acc': test_top4,
                        'Micro F1': test_micro_f1,
                        'Macro F1': test_macro_f1,
                        'Lineage Acc': test_lineage_acc
                    })

                    # Optional plotting (kept as your original logic)
                    if ENABLE_PLOTTING:
                        try:
                            fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5), sharex=True)
                            fig1.suptitle(f'Training Progress (PK: {Pk}, Seed: {globalseed}, LR: {initial_lr})', fontsize=16)

                            ax1.plot(metrics['train_loss'], label='Train Loss')
                            ax1.set_ylabel("Loss")
                            ax1.set_xlabel("Epochs")
                            ax1.set_title("Training Loss vs Epochs")
                            ax1.legend()
                            ax1.grid(True, linestyle='--', alpha=0.6)

                            ax2.plot(metrics['test_top1'], label='Test Top-1')
                            ax2.plot(metrics['test_top2'], label='Test Top-2', linestyle=':')
                            ax2.plot(metrics['test_top4'], label='Test Top-4', linestyle='--')
                            ax2.set_xlabel("Epochs")
                            ax2.set_ylabel("Accuracy")
                            ax2.set_title("Test Top-1, Top-2 and Top-4 Accuracies vs Epochs")
                            ax2.legend()
                            ax2.grid(True, linestyle='--', alpha=0.6)
                            ax2.set_ylim(0, 1.05)

                            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                            plot1_filename = f"Loss_TestAcc_{timestamp}.png"
                            plt.savefig(os.path.join(seed_dir, plot1_filename))
                            plt.close(fig1)

                            fig2, (ax3, ax4) = plt.subplots(1,2, figsize=(15, 5), sharex=True)
                            fig2.suptitle(f'Accuracy Metrics (PK: {Pk}, Seed: {globalseed}, LR: {initial_lr})', fontsize=16)

                            ax3.plot(metrics['train_top1'], label='Train Top-1')
                            ax3.plot(metrics['train_top2'], label='Train Top-2', linestyle=':')
                            ax3.plot(metrics['train_top4'], label='Train Top-4', linestyle='--')
                            ax3.set_ylabel("Accuracy")
                            ax3.set_xlabel("Epochs")
                            ax3.set_title("Training Top-1, Top-2 and Top-4 Accuracies vs Epochs")
                            ax3.legend()
                            ax3.grid(True, linestyle='--', alpha=0.6)
                            ax3.set_ylim(0, 1.05)

                            if 'train_lineage_acc' in metrics and 'test_lineage_acc' in metrics and len(metrics['train_lineage_acc']) > 0:
                                ax4.plot(metrics['train_lineage_acc'], label='Train Lineage Accuracy', color="green")
                                ax4.plot(metrics['test_lineage_acc'], label='Test Lineage Accuracy', color="brown", linestyle='--')
                                ax4.set_xlabel("Epochs")
                                ax4.set_ylabel("Lineage Accuracy")
                                ax4.set_title("Train and Test Lineage Accuracies vs Epochs")
                                ax4.legend()
                                ax4.grid(True)
                                ax4.set_ylim(0, 1.05)
                            else:
                                ax4.set_title("Lineage Accuracy Data Not Available/Empty")
                                ax4.text(0.5, 0.5, "Lineage Accuracy data is empty or not collected.",
                                        horizontalalignment='center', verticalalignment='center',
                                        transform=ax4.transAxes, color='red')
                                print("[WARNING] Lineage accuracy metrics not found or empty. Lineage Accuracy subplot will show a warning.")

                            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                            plot2_filename = f"TrainAcc_LineageAcc_{timestamp}.png"
                            plt.savefig(os.path.join(seed_dir, plot2_filename))
                            plt.close(fig2)
                        except Exception as e:
                            print(f"[WARNING] Plotting failed: {e}")

                # Append final metrics for this Pk
                metrics_dir = os.path.join(experiment_dir, "final_metrics")
                os.makedirs(metrics_dir, exist_ok=True)
                final_metrics_path = os.path.join(metrics_dir, "final_metrics.txt")
                file_exists = os.path.exists(final_metrics_path)

                with open(final_metrics_path, "a") as f:
                    if not file_exists:
                        f.write(f"--- Pk values: {pkstr} ---\n")
                        f.write(f"--- Number of genes: {num_gene} ---\n\n")

                    for m in seed_metrics_for_pk:
                        f.write("--- Run Details ---\n")
                        f.write(f"Seed: {m['seed']}\n")
                        f.write(f"Top1 Acc: {m['Top1 Acc']:.4f}\n")
                        f.write(f"Top2 Acc: {m['Top2 Acc']:.4f}\n")
                        f.write(f"Top4 Acc: {m['Top4 Acc']:.4f}\n")
                        f.write(f"Micro F1: {m['Micro F1']:.4f}\n")
                        f.write(f"Macro F1: {m['Macro F1']:.4f}\n")
                        f.write(f"Lineage Acc: {m['Lineage Acc']:.4f}\n\n")

                print(f"\nAll seed metrics for Pk {pkstr} saved to: {final_metrics_path}")

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main_experiment_runner()
