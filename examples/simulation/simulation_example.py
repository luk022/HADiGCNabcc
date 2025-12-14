#!/usr/bin/env python3
"""
example_simulation_full.py

Unified simulation experiment script for HADiGCN.
------------------------------------------------------
This script reproduces the simulation results from the thesis:
- Runs for multiple seeds (32, 42, 52).
- Runs for multiple training sizes (1000, 1500, 2000, 2500 graphs).
- Compares Cross-Entropy (CE) Loss against Top-K Loss.
- Saves raw logs per run and aggregates final results across seeds.

Usage:
    python example_simulation_full.py               # Full run (300 epochs, all seeds/configs)
    python example_simulation_full.py --debug       # Quick run (3 epochs, single seed/config)
"""

# Standard Libraries
import os
import argparse
import time
from collections import Counter

# External Libraries
import torch
from sklearn.metrics import f1_score
from torch_geometric.data import Data, DataLoader

# Package imports (using the public API)
from hadigcn import HADiGCN, TopKCrossEntropyLoss, CrossEntropyLossWrapper
from hadigcn import set_seed, get_device
from hadigcn.train import train_one_epoch

# Initialize device using the package utility
device = get_device()


# =============================================================================
# 1. PARAMETERS
# =============================================================================
# Base Configuration (used in both original scripts)
DEFAULT_SEEDS = [32, 42, 52]
DEFAULT_TRAIN_SIZES = [80] #[100, 500, 1500]#use smaller size for debugging
TEST_GRAPHS = 20 #1000   # pick smaller size smaller for debugging
NUM_NODES = 15
NUM_FEATURES = 1
NUM_CLASSES = 4
BATCH_SIZE = 4
INITIAL_LR = 0.001
FULL_EPOCHS = 3#300

# Loss Configuration (combined from both original scripts)
# Key = sorter name, Value = list of Pk values
SORTER_TO_PK = {
    # Cross-Entropy Loss (Pk=None implies standard CE Loss in the wrapper)
    "celoss": [None],

    # Top-K Loss configurations
    "odd_even": [
        [0.5, 0.5],  # Pk = [0.5, 0.5] from Top-K script
        [0.2, 0.8],  # Pk = [0.2, 0.8] from Top-K script
    ]
}


# =============================================================================
# 2. DATA GENERATION (Copied from original scripts)
# =============================================================================

def generate_multivariate_features(num_nodes, num_features):
    # Implements the specific covariance structure for the synthetic data
    covariance_matrix = torch.zeros((num_nodes, num_nodes))

    covariance_matrix[0, 1] = 0.5; covariance_matrix[1, 0] = 0.5
    covariance_matrix[2, 1] = 0.5; covariance_matrix[1, 2] = 0.5
    covariance_matrix[0, 2] = 0.5; covariance_matrix[2, 0] = 0.5

    covariance_matrix[4, 3] = 0.5; covariance_matrix[3, 4] = 0.5
    covariance_matrix[4, 5] = 0.5; covariance_matrix[5, 4] = 0.5
    covariance_matrix[4, 6] = 0.5; covariance_matrix[6, 4] = 0.5

    covariance_matrix[7, 8] = 0.5; covariance_matrix[8, 7] = 0.5

    for i in range(num_nodes):
        covariance_matrix[i, i] = 1

    mean_vector = torch.zeros(num_nodes)
    mvn = torch.distributions.MultivariateNormal(mean_vector, covariance_matrix)
    
    features = mvn.sample((num_features,)).t()
    return features




def create_balanced_synthetic_data(num_graphs, num_nodes, num_features, num_classes=4):
    # Define three beta tensors with mean zero
    beta1 = torch.tensor([2.5, 2, 3, -3, -3, -2, -1, 0, 0, 0, 0, 0, 1, 1, 0], dtype=torch.float32)
    beta1 = beta1 - beta1.mean()
    
    beta2 = torch.tensor([-1, 2, 0, 3, -2, 1, -3, 0, 0, 0, 0, 0, 2, -1, 0], dtype=torch.float32)
    beta2 = beta2 - beta2.mean()
    
    beta3 = torch.tensor([0, -2, 1, -1, 3, 0, 2, -3, 0, 0, 0, 0, 1, -2, 0], dtype=torch.float32)
    beta3 = beta3 - beta3.mean()

    # Rebuild edge_index dynamically
    edges = [
        (0, 1), (0, 2), (1, 2),  # Node 0, 1, 2 connections
        (4, 3), (4, 5), (4, 6),  # Node 4 with Nodes 3, 5, 6
        (7, 8)  # Node 7 and Node 8
    ]
    
    edge_index = []
    # NEW: Also prepare edge_attr_list to store weights for each edge
    edge_attr_list = [] 
    
    EDGE_WEIGHT_FOR_GCN = 0.5 # <--- Define the weight for GCN edges here

    for edge in edges:
        edge_index.append(edge)
        edge_index.append((edge[1], edge[0]))  # Add reverse edge
        
        # Add the weight for both forward and reverse edges
        edge_attr_list.append(EDGE_WEIGHT_FOR_GCN)
        edge_attr_list.append(EDGE_WEIGHT_FOR_GCN)
        

    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # NEW: Convert edge_attr_list to a tensor
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    
    # Rest of the data generation code...
    dataset = []
    labels = []
    
    for _ in range(num_graphs):
        # NOTE: Removed the per-graph seeding to rely on the global seed
        x = generate_multivariate_features(num_nodes, num_features)
        logit1 = torch.sum(beta1 * x.squeeze())
        logit2 = torch.sum(beta2 * x.squeeze())
        logit3 = torch.sum(beta3 * x.squeeze())
        p1 = torch.sigmoid(logit1)
        p2 = torch.sigmoid(logit2)
        p3 = torch.sigmoid(logit3)
        b1 = torch.bernoulli(p1).item()
        if b1 == 1.0:
            b2 = torch.bernoulli(p2).item()
            y = 0 if b2 == 1.0 else 1
        else:
            b3 = torch.bernoulli(p3).item()
            y = 2 if b3 == 1.0 else 3
            
        labels.append(y)
        # MODIFIED: Pass edge_attr to the Data object
        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([y], dtype=torch.long)))

    print(f"Class Distribution: {dict(Counter(labels))}")
    return dataset
# =============================================================================
# 3. EVALUATION
# =============================================================================

def evaluate_model(model, loader, device):
    """Evaluates Top-1, Top-3, Micro F1, and Macro F1."""
    model.eval()
    all_preds = []
    all_labels = []
    top1_correct = 0
    top3_correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # Use HADiGCN signature: model(x, edge_index, batch, edge_attr)
            logits = model(data.x, data.edge_index, data.batch, data.edge_attr)
  
            # Top-1 and Top-3 accuracy
            _, top3_preds = logits.topk(k=3, dim=1)
            top1_pred = logits.argmax(dim=1)
            
            top1_correct += (top1_pred == data.y).sum().item()
            top3_correct += (top3_preds == data.y.unsqueeze(1)).any(dim=1).sum().item()
            
            total += data.y.size(0)
            
            all_preds.extend(top1_pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    top1_acc = top1_correct / total
    top3_acc = top3_correct / total
    
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'top1': top1_acc,
        'top3': top3_acc,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }


# =============================================================================
# 4. MAIN EXPERIMENT LOOP
# =============================================================================

def run_simulation(args):
    # Set run parameters
    epochs = args.epochs
    seeds = args.seeds
    train_sizes = args.train_sizes

    base_output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulation_results",
        f"ep_{epochs}_lr_{INITIAL_LR}_b_{BATCH_SIZE}"
    )
    os.makedirs(base_output_dir, exist_ok=True)
    
    # ------------------------------------
    # Generate FULL Dataset (needed once)
    # ------------------------------------
    print(f"Generating full synthetic dataset ({max(train_sizes) + TEST_GRAPHS} graphs)...")
    full_dataset = create_balanced_synthetic_data(
        max(train_sizes) + TEST_GRAPHS,
        NUM_NODES,
        NUM_FEATURES
    )

    # Use a fixed test set
    test_dataset = full_dataset[-TEST_GRAPHS:]
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Fixed test set size: {len(test_dataset)}")
    
    # Run loop over seeds, train sizes, sorters, and Pk values
    for seed in seeds:
        set_seed(seed) # Use package function
        print(f"\n=======================================================")
        print(f"  STARTING EXPERIMENTS FOR SEED: {seed}")
        print(f"=======================================================")

        for train_size in train_sizes:
            # Create training set subset
            train_dataset = full_dataset[:train_size]
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed) # Use seed for train shuffle
            )
            train_eval_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
            print(f"\n--- Train Size: {train_size} graphs ---")
            
            for sorter, pk_list in SORTER_TO_PK.items():
                for Pk in pk_list:
                    # Determine Loss Configuration
                    pk_slug = "celoss"
                    if Pk is None:
                        # Cross-Entropy Loss
                        criterion = CrossEntropyLossWrapper().to(device)
                        print(f"--- Running: Loss=CrossEntropy, Seed={seed}, Size={train_size} ---")
                    else:
                        # Top-K Loss
                        pk_slug = "".join([str(x).replace('.', '') for x in Pk])
                        criterion = TopKCrossEntropyLoss(
                            diffsort_method=sorter,
                            inverse_temperature=2.0,
                            p_k=Pk,
                            n=NUM_CLASSES,
                            m=3, # Fixed value from your topk script
                            distribution='cauchy', # Fixed value from your topk script
                            device=device
                        ).to(device)
                        print(f"--- Running: Loss=TopK({sorter}), Pk={Pk}, Seed={seed}, Size={train_size} ---")

                    # Initialize model
                    torch.cuda.empty_cache()
                    set_seed(seed)
                    
                    # Use the imported HADiGCN class
                    model = HADiGCN(
                        in_channels=NUM_FEATURES,
                        hidden_channels=16,
                        final_channels=32,
                        num_nodes=NUM_NODES,
                        num_classes=NUM_CLASSES
                    ).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
                    
                    # Log file setup
                    outdir = os.path.join(base_output_dir, f"{sorter}_pk_{pk_slug}")
                    os.makedirs(outdir, exist_ok=True)
                    logpath = os.path.join(outdir, f"seed_{seed}_train{train_size}_log.txt")
                    
                    # Training Loop
                    start_time = time.time()
                    log_lines = ["Epoch\tTrainLoss\tTestTop1\tTestTop3\tTestMicroF1\tTestMacroF1"]

                    for epoch in range(1, epochs + 1):
                        # Train one epoch
                        train_loss = train_one_epoch(
                            model=model,
                            dataloader=train_loader,
                            criterion=criterion,
                            optimizer=optimizer,
                            device=device
                        )

                        # Evaluate on test set
                        test_metrics = evaluate_model(model, test_loader, device)
                        
                        # Logging (only final/intermediate, as per your previous design)
                        log_lines.append(
                            f"{epoch}\t{train_loss:.4f}\t{test_metrics['top1']:.4f}\t"
                            f"{test_metrics['top3']:.4f}\t{test_metrics['micro_f1']:.4f}\t"
                            f"{test_metrics['macro_f1']:.4f}"
                        )

                        if epoch % 50 == 0 or epoch == epochs:
                            print(
                                f"  > Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | "
                                f"T1 Acc: {test_metrics['top1']:.4f} | T3 Acc: {test_metrics['top3']:.4f}"
                            )
                    
                    # Final Metrics and Save
                    total_time = time.time() - start_time
                    final_metrics = evaluate_model(model, test_loader, device)

                    with open(logpath, "w") as f:
                        f.write(f"Seed: {seed}, TrainSize: {train_size}, Loss: {sorter}, Pk: {Pk}, Epochs: {epochs}\n")
                        f.write(f"Total Time: {total_time:.1f}s\n")
                        f.write("\n".join(log_lines))
                    
                    print(
                        f"  -> FINAL: T1: {final_metrics['top1']:.4f}, T3: {final_metrics['top3']:.4f}, "
                        f"MicroF1: {final_metrics['micro_f1']:.4f}, MacroF1: {final_metrics['macro_f1']:.4f}"
                    )
                    print(f"  -> Log saved to: {logpath}\n")

# =============================================================================
# 5. ARGUMENT PARSING
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Simulation Experiment for HADiGCN.")
    
    parser.add_argument('--debug', action='store_true', help='Run a quick 3-epoch test with one seed/config.')
    parser.add_argument('--epochs', type=int, default=FULL_EPOCHS, help=f'Number of epochs (default: {FULL_EPOCHS}).')
    
    args = parser.parse_args()

    if args.debug:
        print("!!! RUNNING IN DEBUG MODE (3 EPOCHS, SINGLE SEED, SINGLE CONFIG) !!!")
        args.epochs = 3
        args.seeds = DEFAULT_SEEDS[:1]
        args.train_sizes = DEFAULT_TRAIN_SIZES[:1]
        SORTER_TO_PK = {"odd_even": SORTER_TO_PK["odd_even"][:1]}
    else:
        args.seeds = DEFAULT_SEEDS
        args.train_sizes = DEFAULT_TRAIN_SIZES

    run_simulation(args)
    
    ## run in console for debugging
    ## %run "D:/Github repos/local cloned/HADiGCN_test-new/HADiGCN_test new/examples/simulation/simulation_example.py" --debug