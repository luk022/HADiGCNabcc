
"""
example_tuning_alpha.py

Varying-alpha experiment for Top-k loss (Odd-Even network)
----------------------------------------------------------

This script reproduces the tuning experiment used in the thesis:

- Fixed synthetic dataset (same as simulation setup)
- Top-K loss using Pk = [alpha, 0, 1 - alpha]
- alpha ∈ [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
- 500 training epochs
- One seed
- No plotting (you will generate figures externally)
- Saves validation accuracy for each alpha under four dataset-size cases
  (“Case 1”–“Case 4” from the thesis)

"""
'''
from torch_geometric.data import Data, DataLoader
import torch
import numpy as np
import random
from collections import Counter
'''

# Standard Libraries
import os
import random
from collections import Counter
from datetime import datetime
import pickle

# External Libraries
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader

# Package imports (using the public API)
# HADiGCN, TopKCrossEntropyLoss, set_seed, and get_device are now imported from __init__.py
from hadigcn import HADiGCN, TopKCrossEntropyLoss, set_seed, get_device
from hadigcn.train import train_one_epoch
# Initialize device using the package utility
device = get_device()

def generate_multivariate_features(num_nodes, num_features):
    """
    Generates multivariate normal features for a graph with a specific covariance structure among nodes.
    
    Args:
    - num_nodes (int): Number of nodes in the graph.
    - num_features (int): Number of features for each node.

    Returns:
    - torch.Tensor: A tensor containing node features of shape (num_nodes, num_features).
    """
# Initialize the covariance matrix with zeros for node correlations
    covariance_matrix = torch.zeros((num_nodes, num_nodes))

    # Define specific correlations between nodes
    # Node 1 and Node 2
    covariance_matrix[0, 1] = 0.5
    covariance_matrix[1, 0] = 0.5  # Symmetric
    
    covariance_matrix[2, 1] = 0.5
    covariance_matrix[1, 2] = 0.5  # Symmetric
    
    covariance_matrix[0, 2] = 0.5
    covariance_matrix[2, 0] = 0.5  # Symmetric

    # Node 5 with Nodes 4, 6, and 7
    covariance_matrix[4, 3] = 0.5
    covariance_matrix[3, 4] = 0.5  # Symmetric

    covariance_matrix[4, 5] = 0.5
    covariance_matrix[5, 4] = 0.5  # Symmetric

    covariance_matrix[4, 6] = 0.5
    covariance_matrix[6, 4] = 0.5  # Symmetric

    # Node 8 and Node 9
    covariance_matrix[7, 8] = 0.5
    covariance_matrix[8, 7] = 0.5  # Symmetric

    # Each node should have a variance of 1
    for i in range(num_nodes):
        covariance_matrix[i, i] = 1

    mean_vector = torch.zeros(num_nodes)  # Mean for each node
    mvn = torch.distributions.MultivariateNormal(mean_vector, covariance_matrix)
    
    # Sample features for each node
    features = mvn.sample((num_features,)).t()  # Shape: (num_nodes, num_features)
    ##print(f'features.shape is {features.shape}') # torch.Size([15, 1])
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


# ======================
# Global Configuration
# ======================
train_graphs = 20 #100#500           # Total training graphs (1000 train + 500 test)
test_graphs = 200  #1000#500           # Fixed test set size
num_epochs = 3 #500           # Training epochs per alpha
initial_lr = 0.001          # Learning rate
batch_size = 4              # Batch size
alpha_list = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]  # Alpha values to test
num_nodes = 15              # Nodes per graph
num_features = 1            # Features per node
num_classes = 4             # Number of classes
script_dir = os.path.dirname(os.path.abspath(__file__))  # Script location



def evaluate(model, loader):
    model.eval()
    top1_correct = 0
    top2_correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch, data.edge_attr)
            _, top2_preds = logits.topk(k=2, dim=1)
            top1_pred = logits.argmax(dim=1)
            
            top1_correct += (top1_pred == data.y).sum().item()
            top2_correct += (top2_preds == data.y.unsqueeze(1)).any(dim=1).sum().item()
            total += data.y.size(0)
    
    return top1_correct/total, top2_correct/total



# ======================
# Experiment Execution
# ======================
def run_alpha_experiments():
    set_seed(32)
  
    # Generate dataset
    full_dataset = create_balanced_synthetic_data(
        train_graphs + test_graphs,
        num_nodes,
        num_features,
        num_classes
    )
    random.Random(32).shuffle(full_dataset)
    
    # Create fixed splits
    train_dataset = full_dataset[:train_graphs]
    test_dataset = full_dataset[-test_graphs:]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Store results
    all_metrics = {
        'alpha': [],
        'final_train_top1': [],
        'final_train_top2': [],
        'final_test_top1': [],
        'final_test_top2': []
    }
    
    # Check for directories of completed experiments
    base_save_dir = os.path.join(script_dir, "alpha simulation_results", f"{num_epochs} epochs", f"{test_graphs} test {train_graphs} train")
    completed_alphas = []
    
    for alpha in alpha_list:
        alpha_str = f"{alpha:.2f}".replace('.', '')
        alpha_dir = os.path.join(base_save_dir, f"alpha_{alpha_str}_lr{initial_lr}_ep{num_epochs}")
        if os.path.exists(alpha_dir):
            completed_alphas.append(alpha)
            print(f"skip the finished alpha={alpha:.2f}")
            
    # Determine which alpha values still need to be run (i.e., results not found in directory)
    remaining_alphas = [alpha for alpha in alpha_list if alpha not in completed_alphas]

    for alpha in remaining_alphas:
        alpha_str = f"{alpha:.2f}".replace('.', '')
        print(f"\n=== Running experiment for α={alpha:.2f} ===")

        save_dir = os.path.join(base_save_dir, f"alpha_{alpha_str}_lr{initial_lr}_ep{num_epochs}")
        os.makedirs(save_dir, exist_ok=True)

        # Initialize model
        torch.cuda.empty_cache()
        set_seed(32)
        model = HADiGCN(
            in_channels=num_features,
            hidden_channels=16,
            final_channels=32,
            num_nodes=num_nodes,
            num_classes=num_classes
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        criterion = TopKCrossEntropyLoss(
            diffsort_method='odd_even',
            inverse_temperature=2.0,
            p_k=[1-alpha, alpha],
            n=num_classes,
            m=3,
            distribution='cauchy',
            device=device
        ).to(device)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(32)
        )
        train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        log_lines = ["Epoch\tTrainLoss\tTrainTop1\tTrainTop2\tTestTop1\tTestTop2"]
        for epoch in range(num_epochs):
            # Use the robust train_one_epoch function instead of the manual loop
            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )
           
            train_top1, train_top2 = evaluate(model, train_eval_loader)
            test_top1, test_top2 = evaluate(model, test_loader)
            
            log_lines.append(
                f"{epoch+1}\t{train_loss:.4f}\t{train_top1:.4f}\t"
                f"{train_top2:.4f}\t{test_top1:.4f}\t{test_top2:.4f}"
            )

        # Save training log
        with open(os.path.join(save_dir, "training_log.txt"), "w") as f:
            f.write("\n".join(log_lines))

        # Store final metrics
        all_metrics['alpha'].append(alpha)
        all_metrics['final_train_top1'].append(train_top1)
        all_metrics['final_train_top2'].append(train_top2)
        all_metrics['final_test_top1'].append(test_top1)
        all_metrics['final_test_top2'].append(test_top2)

    def load_existing_metrics():
        existing_metrics = {'alpha': [], 'final_train_top1': [], 'final_train_top2': [], 
                           'final_test_top1': [], 'final_test_top2': []}
        for alpha in completed_alphas:
            alpha_str = f"{alpha:.2f}".replace('.', '')
            log_path = os.path.join(base_save_dir, f"alpha_{alpha_str}_lr{initial_lr}_ep{num_epochs}", "training_log.txt")
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split('\t')
                        existing_metrics['alpha'].append(alpha)
                        existing_metrics['final_train_top1'].append(float(last_line[2]))
                        existing_metrics['final_train_top2'].append(float(last_line[3]))
                        existing_metrics['final_test_top1'].append(float(last_line[4]))
                        existing_metrics['final_test_top2'].append(float(last_line[5]))
        return existing_metrics

    existing_metrics = load_existing_metrics()
    for key in all_metrics:
        all_metrics[key] = existing_metrics[key] + all_metrics[key]

    # 按 alpha 排序
    sorted_indices = sorted(range(len(all_metrics['alpha'])), key=lambda x: all_metrics['alpha'][x])
    for key in all_metrics:
        all_metrics[key] = [all_metrics[key][i] for i in sorted_indices]

    # Save results
    timestamp = datetime.now().strftime("%m%d_%H%M")
    metrics_path = os.path.join(script_dir, "alpha simulation_results", f"full_metrics_ep{num_epochs}_lr{initial_lr}_{timestamp}.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(all_metrics, f)

    # Plot results
    plt.figure(figsize=(10, 8))           
    plt.plot(all_metrics['alpha'], all_metrics['final_test_top1'], 'o-', label='Top-1', linewidth=3,markersize=8, color='lightcoral')  # Lighter red  # ✅ Doubled line width
    plt.plot(all_metrics['alpha'], all_metrics['final_test_top2'], 'o--', label='Top-2', linewidth=3,markersize=8, color='lightblue')  # Lighter blue  # ✅ Slightly larger dot # ✅ Doubled line width
    plt.xlabel(r'$\alpha$ ($P_K$ = [$1-\alpha$, $\alpha$])', fontsize = 24)  
    plt.ylabel('Accuracy', fontsize=24) 
    #plt.title(f'Final Test Performance Comparison (Epoch {num_epochs})')
    
    plt.xticks(all_metrics['alpha'], fontsize=18)  
    plt.yticks(fontsize=18)                        
    
    plt.legend(fontsize=18) 
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, "alpha simulation_results", f"{num_epochs} epochs", f"{test_graphs} test {train_graphs} train", f"new test_comparison_{num_epochs} epochs_lr {initial_lr}__{test_graphs}test{train_graphs}train.png"))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.plot(all_metrics['alpha'], all_metrics['final_train_top1'], 'o-', label='Train Top-1', linewidth=3,markersize=8, color='lightcoral')  # Lighter red  # ✅ Doubled line width
    plt.plot(all_metrics['alpha'], all_metrics['final_train_top2'], 'o--', label='Train Top-2', linewidth=3,markersize=8, color='lightblue')  # Lighter blue  # ✅ Doubled line width
    plt.xlabel(r'$\alpha$ ($P_K$ = [$1-\alpha$, $\alpha$])', fontsize = 24)
    plt.ylabel('Accuracy',fontsize=24)
    #plt.title(f'Final Train Performance Comparison (Epoch {num_epochs})')
    #plt.title(f'Final Train Performance (Epoch {num_epochs}, LR={initial_lr})',
             # fontsize=26,
             # pad=22  
              #)
    
    plt.xticks(all_metrics['alpha'], fontsize=18)  
    plt.yticks(fontsize=18)                        
    plt.legend(fontsize=18)  
    plt.grid(True)
    #plt.ylim(0.6, 1.02)  # Set y-axis 
    #plt.ylim(bottom=0.6)
    plt.savefig(os.path.join(script_dir, "alpha simulation_results", f"{num_epochs} epochs", f"{test_graphs} test {train_graphs} train", f"new train_comparison_{num_epochs} epochs_lr {initial_lr}__{test_graphs}test{train_graphs}train.png"))
    plt.close()

if __name__ == "__main__":
    run_alpha_experiments()
