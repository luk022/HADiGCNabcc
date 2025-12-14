import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class HADiGCN(nn.Module):
    """
    Thesis-default HADiGCN model:
    - 2 GCN layers
    - Flatten
    - FC -> ReLU -> FC
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        final_channels,
        num_nodes,
        num_classes,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, final_channels)

        ##self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.fc1 = nn.Linear(final_channels * num_nodes, num_nodes)
        self.fc2 = nn.Linear(num_nodes, num_classes)

        self.num_nodes = num_nodes

    def forward(self, x, edge_index, batch, edge_attr):
        edge_index_norm, edge_weight_norm = gcn_norm(
            edge_index, edge_attr, x.size(0), dtype=x.dtype
        )

        x = self.conv1(x, edge_index_norm, edge_weight_norm)
        x = F.relu(x)
        x = self.conv2(x, edge_index_norm, edge_weight_norm)
        x = F.relu(x)

        current_batch_size = batch.max().item() + 1
        x = x.view(current_batch_size, -1)

        ##x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x





class MLP1Layer(nn.Module):
    """Simple baseline: 1 hidden layer MLP."""

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        ##self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        ##x = self.dropout(x)
        x = self.fc2(x)
        return x


class MLP2Layer(nn.Module):
    """Simple baseline: 2 hidden layer MLP (for comparison)."""

    def __init__(self, input_dim, hidden_dim, num_classes):##dropout=0.0
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        ##self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        ##x = self.dropout(x)
        x = self.fc3(x)
        return x
