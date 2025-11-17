# arc_gnn_model.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

from arc_rule_families import RuleFamily

class ArcRuleFamilyGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        out_dim = len(RuleFamily)  # multi-label output

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Graph-level pooling (we want 1 prediction per example)
        x = global_mean_pool(x, batch)

        logits = self.lin(x)
        return logits

