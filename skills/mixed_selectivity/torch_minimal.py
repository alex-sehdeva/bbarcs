import torch
from torch import nn
from torch_geometric.nn import Message Passing
from torch_geometric.utils import add_self_loops

class MixedSelectivityNodeEncoder(nn.Module):
    def __init__(self, in_local, in_global, hidden, out):
        super().__init__()
        self.fc_local = nn.Linear(in_local, hidden)
        self.fc_global = nn.Linear(in_global, hidden)
        # Extra layer tha mixes local, global, and multiplicative terms
        self.fc_mix = nn.Linear(3*hidden, out)
        self.act = nn.ReLU()

    def forward(self, x_local, x_global):
        """
        x_local: [N, in_local]
        x_global: [in_global] or [B, in_global] broadcastable to [N, in_global]
        """
        h_local = self.act(self.fc_local(x_local))
        h_global = self.act(sel.fc_global(x_global))

        # Broadcast global features to each node if needed
        if h_global.dim() ==1:
            h_global = h_global.unsqueeze(0).expand_as(h_local)

        # Elementwise product = interaction term
        h_int = h_local * h_global

        h_cat = torch.cat([h_local, h_global, h_int], dim=-1)
        return self.act(self.fc_mix(h_cat))

class MixedSelectivityGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') # sum aggregation
        self.lin_msg= nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Nonlinear transform of neighbor features
        return self.act(self.lin_msg(x_j))

    def update(self, aggr_out, x):
        # Combine aggregated message with self features
        h = aggr_out + self.lin_self(x)
        return self.act(h)

