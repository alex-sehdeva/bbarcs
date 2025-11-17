# Mixed selectivity demo with PyTorch Geometric
# ---------------------------------------------
# Task: Node classification where label = XOR(local_bit, global_context_bit)
# We compare:
#   1) BaselineGNN: additive local + global features
#   2) MixedGNN: uses MixedSelectivityNodeEncoder with multiplicative mixing

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

# ---- 1. Synthetic dataset ---------------------------------------------------

def make_graph(num_nodes, ctx, device="cpu"):
    """
    num_nodes: number of nodes
    ctx: global context bit ∈ {0,1}
    Each node gets a local bit stim ∈ {0,1}.
    Label: y = XOR(stim, ctx)
    """
    # Local bit per node
    stim = torch.randint(0, 2, (num_nodes, 1), dtype=torch.float32)
    # Optional noise feature (not strictly needed)
    noise = torch.randn(num_nodes, 1) * 0.1

    x_local = torch.cat([stim, noise], dim=-1)  # [N, 2]

    # Global/context bit as 1D feature
    x_global = torch.tensor([[ctx]], dtype=torch.float32)  # [1, 1]

    # Labels: XOR(stim, ctx)
    y = (stim.squeeze(-1).long() ^ int(ctx))  # [N]

    # Use a simple fully-connected undirected graph
    adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
    edge_index, _ = dense_to_sparse(adj)

    data = Data(
        x=x_local.to(device),
        edge_index=edge_index.to(device),
        y=y.to(device),
    )
    # Attach global/context feature as a custom attribute
    data.global_feat = x_global.to(device)  # [1,1]
    return data


def make_dataset(num_graphs=500, num_nodes=8, device="cpu"):
    graphs = []
    for _ in range(num_graphs):
        ctx = torch.randint(0, 2, (1,)).item()  # 0 or 1
        g = make_graph(num_nodes, ctx, device=device)
        graphs.append(g)
    return graphs


device = "cuda" if torch.cuda.is_available() else "cpu"
train_graphs = make_dataset(600, num_nodes=8, device=device)
test_graphs  = make_dataset(200, num_nodes=8, device=device)

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_graphs,  batch_size=64, shuffle=False)

# ---- 2. Mixed-selectivity building blocks -----------------------------------

class MixedSelectivityNodeEncoder(nn.Module):
    """
    Takes local node features and a global/context feature vector
    and produces mixed-selective node embeddings.

    h_local = f(W_local * x_local)
    h_global = f(W_global * x_global)
    h_int = h_local * h_global  (elementwise)
    output = f(W_mix * [h_local, h_global, h_int])
    """
    def __init__(self, in_local, in_global, hidden, out, ablate=False):
        super().__init__()
        self.ablate=ablate
        self.fc_local = nn.Linear(in_local, hidden)
        self.fc_global = nn.Linear(in_global, hidden)
        if self.ablate:
            self.fc_mix = nn.Linear(2 * hidden, out)
        else:
            self.fc_mix = nn.Linear(3 * hidden, out)
        self.act = nn.ReLU()

    def forward(self, x_local, x_global, batch):
        """
        x_local: [N, in_local]
        x_global: [B, in_global]  (one global vector per graph)
        batch: [N] mapping nodes → graph index (0..B-1)

        We broadcast each graph's global feature to its nodes.
        """
        h_local = self.act(self.fc_local(x_local))

        # Get global features per node by indexing with batch
        h_global_graph = self.act(self.fc_global(x_global))  # [B, hidden]
        h_global = h_global_graph[batch]                     # [N, hidden]

        if self.ablate:
            h_cat = torch.cat([h_local, h_global], dim=-1)
        else:
            h_int = h_local * h_global                           # elementwise interaction
            h_cat = torch.cat([h_local, h_global, h_int], dim=-1)
        return self.act(self.fc_mix(h_cat))                  # [N, out]


class MixedSelectivityGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # sum aggregation
        self.lin_msg = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.act(self.lin_msg(x_j))

    def update(self, aggr_out, x):
        h = aggr_out + self.lin_self(x)
        return self.act(h)

# ---- 3. Two models: baseline vs mixed-selective -----------------------------

class BaselineGNN(nn.Module):
    """
    Baseline model:
    - Encodes global context additively, no multiplicative mixing.
    - Goal: show it is weaker on XOR-like tasks.
    """
    def __init__(self, in_local=2, in_global=1, hidden=32, num_classes=2):
        super().__init__()
        self.fc_global = nn.Linear(in_global, hidden)
        self.fc_cat = nn.Linear(in_local + hidden, hidden)
        self.gnn = MixedSelectivityGNNLayer(hidden, hidden)
        self.out = nn.Linear(hidden, num_classes)
        self.act = nn.ReLU()

    def forward(self, data):
        x_local = data.x                         # [N, in_local]
        edge_index = data.edge_index             # [2, E]
        batch = data.batch                       # [N]
        # Build global matrix [B, in_global]
        # each graph in batch has a global_feat; we gather them like this:
        # (we rely on the fact that graphs are not shuffled inside a batch)
        # so we reconstruct by taking the first node of each graph.
        # Alternatively, we could store global_feat at data-level in a custom dataset.
        # Here we'll attach x_global at batch-level via data.global_feat.
        x_global = data.global_feat              # expected [B, in_global] or [B]
        # Ensure 2D: [B, in_global]
        if x_global.dim() == 1:
            x_global = x_global.unsqueeze(-1)    # [B, 1]

        # Encode global, then broadcast to nodes:
        h_global_graph = self.act(self.fc_global(x_global))  # [B, hidden]
        h_global = h_global_graph[batch]                     # [N, hidden]

        h_cat = torch.cat([x_local, h_global], dim=-1)       # [N, in_local+hidden]
        h0 = self.act(self.fc_cat(h_cat))                    # [N, hidden]

        h = self.gnn(h0, edge_index)                         # [N, hidden]
        logits = self.out(h)                                 # [N, num_classes]
        return logits


class MixedGNN(nn.Module):
    """
    Mixed-selectivity model:
    - Uses MixedSelectivityNodeEncoder to combine local + global with multiplicative interaction.
    """
    def __init__(self, ablate=False, in_local=2, in_global=1, hidden=32, num_classes=2):
        super().__init__()
        self.encoder = MixedSelectivityNodeEncoder(in_local, in_global, hidden, hidden, ablate)
        self.gnn = MixedSelectivityGNNLayer(hidden, hidden)
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x_local = data.x
        edge_index = data.edge_index
        batch = data.batch
        x_global = data.global_feat              # [B, in_global] or [B]

        if x_global.dim() == 1:
            x_global = x_global.unsqueeze(-1)    # [B, 1]

        h0 = self.encoder(x_local, x_global, batch)  # [N, hidden]
        h = self.gnn(h0, edge_index)                 # [N, hidden]
        logits = self.out(h)                         # [N, num_classes]
        return logits

# ---- 4. Attach global_feat to batched data properly -------------------------

# PyG doesn't know about global_feat by default, so we need a custom collate.
# But DataLoader already merges attributes with same shape logic.
# Here each graph has data.global_feat with shape [1]; batch will stack them -> [B, 1].
# That’s exactly what we want.

# ---- 5. Training / evaluation helpers ---------------------------------------

def train_one_epoch(model, loader, optimizer, device="cpu"):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_nodes
        total_examples += batch.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    total_correct = 0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        pred = logits.argmax(dim=-1)
        total_correct += (pred == batch.y).sum().item()
        total_examples += batch.num_nodes
    return total_correct / total_examples


# ---- 6. Run ablation --------------------------------------------------------

def run_experiment(ModelClass, name, ablate=False, epochs=20):
    model = ModelClass(ablate).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n=== Training {name} ===")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, device=device)
        if epoch % 5 == 0 or epoch == 1:
            acc_train = evaluate(model, train_loader, device=device)
            acc_test = evaluate(model, test_loader, device=device)
            print(f"Epoch {epoch:02d} | loss={loss:.4f} | "
                  f"train_acc={acc_train:.3f} | test_acc={acc_test:.3f}")
    final_test = evaluate(model, test_loader, device=device)
    print(f"{name} final test accuracy: {final_test:.3f}")
    return final_test


if __name__ == "__main__":
    #acc_baseline = run_experiment(BaselineGNN, "BaselineGNN (additive)")
    acc_baseline    = run_experiment(MixedGNN, "MixedGNN (mixed selectivity)", ablate=True)
    acc_mixed    = run_experiment(MixedGNN, "MixedGNN (mixed selectivity)", ablate=False)

    print("\n=== Summary ===")
    print(f"Baseline (additive) test acc: {acc_baseline:.3f}")
    print(f"Mixed-selectivity test acc : {acc_mixed:.3f}")

