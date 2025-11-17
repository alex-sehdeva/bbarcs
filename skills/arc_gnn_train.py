# arc_gnn_train.py

import torch
from torch_geometric.loader import DataLoader
from torch.nn import BCEWithLogitsLoss

from arc_synthetic_dataset_builder import build_synthetic_training_pairs
from arc_rule_families import RuleFamily
from arc_gnn_dataset import ArcRuleFamilyDataset
from arc_gnn_model import ArcRuleFamilyGNN


def train_rule_family_gnn(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
):
    # 1. Build synthetic training data
    training_pairs = build_synthetic_training_pairs(
        n_translation=100,
        n_recolor=100,
        n_both=100,
    )

    dataset = ArcRuleFamilyDataset(training_pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Instantiate model
    in_dim = dataset[0].x.shape[1]
    model = ArcRuleFamilyGNN(in_dim=in_dim, hidden_dim=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch)  # [batch_size, num_families]

            # batch.y is flat: [batch_size * num_families]
            num_graphs = batch.num_graphs
            num_families = logits.size(1)

            targets = batch.y.view(num_graphs, num_families)  # [batch_size, num_families]

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * num_graphs

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f}")

    return model

if __name__ == "__main__":
    model = train_rule_family_gnn()

