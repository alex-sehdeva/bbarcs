# arc_gnn_inference.py

from typing import List
import torch

from arc_graph_core import extract_objects_from_grid
from arc_gnn_dataset import graphs_to_pyg_data
from arc_rule_families import RuleFamily


def predict_rule_family_probs_for_pair(
    model,
    inp_grid: list[list[int]],
    out_grid: list[list[int]],
    device: str = "cpu",
) -> List[float]:
    """
    Run the trained GNN on a single (input, output) pair and return
    a list of probabilities, one per RuleFamily.
    """
    model.to(device)
    model.eval()

    ig = extract_objects_from_grid(inp_grid)
    og = extract_objects_from_grid(out_grid)

    num_families = len(RuleFamily)
    # label_indices is empty because we don't care about labels here
    data = graphs_to_pyg_data(ig, og, label_indices=[], num_families=num_families)

    # For a single graph, batch is all zeros
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

    # Move tensors to device
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.batch = data.batch.to(device)

    with torch.no_grad():
        logits = model(data)          # shape [1, num_families]
        probs = torch.sigmoid(logits) # same shape
        probs = probs[0].cpu().tolist()

    return probs


def predict_active_rule_families_for_pair(
    model,
    inp_grid,
    out_grid,
    threshold: float = 0.5,
    device: str = "cpu",
) -> List[RuleFamily]:
    """
    Return a list of RuleFamily values whose probability >= threshold.
    """
    probs = predict_rule_family_probs_for_pair(model, inp_grid, out_grid, device=device)
    active: List[RuleFamily] = []
    for i, fam in enumerate(RuleFamily):
        if probs[i] >= threshold:
            active.append(fam)
    return active

