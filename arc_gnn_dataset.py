# arc_gnn_dataset.py

from typing import Tuple, List, Dict, Set
import torch
from torch_geometric.data import Data

from arc_graph_core import ObjectGraph, GridObject


def _build_node_features(
    objs: Dict[int, GridObject],
    is_output_flag: int,
) -> Tuple[List[List[float]], List[int]]:
    """
    Build node feature vectors and a parallel list of node ids (for debugging).
    Features right now:
      [area, centroid_row, centroid_col, main_color, is_output_flag]
    """
    x: List[List[float]] = []
    node_ids: List[int] = []

    for obj_id, obj in objs.items():
        area = float(obj.area)
        r, c = obj.centroid
        color = float(obj.main_color if obj.main_color is not None else -1)
        is_out = float(is_output_flag)

        feat = [area, float(r), float(c), color, is_out]
        x.append(feat)
        node_ids.append(obj_id)

    return x, node_ids


def _fully_connect_indices(start: int, count: int) -> List[Tuple[int, int]]:
    """
    Fully connect nodes [start, start+count) undirected (we’ll add both directions).
    """
    edges = []
    for i in range(start, start + count):
        for j in range(start, start + count):
            if i == j:
                continue
            edges.append((i, j))
    return edges


def graphs_to_pyg_data(
    input_graph: ObjectGraph,
    output_graph: ObjectGraph,
    label_indices: List[int],
    num_families: int,
) -> Data:
    """
    Turn (input_graph, output_graph) into a torch_geometric Data object.

    Args:
      label_indices: indices of RuleFamily that apply to this example
                    (for multi-label classification).
      num_families: total number of RuleFamily values

    Returns:
      Data with fields:
        - x: [num_nodes, num_features]
        - edge_index: [2, num_edges]
        - y: [num_families] multi-hot vector
    """
    # Build nodes for input objects
    x_in, in_ids = _build_node_features(input_graph.objects, is_output_flag=0)
    # Build nodes for output objects
    x_out, out_ids = _build_node_features(output_graph.objects, is_output_flag=1)

    x_all = x_in + x_out
    x = torch.tensor(x_all, dtype=torch.float32)

    num_in = len(in_ids)
    num_out = len(out_ids)

    # Simple edge pattern: fully connect input objects among themselves,
    # and fully connect output objects among themselves.
    # (We can add cross-graph edges later if useful.)
    edges: List[Tuple[int, int]] = []
    edges += _fully_connect_indices(start=0, count=num_in)
    edges += _fully_connect_indices(start=num_in, count=num_out)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Multi-hot label vector
    y = torch.zeros(num_families, dtype=torch.float32)
    for idx in label_indices:
        y[idx] = 1.0

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


from typing import List, Tuple, Callable
import torch
from torch_geometric.data import InMemoryDataset

from arc_graph_core import extract_objects_from_grid
from arc_graph_delta import compute_graph_delta
from arc_rule_families import RuleFamily, infer_rule_families_from_delta
from arc_gnn_dataset import graphs_to_pyg_data


class ArcRuleFamilyDataset(InMemoryDataset):
    def __init__(
        self,
        training_pairs: List[Tuple[List[List[int]], List[List[int]]]],
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        """
        training_pairs: list of (input_grid, output_grid)
        where each grid is List[List[int]] of colors.

        We immediately build all Data objects in memory.
        """
        self.training_pairs = training_pairs
        super().__init__(None, transform, pre_transform)

        data_list = self._build_data_list()
        self.data, self.slices = self.collate(data_list)

    def _build_data_list(self) -> List[Data]:
        from arc_graph_core import extract_objects_from_grid
        from arc_graph_delta import compute_graph_delta
        from arc_rule_families import RuleFamily, infer_rule_families_from_delta

        num_families = len(RuleFamily)
        data_list: List[Data] = []

        for (inp_grid, out_grid) in self.training_pairs:
            inp_g = extract_objects_from_grid(inp_grid)
            out_g = extract_objects_from_grid(out_grid)

            gd = compute_graph_delta(inp_g, out_g)
            fams: Set[RuleFamily] = infer_rule_families_from_delta(gd, inp_g, out_g)
            if not fams:
                # Optionally skip examples where we can't infer any family yet
                continue

            # Map RuleFamily enums to indices
            label_indices = [f.value - 1 for f in fams]  # because auto() starts at 1

            data = graphs_to_pyg_data(
                input_graph=inp_g,
                output_graph=out_g,
                label_indices=label_indices,
                num_families=num_families,
            )
            data_list.append(data)

        return data_list

