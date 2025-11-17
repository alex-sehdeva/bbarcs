from arc_graph_core import extract_objects_from_grid
from arc_graph_delta import compute_graph_delta, intersect_rule_summaries
from arc_rules import propose_rules_from_intersection

def analyze_arc_task(training_pairs):
    """
    training_pairs: list of (input_grid, output_grid)
    where each grid is List[List[int]].
    """
    graph_deltas = []

    for (inp_grid, out_grid) in training_pairs:
        inp_g = extract_objects_from_grid(inp_grid)
        out_g = extract_objects_from_grid(out_grid)

        gd = compute_graph_delta(inp_g, out_g)
        graph_deltas.append(gd)

    # Collect the per-example rule summaries
    summaries = [gd.rule_summary for gd in graph_deltas]

    # Intersect them to get a shared rule sketch
    intersected = intersect_rule_summaries(summaries)

    return graph_deltas, intersected

def solve_arc_task_with_simple_rules(training_pairs, test_grid):
    """
    training_pairs: list of (input_grid, output_grid)
    test_grid: a single grid (List[List[int]])
    """

    # Step 1–3: compute per-example deltas and intersect
    graph_deltas = []
    for (inp_grid, out_grid) in training_pairs:
        inp_g = extract_objects_from_grid(inp_grid)
        out_g = extract_objects_from_grid(out_grid)

        gd = compute_graph_delta(inp_g, out_g)
        graph_deltas.append(gd)

    # Collect the per-example rule summaries
    summaries = [gd.rule_summary for gd in graph_deltas]

    # Intersect them to get a shared rule sketch
    intersected = intersect_rule_summaries(summaries)

    # Step 4: build an executable rule from the intersected summary
    rule = propose_rules_from_intersection(intersected)

    # Apply to test grid to get prediction
    predicted_output = rule.apply(test_grid)
    return predicted_output, rule, intersected

