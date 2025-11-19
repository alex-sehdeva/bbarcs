from arc_graph_core import extract_objects_from_grid
from arc_graph_delta import compute_graph_delta, intersect_rule_summaries
from arc_gnn_inference import predict_active_rule_families_for_pair
from arc_program_synth import find_consistent_program, generate_candidate_programs_guided

def solve_arc_task_with_program_synth_and_gnn(
    training_pairs,
    test_grids,
    model,
    device: str = "cpu",
    threshold: float = 0.5,
):
    """
    training_pairs: list of (input_grid, output_grid)
    test_grids: list of input grids to solve
    model: trained ArcRuleFamilyGNN
    """
    # 1. Symbolic deltas + intersected summary
    graph_deltas = []
    for (inp, out) in training_pairs:
        ig = extract_objects_from_grid(inp)
        og = extract_objects_from_grid(out)
        gd = compute_graph_delta(ig, og)
        graph_deltas.append(gd)

    summaries = [gd.rule_summary for gd in graph_deltas]
    intersected = intersect_rule_summaries(summaries)

    # 2. GNN predictions across training examples (union of active families)
    active_fams_union: set[RuleFamily] = set()
    for (inp, out) in training_pairs:
        fams = predict_active_rule_families_for_pair(
            model,
            inp,
            out,
            threshold=threshold,
            device=device,
        )
        active_fams_union.update(fams)

    active_fams_list = list(active_fams_union)

    # 3. Guided candidate generation
    candidates = generate_candidate_programs_guided(intersected, active_fams_list)

    # 4. Search for a consistent program
    best_prog, all_progs = find_consistent_program(
        training_pairs,
        intersected,
        candidates=candidates,
    )

    if best_prog is None:
        # No program found: return identity for now
        predictions = [g for g in test_grids]
        return predictions, None, intersected, active_fams_list

    # 5. Apply to test grids
    predictions = [best_prog.apply(g) for g in test_grids]
    return predictions, best_prog, intersected, active_fams_list

