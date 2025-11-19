
# arc_program_synth.py

from dataclasses import dataclass, field
from typing import List, Callable, Tuple

from arc_rule_families import RuleFamily
from arc_rules import Rule, TranslateAllObjects, RecolorObjects, CompositeRule
from arc_graph_delta import IntersectedRuleSummary
from arc_graph_core import extract_objects_from_grid

Grid = List[List[int]]


@dataclass
class Program:
    """
    A simple program is just a sequence of Rule objects applied in order.
    """
    rules: List[Rule] = field(default_factory=list)

    def apply(self, grid: Grid) -> Grid:
        current = grid
        for r in self.rules:
            current = r.apply(current)
        return current

def generate_candidate_programs(summary: IntersectedRuleSummary) -> List[Program]:
    """
    Generate a small set of candidate programs based on the intersected summary.
    This is *not* exhaustive; it’s a guided search.
    """
    from arc_rules import TranslateAllObjects, RecolorObjects, CompositeRule

    candidates: List[Program] = []

    # Potential primitive rules
    primitive_rules: List[Rule] = []

    # 1. Translation primitive
    if summary.global_translation is not None:
        dr, dc = summary.global_translation
        dr_int = int(round(dr))
        dc_int = int(round(dc))
        primitive_rules.append(TranslateAllObjects(dr_int, dc_int))

    # 2. Recolor primitive
    if summary.color_mapping:
        primitive_rules.append(RecolorObjects(summary.color_mapping))

    # Fallback: no info → no primitives
    if not primitive_rules:
        return []

    # Single-rule programs
    for r in primitive_rules:
        candidates.append(Program(rules=[r]))

    # Two-rule programs (all ordered pairs)
    for r1 in primitive_rules:
        for r2 in primitive_rules:
            if r1 is r2:
                continue
            candidates.append(Program(rules=[r1, r2]))

    return candidates


def grids_equal(g1: Grid, g2: Grid) -> bool:
    if len(g1) != len(g2):
        return False
    if len(g1) == 0:
        return len(g2) == 0
    if len(g1[0]) != len(g2[0]):
        return False
    for r in range(len(g1)):
        for c in range(len(g1[0])):
            if g1[r][c] != g2[r][c]:
                return False
    return True

def find_consistent_program(
    training_pairs: list[tuple[list[list[int]], list[list[int]]]],
    summary: IntersectedRuleSummary,
    candidates: list[Program] = None,
) -> tuple[Program | None, list[Program]]:
    """
    Search for a small program that is consistent with all training examples.
    If 'candidates' is provided, search over that; otherwise generate from summary.
    """
    if candidates is None:
        candidates = generate_candidate_programs(summary)

    consistent: List[Program] = []

    for prog in candidates:
        ok = True
        for inp, out in training_pairs:
            pred = prog.apply(inp)
            if not grids_equal(pred, out):
                ok = False
                break
        if ok:
            consistent.append(prog)

    if not consistent:
        return None, []

    best = min(consistent, key=lambda p: len(p.rules))
    return best, consistent

def solve_arc_task_with_program_synth(training_pairs, test_grids):
    """
    training_pairs: list of (input_grid, output_grid)
    test_grids: list of input grids to solve.

    Returns:
      predictions: list of predicted output grids
      best_program: the Program object found
      intersected_summary: IntersectedRuleSummary used
    """
    from arc_graph_core import extract_objects_from_grid
    from arc_graph_delta import compute_graph_delta, intersect_rule_summaries

    # 1. Symbolic Δs + summaries
    graph_deltas = []
    for (inp, out) in training_pairs:
        ig = extract_objects_from_grid(inp)
        og = extract_objects_from_grid(out)
        gd = compute_graph_delta(ig, og)
        graph_deltas.append(gd)

    # 2. Intersect rule summaries across examples
    summaries = [gd.rule_summary for gd in graph_deltas]
    intersected = intersect_rule_summaries(summaries)

    # 3. Search for a small program consistent with all training pairs
    best_prog, all_progs = find_consistent_program(training_pairs, intersected)

    if best_prog is None:
        # Couldn’t find any program with this tiny DSL
        predictions = [grid for grid in test_grids]  # identity fallback
        return predictions, None, intersected

    # 4. Apply program to test grids
    predictions = [best_prog.apply(g) for g in test_grids]
    return predictions, best_prog, intersected


def generate_candidate_programs_guided(
    summary: IntersectedRuleSummary,
    active_families: list[RuleFamily],
) -> List[Program]:
    """
    Like generate_candidate_programs, but only uses primitives that are:
      - supported by the symbolic summary
      - AND predicted active by the GNN (active_families).
    """
    candidates: List[Program] = []
    primitive_rules: List[Rule] = []

    active_fams_set = set(active_families)

    # Translation primitive: require both a global_translation and GNN believing TRANSLATION
    if summary.global_translation is not None and RuleFamily.TRANSLATION in active_fams_set:
        dr, dc = summary.global_translation
        dr_int = int(round(dr))
        dc_int = int(round(dc))
        primitive_rules.append(TranslateAllObjects(dr_int, dc_int))

    # Recolor primitive: require color_mapping and GNN believing RECOLOR
    if summary.color_mapping and RuleFamily.RECOLOR in active_fams_set:
        primitive_rules.append(RecolorObjects(summary.color_mapping))

    if not primitive_rules:
        # If nothing survives, fall back to unguided generator (symbolic-only)
        return generate_candidate_programs(summary)

    # Single-rule programs
    for r in primitive_rules:
        candidates.append(Program(rules=[r]))

    # Two-rule programs (ordered pairs)
    for r1 in primitive_rules:
        for r2 in primitive_rules:
            if r1 is r2:
                continue
            candidates.append(Program(rules=[r1, r2]))

    return candidates
