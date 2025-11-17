import arc_graph_core

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set

# Assuming these come from your step-1 module:
# from arc_graph_core import ObjectGraph, GridObject, Color


Color = int  # keep consistent with core module
Coord = Tuple[int, int]


@dataclass
class ObjectMatch:
    """
    A pairing between an input object and an output object.
    """
    input_id: int
    output_id: int
    score: float  # smaller = better (cost)


@dataclass
class ObjectDelta:
    """
    Per-object change between input and output.
    """
    input_id: int
    output_id: int

    # Geometric deltas
    translation: Tuple[float, float]  # (d_row, d_col) using centroids
    area_in: int
    area_out: int
    area_ratio: float  # area_out / max(area_in, 1)

    # Color changes
    color_in: Optional[Color]
    color_out: Optional[Color]
    color_same: bool

    # Bounding box change (optional, can be useful later)
    bbox_in: Tuple[int, int, int, int]
    bbox_out: Tuple[int, int, int, int]


@dataclass
class RuleSummary:
    """
    A coarse rule-summary for a single (input, output) example.
    Later we will intersect these across multiple examples.
    """
    # If all matched objects share the same translation, record it
    global_translation: Optional[Tuple[float, float]] = None

    # If all matched objects define a consistent color mapping, record it
    color_mapping: Dict[Color, Color] = field(default_factory=dict)

    # Flags for simple invariants
    preserves_object_count: bool = False
    preserves_areas: bool = False  # all matched area_ratio == 1


@dataclass
class GraphDelta:
    """
    Result of comparing an input object graph to an output object graph.
    """
    matches: List[ObjectMatch]
    object_deltas: List[ObjectDelta]
    unmatched_input_ids: List[int]
    unmatched_output_ids: List[int]
    rule_summary: RuleSummary


# ---------- Matching ----------

def _object_match_cost(o_in: "GridObject", o_out: "GridObject") -> float:
    """
    Heuristic cost between two objects.

    Lower cost means more similar. We combine:
    - area difference
    - centroid distance
    - color mismatch penalty
    """
    # Area term (relative difference)
    area_in = o_in.area
    area_out = o_out.area
    if max(area_in, area_out, 1) > 0:
        area_term = abs(area_in - area_out) / max(area_in, area_out, 1)
    else:
        area_term = 0.0

    # Centroid term (L1 distance normalized by grid size later if needed)
    (r1, c1) = o_in.centroid
    (r2, c2) = o_out.centroid
    centroid_term = abs(r1 - r2) + abs(c1 - c2)

    # Color term
    if o_in.main_color is None or o_out.main_color is None:
        color_term = 0.0
    else:
        color_term = 0.0 if o_in.main_color == o_out.main_color else 1.0

    # Weighted sum (weights are heuristic and can be tuned)
    return 2.0 * area_term + 0.5 * centroid_term + 3.0 * color_term


def match_objects_greedy(
    input_graph: "ObjectGraph",
    output_graph: "ObjectGraph",
    max_cost: float = 9999.0,
) -> Tuple[List[ObjectMatch], List[int], List[int]]:
    """
    Greedy bipartite matching between input and output objects using the cost function.
    - For each input object, we pick the best output object that is not already taken.
    - If the best cost exceeds `max_cost`, we leave that input object unmatched.

    Returns:
        matches: list of ObjectMatch
        unmatched_input_ids
        unmatched_output_ids
    """
    in_objs: Dict[int, "GridObject"] = input_graph.objects
    out_objs: Dict[int, "GridObject"] = output_graph.objects

    # Precompute all pairwise costs
    candidates: List[Tuple[float, int, int]] = []
    for in_id, o_in in in_objs.items():
        for out_id, o_out in out_objs.items():
            cost = _object_match_cost(o_in, o_out)
            candidates.append((cost, in_id, out_id))

    # Sort by cost (best matches first)
    candidates.sort(key=lambda x: x[0])

    matched_in: Set[int] = set()
    matched_out: Set[int] = set()
    matches: List[ObjectMatch] = []

    for cost, in_id, out_id in candidates:
        if cost > max_cost:
            break
        if in_id in matched_in or out_id in matched_out:
            continue
        matches.append(ObjectMatch(input_id=in_id, output_id=out_id, score=cost))
        matched_in.add(in_id)
        matched_out.add(out_id)

    unmatched_input_ids = [i for i in in_objs.keys() if i not in matched_in]
    unmatched_output_ids = [j for j in out_objs.keys() if j not in matched_out]

    return matches, unmatched_input_ids, unmatched_output_ids


# ---------- Δ computation ----------

def _compute_object_delta(
    o_in: "GridObject",
    o_out: "GridObject",
) -> ObjectDelta:
    # Translation using centroids
    r1, c1 = o_in.centroid
    r2, c2 = o_out.centroid
    translation = (r2 - r1, c2 - c1)

    # Areas
    area_in = o_in.area
    area_out = o_out.area
    area_ratio = area_out / max(area_in, 1)

    # Colors
    color_in = o_in.main_color
    color_out = o_out.main_color
    color_same = (color_in == color_out)

    # Bboxes
    bbox_in = o_in.bbox
    bbox_out = o_out.bbox

    return ObjectDelta(
        input_id=o_in.id,
        output_id=o_out.id,
        translation=translation,
        area_in=area_in,
        area_out=area_out,
        area_ratio=area_ratio,
        color_in=color_in,
        color_out=color_out,
        color_same=color_same,
        bbox_in=bbox_in,
        bbox_out=bbox_out,
    )


def _summarize_rules(object_deltas: List[ObjectDelta], matches: List[ObjectMatch]) -> RuleSummary:
    summary = RuleSummary()

    if not matches:
        # No matched objects: we can only say trivial things
        summary.preserves_object_count = (len(object_deltas) == 0)
        return summary

    # --- Global translation: check if all translations are the same (within tolerance) ---
    translations = [od.translation for od in object_deltas]
    base_tr = translations[0]
    # Tolerance because centroids are floats
    tol = 1e-6
    same_translation = all(
        abs(tr[0] - base_tr[0]) < tol and abs(tr[1] - base_tr[1]) < tol
        for tr in translations
    )
    if same_translation:
        summary.global_translation = base_tr

    # --- Color mapping: check if a consistent color_in -> color_out mapping exists ---
    color_map: Dict[Color, Color] = {}
    consistent = True
    for od in object_deltas:
        if od.color_in is None or od.color_out is None:
            continue
        if od.color_in in color_map:
            if color_map[od.color_in] != od.color_out:
                consistent = False
                break
        else:
            color_map[od.color_in] = od.color_out

    if consistent:
        summary.color_mapping = color_map

    # --- Simple invariants ---
    # Object count preserved if no unmatched ids
    # (We don't know the unmatched sets here, so caller should set this field)
    # We can still check area preservation for matched objects:
    preserves_areas = all(abs(od.area_ratio - 1.0) < 1e-6 for od in object_deltas)
    summary.preserves_areas = preserves_areas

    return summary


def compute_graph_delta(
    input_graph: "ObjectGraph",
    output_graph: "ObjectGraph",
    max_match_cost: float = 9999.0,
) -> GraphDelta:
    """
    Main entry point:
    - Matches objects between input and output graphs
    - Computes per-object deltas
    - Builds a simple rule summary

    This operates on a *single* ARC example (one input grid, one output grid).
    Later we will intersect RuleSummary objects across multiple examples.
    """
    matches, unmatched_in, unmatched_out = match_objects_greedy(
        input_graph, output_graph, max_cost=max_match_cost
    )

    # Compute per-object deltas
    object_deltas: List[ObjectDelta] = []
    for m in matches:
        o_in = input_graph.objects[m.input_id]
        o_out = output_graph.objects[m.output_id]
        od = _compute_object_delta(o_in, o_out)
        object_deltas.append(od)

    # Summarize rules for this example
    summary = _summarize_rules(object_deltas, matches)

    # Now we can set object-count preservation flag here
    summary.preserves_object_count = (len(unmatched_in) == 0 and len(unmatched_out) == 0)

    return GraphDelta(
        matches=matches,
        object_deltas=object_deltas,
        unmatched_input_ids=unmatched_in,
        unmatched_output_ids=unmatched_out,
        rule_summary=summary,
    )


@dataclass
class IntersectedRuleSummary:
    """
    Intersection of multiple RuleSummary objects (from multiple examples).

    This is our shared 'rule sketch' that we’ll pass downstream.
    """
    # If all examples agree on a single global translation
    global_translation: Optional[Tuple[float, float]] = None

    # Color mapping that is consistent across all examples
    color_mapping: Dict[Color, Color] = field(default_factory=dict)

    # Invariants that hold across all examples
    preserves_object_count: bool = False
    preserves_areas: bool = False

    # You might extend this later with other invariants
    # e.g. "uses only one object", "always keeps largest object", etc.


def intersect_rule_summaries(summaries: List[RuleSummary]) -> IntersectedRuleSummary:
    """
    Given RuleSummary objects from multiple (input, output) examples,
    compute what is *common* among them.

    This is a conservative intersection: if there is disagreement, we drop the claim.
    """
    if not summaries:
        return IntersectedRuleSummary()

    # --- Global translation ---
    # Only keep it if *all* examples have a non-None translation and they agree (within tolerance).
    translations = [s.global_translation for s in summaries if s.global_translation is not None]
    if len(translations) == len(summaries):
        base_tr = translations[0]
        tol = 1e-6
        same_tr = all(
            abs(tr[0] - base_tr[0]) < tol and abs(tr[1] - base_tr[1]) < tol
            for tr in translations
        )
        global_translation = base_tr if same_tr else None
    else:
        global_translation = None

    # --- Color mapping ---
    # We keep only mappings that:
    # - appear in *every* summary that has a mapping for that color_in
    # - never contradict (i.e. color_in -> two different color_out across examples)
    common_color_map: Dict[Color, Color] = {}

    # Collect all candidate input colors that appear in any summary
    all_input_colors = set()
    for s in summaries:
        all_input_colors.update(s.color_mapping.keys())

    for cin in all_input_colors:
        candidate_out: Optional[Color] = None
        consistent = True

        for s in summaries:
            if cin not in s.color_mapping:
                # If a summary has no opinion about this color, we can either
                # (a) demand it appears in all, or
                # (b) allow partial evidence.
                # For now, we choose (a) to be conservative: require it in all.
                consistent = False
                break

            cout = s.color_mapping[cin]
            if candidate_out is None:
                candidate_out = cout
            elif candidate_out != cout:
                consistent = False
                break

        if consistent and candidate_out is not None:
            common_color_map[cin] = candidate_out

    # --- Invariants (object count & areas) ---
    preserves_object_count = all(s.preserves_object_count for s in summaries)
    preserves_areas = all(s.preserves_areas for s in summaries)

    return IntersectedRuleSummary(
        global_translation=global_translation,
        color_mapping=common_color_map,
        preserves_object_count=preserves_object_count,
        preserves_areas=preserves_areas,
    )

