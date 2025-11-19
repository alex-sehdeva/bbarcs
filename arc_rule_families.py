# arc_rule_families.py

from enum import Enum, auto
from typing import List, Set

class RuleFamily(Enum):
    TRANSLATION = auto()
    RECOLOR = auto()
    AREA_PRESERVING = auto()
    OBJECT_COUNT_PRESERVING = auto()
    LARGEST_OBJECT = auto()
    # You can add: MIRROR, SCALE, COPY_DELETE, etc. later

from arc_graph_core import ObjectGraph
from arc_graph_delta import GraphDelta

def infer_rule_families_from_delta(
    delta: GraphDelta,
    input_graph: ObjectGraph,
    output_graph: ObjectGraph,
) -> Set[RuleFamily]:
    """
    Heuristically label this example with coarse rule families,
    based on its symbolic RuleSummary.
    """
    fams: Set[RuleFamily] = set()
    s = delta.rule_summary

    if s.global_translation is not None:
        fams.add(RuleFamily.TRANSLATION)

    if s.color_mapping:
        fams.add(RuleFamily.RECOLOR)

    if s.preserves_areas:
        fams.add(RuleFamily.AREA_PRESERVING)

    if s.preserves_object_count:
        fams.add(RuleFamily.OBJECT_COUNT_PRESERVING)

    if _looks_like_keep_largest(delta, input_graph, output_graph):
        fams.add(RuleFamily.LARGEST_OBJECT)

    return fams

from typing import Set

def _looks_like_keep_largest(delta: GraphDelta, input_graph: ObjectGraph, output_graph: ObjectGraph) -> bool:
    """
    Heuristic:
      - more than one input object
      - exactly one output object
      - output object's area == max area of input objects
    """
    if len(input_graph.objects) <= 1:
        return False
    if len(output_graph.objects) != 1:
        return False

    in_areas = [o.area for o in input_graph.objects.values()]
    max_in_area = max(in_areas)

    out_obj = next(iter(output_graph.objects.values()))
    return out_obj.area == max_in_area

