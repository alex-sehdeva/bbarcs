# arc_rule_families.py

from enum import Enum, auto
from typing import List, Set


class RuleFamily(Enum):
    TRANSLATION = auto()
    RECOLOR = auto()
    AREA_PRESERVING = auto()
    OBJECT_COUNT_PRESERVING = auto()
    # You can add: MIRROR, SCALE, COPY_DELETE, etc. later

from arc_graph_delta import GraphDelta

def infer_rule_families_from_delta(delta: "GraphDelta") -> Set[RuleFamily]:
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

    return fams

