from arc_gnn_dataset import *
from arc_gnn_model import *
from arc_rule_families import *
from arc_graph_core import * #import extract_objects_from_grid
from arc_graph_delta import * # import compute_graph_delta, intersect_rule_summaries
from arc_rules import * # import propose_rules_from_intersection

def test_extract_objects_simple():
    # grid:
    # 0 0 0 0
    # 0 2 2 0
    # 0 0 0 0
    grid = [
        [0,0,0,0],
        [0,2,2,0],
        [0,0,0,0],
    ]

    og = extract_objects_from_grid(grid)
    assert len(og.objects) == 1, "Should detect one object"

    obj = list(og.objects.values())[0]
    assert obj.area == 2
    assert obj.main_color == 2


def test_delta_translation():
    # Input:
    # 0 1 0
    # Output:
    # 0 0 1

    inp = [
        [0,1,0],
        [0,0,0],
    ]
    out = [
        [0,0,0],
        [0,0,1],
    ]

    ig = extract_objects_from_grid(inp)
    og = extract_objects_from_grid(out)

    delta = compute_graph_delta(ig, og)
    summary = delta.rule_summary

    assert summary.global_translation is not None
    # The object moved from (0,1) to (1,2) = translation (1,+1)
    dr, dc = summary.global_translation
    assert round(dr) == 1 and round(dc) == 1


def test_delta_color_mapping():
    inp = [
        [0,2],
        [0,0],
    ]
    out = [
        [0,5],
        [0,0],
    ]

    ig = extract_objects_from_grid(inp)
    og = extract_objects_from_grid(out)
    delta = compute_graph_delta(ig, og)

    cm = delta.rule_summary.color_mapping
    assert cm.get(2) == 5


def test_intersection_translation():
    # Two training examples that both move objects right by +2
    # Example grids simplified
    inp1 = [[0,3,0]]
    out1 = [[0,0,3]]

    inp2 = [[3,0,0]]
    out2 = [[0,3,0]]

    deltas = []
    for inp, out in [(inp1,out1),(inp2,out2)]:
        ig = extract_objects_from_grid(inp)
        og = extract_objects_from_grid(out)
        deltas.append(compute_graph_delta(ig, og))

    summaries = [d.rule_summary for d in deltas]
    inter = intersect_rule_summaries(summaries)

    assert inter.global_translation is not None
    dr, dc = inter.global_translation
    assert round(dc) == 1 or round(dc) == 2  # depends on geometry but should be consistent


def test_rule_execute_translation():
    inp = [
        [0,1,0],
        [0,0,0],
    ]
    rule = TranslateAllObjects(d_row=1, d_col=1)
    out = rule.apply(inp)

    assert out[1][2] == 1  # moved bottom-right


def test_rule_execute_recolor():
    inp = [
        [0,2],
        [0,0]
    ]
    rule = RecolorObjects({2: 9})
    out = rule.apply(inp)
    assert out[0][1] == 9


def test_gnn_dataset_shapes():
    # trivial example: one object recolored + translated
    inp = [[0,4]]
    out = [[4,0]]  # recolor + move
    training_pairs = [(inp, out)]

    ds = ArcRuleFamilyDataset(training_pairs)
    assert len(ds) == 1

    data = ds[0]
    assert data.x.shape[1] == 5   # features: area, r, c, color, is_output
    assert data.y.shape[0] == len(RuleFamily)


