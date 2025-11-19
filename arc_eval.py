# arc_eval.py

from typing import List, Tuple
from arc_synthetic_dataset_builder import build_synthetic_training_pairs
from arc_synthetic_data import (
    make_translation_example,
    make_recolor_example,
    make_translation_recolor_example,
    make_keep_largest_example,
)
from arc_gnn_train import train_rule_family_gnn
from arc_solver import solve_arc_task_with_program_synth_and_gnn
from arc_rule_families import RuleFamily


Grid = List[List[int]]


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


def eval_on_synthetic(
    n_per_family: int = 20,
    epochs: int = 10,
):
    # 1. Train GNN
    training_pairs = build_synthetic_training_pairs(
        n_translation=100,
        n_recolor=100,
        n_both=100,
        n_keep_largest=100,
    )
    model = train_rule_family_gnn(epochs=epochs)

    # 2. Build eval sets per rule family
    families = {
        "translation": make_translation_example,
        "recolor": make_recolor_example,
        "both": make_translation_recolor_example,
        "keep_largest": make_keep_largest_example,
    }

    results = {name: {"total": 0, "correct": 0} for name in families.keys()}

    for name, gen_fn in families.items():
        for _ in range(n_per_family):
            inp, out = gen_fn()
            training_pairs = [(inp, out)]
            test_grids = [inp]

            preds, prog, summary, fams = solve_arc_task_with_program_synth_and_gnn(
                training_pairs,
                test_grids,
                model,
                device="cpu",
                threshold=0.4,
            )

            results[name]["total"] += 1
            if grids_equal(preds[0], out):
                results[name]["correct"] += 1

    print("=== Synthetic Evaluation ===")
    for name, stats in results.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"{name:15s}  acc = {acc:.2f}  ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    eval_on_synthetic(n_per_family=20, epochs=10)

