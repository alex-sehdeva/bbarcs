# quick_test_solver.py

from arc_synthetic_dataset_builder import build_synthetic_training_pairs
from arc_gnn_train import train_rule_family_gnn
from arc_solver import solve_arc_task_with_program_synth_and_gnn

if __name__ == "__main__":
    # Train GNN on synthetic data
    model = train_rule_family_gnn(epochs=5)  # fewer epochs just to test the loop

    # Build a tiny synthetic "task"
    pairs = build_synthetic_training_pairs(n_translation=1, n_recolor=0, n_both=0)
    training_pairs = pairs[:1]
    test_grids = [training_pairs[0][0]]  # apply rule to the same input

    preds, prog, summary, fams = solve_arc_task_with_program_synth_and_gnn(
        training_pairs,
        test_grids,
        model,
        device="cpu",
        threshold=0.4,
    )

    print("Active families (GNN):", fams)
    print("Intersected summary:", summary)
    print("Program:", prog)
    print("Predicted grid:", preds[0])

