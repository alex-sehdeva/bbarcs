from arc_synthetic_data import make_keep_largest_example
from arc_synthetic_dataset_builder import build_synthetic_training_pairs
from arc_gnn_train import train_rule_family_gnn
from arc_solver import solve_arc_task_with_program_synth_and_gnn

if __name__ == "__main__":
    # Train GNN with some keep-largest examples included
    model = train_rule_family_gnn(epochs=10)

    # Build one keep-largest example as a "task"
    inp, out = make_keep_largest_example()
    training_pairs = [(inp, out)]
    test_grids = [inp]

    preds, prog, summary, fams = solve_arc_task_with_program_synth_and_gnn(
        training_pairs,
        test_grids,
        model,
        device="cpu",
        threshold=0.4,
    )

    print("Active families (GNN):", fams)
    print("Program:", prog)
    print("Predicted grid:")
    for row in preds[0]:
        print(row)

