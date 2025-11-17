# arc_synthetic_dataset_builder.py

from typing import List, Tuple
from arc_synthetic_data import (
    make_translation_example,
    make_recolor_example,
    make_translation_recolor_example,
)

Grid = List[List[int]]

def build_synthetic_training_pairs(
    n_translation: int = 50,
    n_recolor: int = 50,
    n_both: int = 50,
) -> List[Tuple[Grid, Grid]]:
    pairs: List[Tuple[Grid, Grid]] = []

    for _ in range(n_translation):
        pairs.append(make_translation_example())

    for _ in range(n_recolor):
        pairs.append(make_recolor_example())

    for _ in range(n_both):
        pairs.append(make_translation_recolor_example())

    return pairs

