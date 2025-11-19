# arc_synthetic_data.py

import random
from typing import List, Tuple

Grid = List[List[int]]

def make_empty_grid(rows: int, cols: int, bg: int = 0) -> Grid:
    return [[bg for _ in range(cols)] for _ in range(rows)]


def place_block(grid: Grid, top: int, left: int, h: int, w: int, color: int) -> None:
    for r in range(top, top + h):
        for c in range(left, left + w):
            grid[r][c] = color


def make_translation_example(
    rows: int = 6,
    cols: int = 6,
    bg: int = 0,
    obj_color: int = 2,
) -> Tuple[Grid, Grid]:
    """
    One colored block translated by (dr, dc).
    """
    grid_in = make_empty_grid(rows, cols, bg)
    grid_out = make_empty_grid(rows, cols, bg)

    # Random block size
    h = random.randint(1, 3)
    w = random.randint(1, 3)

    # Random starting position such that it fits
    top_in = random.randint(0, rows - h)
    left_in = random.randint(0, cols - w)

    place_block(grid_in, top_in, left_in, h, w, obj_color)

    # Sample a shift (dr, dc) that stays in-bounds
    max_dr_pos = rows - (top_in + h)
    max_dr_neg = top_in
    max_dc_pos = cols - (left_in + w)
    max_dc_neg = left_in

    dr = random.randint(-max_dr_neg, max_dr_pos)
    dc = random.randint(-max_dc_neg, max_dc_pos)

    place_block(grid_out, top_in + dr, left_in + dc, h, w, obj_color)

    return grid_in, grid_out


def make_recolor_example(
    rows: int = 6,
    cols: int = 6,
    bg: int = 0,
    obj_color_in: int = 2,
    obj_color_out: int = 5,
) -> Tuple[Grid, Grid]:
    """
    One object recolored, same location.
    """
    grid_in = make_empty_grid(rows, cols, bg)
    grid_out = make_empty_grid(rows, cols, bg)

    h = random.randint(1, 3)
    w = random.randint(1, 3)

    top = random.randint(0, rows - h)
    left = random.randint(0, cols - w)

    place_block(grid_in, top, left, h, w, obj_color_in)
    place_block(grid_out, top, left, h, w, obj_color_out)

    return grid_in, grid_out


def make_translation_recolor_example(
    rows: int = 6,
    cols: int = 6,
    bg: int = 0,
    obj_color_in: int = 3,
    obj_color_out: int = 7,
) -> Tuple[Grid, Grid]:
    """
    Block that is both translated and recolored.
    """
    inp, _ = make_recolor_example(rows, cols, bg, obj_color_in, obj_color_out)
    # Re-use the same block position but now shift in output
    # Find the object cells in inp
    coords = [(r, c)
              for r in range(rows)
              for c in range(cols)
              if inp[r][c] == obj_color_in]
    if not coords:
        return make_translation_recolor_example(rows, cols, bg, obj_color_in, obj_color_out)

    r_vals = [r for r, c in coords]
    c_vals = [c for r, c in coords]
    top = min(r_vals)
    left = min(c_vals)
    h = max(r_vals) - top + 1
    w = max(c_vals) - left + 1

    grid_out = make_empty_grid(rows, cols, bg)

    # Sample shift that stays in bounds
    max_dr_pos = rows - (top + h)
    max_dr_neg = top
    max_dc_pos = cols - (left + w)
    max_dc_neg = left

    dr = random.randint(-max_dr_neg, max_dr_pos)
    dc = random.randint(-max_dc_neg, max_dc_pos)

    # Place recolored block at shifted location
    place_block(grid_out, top + dr, left + dc, h, w, obj_color_out)

    return inp, grid_out

