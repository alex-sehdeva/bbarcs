from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from arc_graph_core import extract_objects_from_grid, ObjectGraph, GridObject, infer_background_color
from arc_graph_delta import IntersectedRuleSummary

Color = int
Coord = Tuple[int, int]
Grid = List[List[Color]]


class Rule:
    """
    Base class for all rules.
    """
    def apply(self, grid: Grid) -> Grid:
        raise NotImplementedError

@dataclass
class TranslateAllObjects(Rule):
    """
    Translate every non-background object by (d_row, d_col).
    """
    d_row: int
    d_col: int

    def apply(self, grid: Grid) -> Grid:
        from arc_graph_core import extract_objects_from_grid, infer_background_color

        n_rows = len(grid)
        n_cols = len(grid[0]) if n_rows > 0 else 0
        bg = infer_background_color(grid)

        og: ObjectGraph = extract_objects_from_grid(grid)
        # Start with a background-filled grid
        new_grid: Grid = [[bg for _ in range(n_cols)] for _ in range(n_rows)]

        for obj in og.objects.values():
            for (r, c) in obj.cells:
                nr = r + self.d_row
                nc = c + self.d_col
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    new_grid[nr][nc] = grid[r][c]
                # else: we silently drop anything that would go out of bounds
                # (you can choose to handle this differently later)

        return new_grid

@dataclass
class RecolorObjects(Rule):
    """
    Apply a global color mapping to all cells:
    if color in mapping, replace it; else leave as is.
    """
    color_mapping: Dict[Color, Color] = field(default_factory=dict)

    def apply(self, grid: Grid) -> Grid:
        n_rows = len(grid)
        n_cols = len(grid[0]) if n_rows > 0 else 0

        new_grid: Grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
        for r in range(n_rows):
            for c in range(n_cols):
                col = grid[r][c]
                new_grid[r][c] = self.color_mapping.get(col, col)
        return new_grid

@dataclass
class CompositeRule(Rule):
    """
    Apply a sequence of rules in order.
    """
    rules: List[Rule] = field(default_factory=list)

    def apply(self, grid: Grid) -> Grid:
        current = grid
        for rule in self.rules:
            current = rule.apply(current)
        return current


def propose_rules_from_intersection(
    summary: IntersectedRuleSummary,
) -> Rule:
    """
    Turn an IntersectedRuleSummary into a concrete Rule (or CompositeRule).

    This is a heuristic: we only handle a couple of common cases for now:
    - global translation
    - global recoloring

    You can extend this as you add more invariants and rule types.
    """
    rules: List[Rule] = []

    # 1. Translation rule if present and "nice" (close to integer)
    if summary.global_translation is not None:
        d_row, d_col = summary.global_translation
        # Round to nearest int; in ARC these should typically be integers.
        d_row_int = int(round(d_row))
        d_col_int = int(round(d_col))
        if abs(d_row - d_row_int) < 1e-6 and abs(d_col - d_col_int) < 1e-6:
            rules.append(TranslateAllObjects(d_row=d_row_int, d_col=d_col_int))

    # 2. Recolor rule if we have a non-empty color mapping
    if summary.color_mapping:
        rules.append(RecolorObjects(color_mapping=summary.color_mapping))

    if not rules:
        # Fallback: identity rule (no change)
        class IdentityRule(Rule):
            def apply(self, grid: Grid) -> Grid:
                return [row[:] for row in grid]
        return IdentityRule()

    if len(rules) == 1:
        return rules[0]

    return CompositeRule(rules=rules)


@dataclass
class KeepLargestObject(Rule):
    """
    Remove all objects except the one with the largest area.
    Ties: keep one arbitrarily (first encountered).
    """
    def apply(self, grid: Grid) -> Grid:
        og = extract_objects_from_grid(grid)
        if not og.objects:
            return [row[:] for row in grid]

        # find largest-area object
        largest = max(og.objects.values(), key=lambda o: o.area)
        bg = infer_background_color(grid)

        n_rows = len(grid)
        n_cols = len(grid[0]) if n_rows > 0 else 0
        new_grid: Grid = [[bg for _ in range(n_cols)] for _ in range(n_rows)]

        for (r, c) in largest.cells:
            new_grid[r][c] = grid[r][c]

        return new_grid

