from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter, deque

Color = int
Coord = Tuple[int, int]  # (row, col)


@dataclass
class GridObject:
    """
    One 'object' in an ARC grid: a set of cells plus derived geometry features.
    """
    id: int
    cells: Set[Coord]

    # Basic features (will be computed)
    main_color: Optional[Color] = None
    area: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (min_row, min_col, max_row, max_col)
    centroid: Tuple[float, float] = (0.0, 0.0)

    def compute_basic_features(self, grid: List[List[Color]]) -> None:
        """
        Fill in area, bbox, centroid, main_color based on cells and grid.
        """
        if not self.cells:
            self.area = 0
            self.bbox = (0, 0, 0, 0)
            self.centroid = (0.0, 0.0)
            self.main_color = None
            return

        rows = [r for (r, c) in self.cells]
        cols = [c for (r, c) in self.cells]

        self.area = len(self.cells)
        self.bbox = (min(rows), min(cols), max(rows), max(cols))

        # Centroid in grid coordinates (row, col)
        self.centroid = (
            sum(rows) / len(rows),
            sum(cols) / len(cols),
        )

        # Dominant color inside the object
        color_counts = Counter(grid[r][c] for (r, c) in self.cells)
        self.main_color = color_counts.most_common(1)[0][0]


@dataclass
class ObjectGraph:
    """
    Graph of objects extracted from a grid.
    For now, edges are optional; we mainly care about the object set + features.
    """
    objects: Dict[int, GridObject] = field(default_factory=dict)
    # Later: relation edges like adjacency, containment, symmetry, etc.
    # edges: List[ObjectEdge] = field(default_factory=list)

    grid_shape: Tuple[int, int] = (0, 0)  # (n_rows, n_cols)

    def compute_features(self, grid: List[List[Color]]) -> None:
        """
        Compute per-object features for all objects.
        """
        for obj in self.objects.values():
            obj.compute_basic_features(grid)


def infer_background_color(grid: List[List[Color]]) -> Color:
    """
    Heuristic: background color is the most frequent color in the grid.
    """
    counts = Counter()
    for row in grid:
        counts.update(row)
    return counts.most_common(1)[0][0]


def extract_objects_from_grid(
    grid: List[List[Color]],
    connectivity: int = 4,
) -> ObjectGraph:
    """
    Very simple object extraction:
    - Infer background color
    - Find connected components of non-background cells
    - Each component becomes a GridObject

    NOTE: This is intentionally simple and can be refined later
    (e.g., multi-color objects, shape-based grouping, etc.).
    """
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 0
    bg = infer_background_color(grid)

    visited = [[False] * n_cols for _ in range(n_rows)]
    objects: Dict[int, GridObject] = {}
    next_id = 1

    # Neighbor offsets
    if connectivity == 4:
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    elif connectivity == 8:
        nbrs = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]
    else:
        raise ValueError("connectivity must be 4 or 8")

    for r in range(n_rows):
        for c in range(n_cols):
            if visited[r][c]:
                continue
            if grid[r][c] == bg:
                continue  # skip background

            # BFS / flood fill for this object
            comp_cells: Set[Coord] = set()
            queue = deque([(r, c)])
            visited[r][c] = True

            while queue:
                cr, cc = queue.popleft()
                comp_cells.add((cr, cc))

                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                        continue
                    if visited[nr][nc]:
                        continue
                    if grid[nr][nc] == bg:
                        continue

                    # For now, any non-background cell belongs to some object.
                    visited[nr][nc] = True
                    queue.append((nr, nc))

            # Create object if we found any cells
            if comp_cells:
                obj = GridObject(id=next_id, cells=comp_cells)
                objects[next_id] = obj
                next_id += 1

    og = ObjectGraph(objects=objects, grid_shape=(n_rows, n_cols))
    og.compute_features(grid)
    return og

