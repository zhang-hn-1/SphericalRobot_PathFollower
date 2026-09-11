"""Deterministic occupancy-grid maze generation for Rotunbot tasks.

The map is represented by a two-dimensional uint8 array.  ``0`` is a free
cell and ``1`` is a solid wall cell.  Keeping this module independent of
Isaac Gym makes the layout logic fast to test and reusable by visualizers.
"""

from collections import deque

import numpy as np


FREE = np.uint8(0)
WALL = np.uint8(1)


def _validate_grid_size(grid_size):
    if len(grid_size) != 2:
        raise ValueError("grid_size must contain exactly two dimensions")

    width, height = (int(grid_size[0]), int(grid_size[1]))
    if width < 5 or height < 5:
        raise ValueError("maze dimensions must both be at least 5")
    if width % 2 == 0 or height % 2 == 0:
        raise ValueError("maze dimensions must be odd so boundary walls remain closed")
    return width, height


def _generate_depth_first_maze(grid_size, seed):
    width, height = _validate_grid_size(grid_size)
    rng = np.random.default_rng(int(seed))
    maze = np.full((width, height), WALL, dtype=np.uint8)
    maze[1, 1] = FREE
    stack = [(1, 1)]
    directions = ((-2, 0), (2, 0), (0, -2), (0, 2))

    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                1 <= nx < width - 1
                and 1 <= ny < height - 1
                and maze[nx, ny] == WALL
            ):
                neighbors.append((nx, ny))

        if not neighbors:
            stack.pop()
            continue

        nx, ny = neighbors[int(rng.integers(len(neighbors)))]
        maze[(x + nx) // 2, (y + ny) // 2] = FREE
        maze[nx, ny] = FREE
        stack.append((nx, ny))

    return maze


def _fill_corner_gaps(maze):
    """Fill diagonal pinholes that a spherical shell could slip through."""
    fixed = maze.copy()
    for x in range(1, maze.shape[0] - 1):
        for y in range(1, maze.shape[1] - 1):
            if maze[x, y] != FREE:
                continue
            if maze[x - 1, y] == WALL and maze[x, y - 1] == WALL:
                if maze[x - 1, y - 1] == FREE:
                    fixed[x - 1, y - 1] = WALL
            if maze[x + 1, y] == WALL and maze[x, y - 1] == WALL:
                if maze[x + 1, y - 1] == FREE:
                    fixed[x + 1, y - 1] = WALL
            if maze[x - 1, y] == WALL and maze[x, y + 1] == WALL:
                if maze[x - 1, y + 1] == FREE:
                    fixed[x - 1, y + 1] = WALL
            if maze[x + 1, y] == WALL and maze[x, y + 1] == WALL:
                if maze[x + 1, y + 1] == FREE:
                    fixed[x + 1, y + 1] = WALL
    return fixed


def _clear_center(maze, radius):
    radius = int(radius)
    if radius < 0:
        raise ValueError("center_clearance_radius must be non-negative")

    cleared = maze.copy()
    center_x, center_y = cleared.shape[0] // 2, cleared.shape[1] // 2
    x_min, x_max = center_x - radius, center_x + radius + 1
    y_min, y_max = center_y - radius, center_y + radius + 1
    cleared[x_min:x_max, y_min:y_max] = FREE
    return cleared


def build_maze(grid_size=(15, 15), seed=0, center_clearance_radius=2):
    """Build one reproducible maze with a collision-free spawn area."""
    maze = _generate_depth_first_maze(grid_size, seed)
    maze = _fill_corner_gaps(maze)
    maze = _clear_center(maze, center_clearance_radius)
    return maze


def wall_cells(maze):
    """Return integer ``[x, y]`` indices for every solid wall cell."""
    return np.argwhere(np.asarray(maze) == WALL)


def reachable_free_cells(maze, start=None):
    """Return free cells connected to ``start`` by four-neighbor motion."""
    maze = np.asarray(maze)
    if maze.ndim != 2:
        raise ValueError("maze must be a two-dimensional array")

    if start is None:
        start = (maze.shape[0] // 2, maze.shape[1] // 2)
    start = (int(start[0]), int(start[1]))
    if not (0 <= start[0] < maze.shape[0] and 0 <= start[1] < maze.shape[1]):
        raise ValueError("start cell lies outside the maze")
    if maze[start] != FREE:
        raise ValueError("start cell must be free")

    queue = deque([start])
    visited = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (x + dx, y + dy)
            if (
                0 <= neighbor[0] < maze.shape[0]
                and 0 <= neighbor[1] < maze.shape[1]
                and maze[neighbor] == FREE
                and neighbor not in visited
            ):
                visited.add(neighbor)
                queue.append(neighbor)

    return np.asarray(sorted(visited), dtype=np.int64)


def cell_centers_to_world(cells, maze_shape, cell_size):
    """Convert occupancy-grid cell indices to map-local XY coordinates."""
    cells = np.asarray(cells, dtype=np.float64)
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("cells must have shape [N, 2]")
    cell_size = float(cell_size)
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive")

    maze_shape = np.asarray(maze_shape, dtype=np.float64)
    return (cells - maze_shape / 2.0 + 0.5) * cell_size
