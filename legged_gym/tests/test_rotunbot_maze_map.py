"""Unit tests for the Isaac-Gym-independent maze layout module."""

import unittest

import numpy as np

from legged_gym.maps import (
    FREE,
    WALL,
    build_maze,
    cell_centers_to_world,
    reachable_free_cells,
    wall_cells,
)


class RotunbotMazeMapTests(unittest.TestCase):
    def test_layout_is_deterministic_and_bounded(self):
        first = build_maze((15, 15), seed=7, center_clearance_radius=2)
        second = build_maze((15, 15), seed=7, center_clearance_radius=2)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[0, :] == WALL))
        self.assertTrue(np.all(first[-1, :] == WALL))
        self.assertTrue(np.all(first[:, 0] == WALL))
        self.assertTrue(np.all(first[:, -1] == WALL))

    def test_spawn_area_and_reachable_goals_are_free(self):
        maze = build_maze((15, 15), seed=0, center_clearance_radius=2)
        center = (maze.shape[0] // 2, maze.shape[1] // 2)
        self.assertTrue(
            np.all(
                maze[
                    center[0] - 2 : center[0] + 3,
                    center[1] - 2 : center[1] + 3,
                ]
                == FREE
            )
        )
        reachable = reachable_free_cells(maze, center)
        self.assertGreater(len(reachable), 25)
        self.assertTrue(np.all(maze[reachable[:, 0], reachable[:, 1]] == FREE))

    def test_geometry_conversion_matches_grid(self):
        maze = build_maze((15, 15), seed=0, center_clearance_radius=2)
        walls = wall_cells(maze)
        world = cell_centers_to_world(walls, maze.shape, cell_size=2.0)
        self.assertEqual(world.shape, walls.shape)
        self.assertTrue(np.all(np.abs(world) <= 14.0))

    def test_invalid_even_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            build_maze((14, 15))


if __name__ == "__main__":
    unittest.main()
