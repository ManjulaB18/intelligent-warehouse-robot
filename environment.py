"""Warehouse environment (a simple grid world) for the Q-Learning robot."""
import numpy as np

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Rewards (each one has a purpose, explained in README / report)
R_STEP = -1            # small cost per move -> encourages short routes
R_OBSTACLE = -10       # hit an obstacle -> robot stays in place
R_WALL = -10           # tried to leave the grid -> robot stays in place
R_PICKUP = 20          # collected the package
R_DELIVER = 100        # delivered the package (episode ends)
R_EMPTY_STATION = -5   # reached delivery station without package

DEFAULT_OBSTACLES = [
    (1, 3), (2, 3), (3, 3), (4, 3), (4, 4), (4, 5),
    (6, 1), (6, 2), (6, 3), (6, 6), (7, 6), (8, 6),
    (2, 7), (3, 7), (3, 8), (5, 8), (8, 2), (8, 3),
]


class WarehouseEnv:
    """State = (row, col, has_package). Pickup/delivery happen automatically
    when the robot steps on the package / delivery cell."""

    def __init__(self, size=10, obstacles=None, start=(0, 0),
                 package=(7, 1), station=(9, 9), max_steps=200):
        self.size = size
        self.obstacles = set(DEFAULT_OBSTACLES if obstacles is None else obstacles)
        self.start, self.package, self.station = start, package, station
        self.max_steps = max_steps
        self.n_actions = 4
        self.n_states = size * size * 2
        for p in (start, package, station):
            assert p not in self.obstacles, f"{p} is on an obstacle"
        self.reset()

    # --- state <-> index helpers -------------------------------------
    def encode(self, row, col, has_pkg):
        return (row * self.size + col) * 2 + int(has_pkg)

    def reset(self):
        self.pos = self.start
        self.has_package = False
        self.delivered = False
        self.steps = 0
        return self.encode(*self.pos, self.has_package)

    def step(self, action):
        """Returns next_state, reward, done, info"""
        self.steps += 1
        dr, dc = MOVES[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        reward = R_STEP
        info = {"obstacle_hit": False, "wall_hit": False}

        if not (0 <= r < self.size and 0 <= c < self.size):
            reward += R_WALL; info["wall_hit"] = True      # stay in place
        elif (r, c) in self.obstacles:
            reward += R_OBSTACLE; info["obstacle_hit"] = True  # stay in place
        else:
            self.pos = (r, c)
            if self.pos == self.package and not self.has_package and not self.delivered:
                self.has_package = True
                reward += R_PICKUP
            elif self.pos == self.station:
                if self.has_package:
                    self.has_package = False
                    self.delivered = True
                    reward += R_DELIVER
                else:
                    reward += R_EMPTY_STATION

        done = self.delivered or self.steps >= self.max_steps
        info["delivered"] = self.delivered
        return self.encode(*self.pos, self.has_package), reward, done, info

    def grid(self):
        """Integer grid for drawing: 0 empty, 1 obstacle"""
        g = np.zeros((self.size, self.size), dtype=int)
        for (r, c) in self.obstacles:
            g[r, c] = 1
        return g
