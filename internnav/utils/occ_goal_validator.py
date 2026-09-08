"""Occupancy-derived feasibility checks for real-world pixel goals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


UNKNOWN = 0
FREE = 1
OCCUPIED = 2
INFLATED = 3


@dataclass
class GoalFeasibility:
    status: str
    score: float
    reason: str
    goal_xy: Optional[Tuple[float, float]]
    grid_xy: Optional[Tuple[int, int]]
    cell_value: Optional[int]
    reachable: Optional[bool]
    distance_to_blocking_m: Optional[float]

    def to_dict(self):
        return {
            "status": self.status,
            "score": self.score,
            "reason": self.reason,
            "goal_xy": list(self.goal_xy) if self.goal_xy is not None else None,
            "grid_xy": list(self.grid_xy) if self.grid_xy is not None else None,
            "cell_value": self.cell_value,
            "reachable": self.reachable,
            "distance_to_blocking_m": self.distance_to_blocking_m,
        }


class TraversabilityMap:
    def __init__(self, map_path, meta_path):
        self.map_path = Path(map_path).expanduser()
        self.meta_path = Path(meta_path).expanduser()
        self.grid = np.load(self.map_path)
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.resolution = float(self.meta["resolution_m"])
        self.origin_xy = np.asarray(self.meta["origin_xy"], dtype=np.float32)
        encoding = self.meta.get("encoding", {})
        self.free_values = set(self.meta.get("free_is_traversable", [encoding.get("free", FREE)]))
        self.blocking_values = set(
            self.meta.get("occupied_is_blocking", [encoding.get("occupied", OCCUPIED), encoding.get("inflated", INFLATED)])
        )

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    def world_to_grid(self, xy: Iterable[float]) -> Optional[Tuple[int, int]]:
        xy = np.asarray(list(xy), dtype=np.float32)
        ij = np.floor((xy[:2] - self.origin_xy) / self.resolution).astype(np.int32)
        gx, gy = int(ij[0]), int(ij[1])
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return None
        return gx, gy

    def cell_value(self, grid_xy: Tuple[int, int]) -> int:
        gx, gy = grid_xy
        return int(self.grid[gy, gx])

    def nearest_free(self, grid_xy: Tuple[int, int], max_radius_cells: int = 4) -> Optional[Tuple[int, int]]:
        gx, gy = grid_xy
        if self.cell_value((gx, gy)) in self.free_values:
            return gx, gy
        for radius in range(1, max_radius_cells + 1):
            for yy in range(max(0, gy - radius), min(self.height, gy + radius + 1)):
                for xx in range(max(0, gx - radius), min(self.width, gx + radius + 1)):
                    if max(abs(xx - gx), abs(yy - gy)) != radius:
                        continue
                    if int(self.grid[yy, xx]) in self.free_values:
                        return xx, yy
        return None

    def reachable_from(self, start_xy: Optional[Iterable[float]], goal_grid_xy: Tuple[int, int]) -> Optional[bool]:
        if start_xy is None:
            return None
        start_grid = self.world_to_grid(start_xy)
        if start_grid is None:
            return False
        start_grid = self.nearest_free(start_grid)
        goal_grid = self.nearest_free(goal_grid_xy)
        if start_grid is None or goal_grid is None:
            return False
        if start_grid == goal_grid:
            return True

        visited = np.zeros(self.grid.shape, dtype=bool)
        q = deque([start_grid])
        visited[start_grid[1], start_grid[0]] = True
        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height or visited[ny, nx]:
                    continue
                if int(self.grid[ny, nx]) not in self.free_values:
                    continue
                if (nx, ny) == goal_grid:
                    return True
                visited[ny, nx] = True
                q.append((nx, ny))
        return False

    def distance_to_blocking(self, grid_xy: Tuple[int, int], max_radius_m: float = 1.0) -> Optional[float]:
        gx, gy = grid_xy
        max_radius_cells = max(1, int(np.ceil(max_radius_m / self.resolution)))
        best = None
        for radius in range(0, max_radius_cells + 1):
            y0, y1 = max(0, gy - radius), min(self.height, gy + radius + 1)
            x0, x1 = max(0, gx - radius), min(self.width, gx + radius + 1)
            window = self.grid[y0:y1, x0:x1]
            if not np.isin(window, list(self.blocking_values)).any():
                continue
            yy, xx = np.where(np.isin(window, list(self.blocking_values)))
            xx = xx + x0
            yy = yy + y0
            dist_cells = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
            best = float(dist_cells.min() * self.resolution)
            break
        return best

    def evaluate_goal(
        self,
        goal_xy: Optional[Iterable[float]],
        robot_xy: Optional[Iterable[float]] = None,
        min_clearance_m: float = 0.20,
    ) -> GoalFeasibility:
        if goal_xy is None:
            return GoalFeasibility("missing", 0.0, "no_goal_xy", None, None, None, None, None)
        goal_values = list(goal_xy)
        goal_tuple = (float(goal_values[0]), float(goal_values[1]))
        grid_xy = self.world_to_grid(goal_tuple)
        if grid_xy is None:
            return GoalFeasibility("invalid", 0.0, "goal_outside_map", goal_tuple, None, None, None, None)

        value = self.cell_value(grid_xy)
        clearance = self.distance_to_blocking(grid_xy)
        reachable = self.reachable_from(robot_xy, grid_xy)

        if value in self.blocking_values:
            return GoalFeasibility("invalid", 0.0, "goal_in_blocking_cell", goal_tuple, grid_xy, value, reachable, clearance)
        if value not in self.free_values:
            return GoalFeasibility("risky", 0.35, "goal_in_unknown_cell", goal_tuple, grid_xy, value, reachable, clearance)
        if reachable is False:
            return GoalFeasibility("invalid", 0.15, "goal_not_reachable_from_robot", goal_tuple, grid_xy, value, reachable, clearance)
        if clearance is not None and clearance < min_clearance_m:
            return GoalFeasibility("risky", 0.55, "goal_clearance_too_small", goal_tuple, grid_xy, value, reachable, clearance)
        return GoalFeasibility("valid", 1.0, "goal_free_reachable_clear", goal_tuple, grid_xy, value, reachable, clearance)
