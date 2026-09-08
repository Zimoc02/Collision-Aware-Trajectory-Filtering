#!/usr/bin/env python3
"""Smoke test for occupancy-based pixel-goal evaluation."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    repo_root = Path(__file__).resolve().parents[2]
    eval_script = repo_root / "scripts" / "realworld_trajectory_exp" / "evaluate_pixel_goal_occ.py"

    with tempfile.TemporaryDirectory(prefix="internnav_occ_smoke_") as tmp_name:
        tmp = Path(tmp_name)
        grid = np.ones((20, 20), dtype=np.uint8)
        grid[8:12, 8:12] = 2
        grid[15:18, 15:18] = 0
        map_path = tmp / "traversability_map.npy"
        meta_path = tmp / "traversability_meta.json"
        np.save(map_path, grid)
        meta_path.write_text(
            json.dumps(
                {
                    "format": "internnav_traversability_v1",
                    "resolution_m": 0.1,
                    "origin_xy": [-1.0, -1.0],
                    "encoding": {"unknown": 0, "free": 1, "occupied": 2, "inflated": 3},
                    "free_is_traversable": [1],
                    "occupied_is_blocking": [2, 3],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        run_dir = tmp / "baseline_run"
        run_dir.mkdir()
        steps = [
            {
                "idx": 1,
                "instruction": "smoke valid",
                "pixel_goal": [10, 10],
                "odom_infer": [-0.8, -0.8, 0.0],
                "trajectory_world": [[-0.7, -0.7]],
            },
            {
                "idx": 2,
                "instruction": "smoke occupied",
                "pixel_goal": [11, 11],
                "odom_infer": [-0.8, -0.8, 0.0],
                "trajectory_world": [[0.0, 0.0]],
            },
            {
                "idx": 3,
                "instruction": "smoke unknown",
                "pixel_goal": [12, 12],
                "odom_infer": [-0.8, -0.8, 0.0],
                "trajectory_world": [[0.6, 0.6]],
            },
        ]
        (run_dir / "trajectory_steps.jsonl").write_text(
            "\n".join(json.dumps(step) for step in steps) + "\n",
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            str(eval_script),
            "--run-dir",
            str(run_dir),
            "--traversability-map",
            str(map_path),
            "--traversability-meta",
            str(meta_path),
        ]
        result = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            return result.returncode

        summary = json.loads((run_dir / "pixel_goal_occ_eval_summary.json").read_text(encoding="utf-8"))
        expected = {"valid": 1, "invalid": 1, "risky": 1}
        if summary.get("status_counts") != expected:
            print(f"Unexpected status_counts: {summary.get('status_counts')} != {expected}", file=sys.stderr)
            return 1
        print("SMOKE_TEST_OCC_EVAL_OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
