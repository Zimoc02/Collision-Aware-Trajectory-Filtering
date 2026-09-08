#!/usr/bin/env python3
"""Evaluate logged pixel goals against an exported traversability map.

This is an offline metric layer for baseline runs. It does not change actions
or success metrics; it reads trajectory_steps.jsonl and writes occupancy-derived
pixel-goal feasibility records next to the run.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from internnav.utils.occ_goal_validator import TraversabilityMap


def read_jsonl(path):
    records = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def infer_run_dir(args):
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()
    steps_path = Path(args.trajectory_steps).expanduser().resolve()
    return steps_path.parent


def goal_xy_from_step(step, source):
    if source == "trajectory_world_endpoint":
        traj = step.get("trajectory_world")
        if traj:
            return traj[-1], "trajectory_world_endpoint"
        return None, "missing_trajectory_world"
    if source == "odom_plus_local_endpoint":
        traj = step.get("trajectory_local")
        odom = step.get("odom_infer")
        if not traj or not odom or len(odom) < 3:
            return None, "missing_local_trajectory_or_odom"
        local_x, local_y = traj[-1][:2]
        ox, oy, yaw = odom[:3]
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        world_x = float(ox + c * local_x - s * local_y)
        world_y = float(oy + s * local_x + c * local_y)
        return [world_x, world_y], "odom_plus_local_endpoint"
    raise ValueError(f"Unsupported goal source: {source}")


def robot_xy_from_step(step):
    odom = step.get("odom_infer")
    if odom and len(odom) >= 2:
        return [float(odom[0]), float(odom[1])]
    return [0.0, 0.0]


def evaluate(args):
    run_dir = infer_run_dir(args)
    steps_path = Path(args.trajectory_steps).expanduser().resolve() if args.trajectory_steps else run_dir / "trajectory_steps.jsonl"
    out_jsonl = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else run_dir / "pixel_goal_occ_eval.jsonl"
    out_summary = Path(args.output_summary).expanduser().resolve() if args.output_summary else run_dir / "pixel_goal_occ_eval_summary.json"

    validator = TraversabilityMap(args.traversability_map, args.traversability_meta)
    steps = read_jsonl(steps_path)
    records = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for step in steps:
            pixel_goal = step.get("pixel_goal_mapped", step.get("pixel_goal"))
            has_pixel_goal = pixel_goal is not None
            goal_xy, goal_source = goal_xy_from_step(step, args.goal_source)
            if args.only_steps_with_pixel_goal and not has_pixel_goal:
                feasibility = validator.evaluate_goal(None)
            else:
                feasibility = validator.evaluate_goal(
                    goal_xy,
                    robot_xy=robot_xy_from_step(step),
                    min_clearance_m=args.min_clearance_m,
                )
            record = {
                "idx": step.get("idx"),
                "time": step.get("time"),
                "instruction": step.get("instruction"),
                "has_pixel_goal": bool(has_pixel_goal),
                "pixel_goal": pixel_goal,
                "goal_source": goal_source,
                "feasibility": feasibility.to_dict(),
            }
            records.append(record)
            f.write(json.dumps(record) + "\n")

    evaluated = [r for r in records if (r["has_pixel_goal"] or not args.only_steps_with_pixel_goal)]
    counts = {}
    for record in evaluated:
        status = record["feasibility"]["status"]
        counts[status] = counts.get(status, 0) + 1
    valid_scores = [r["feasibility"]["score"] for r in evaluated if r["feasibility"]["status"] != "missing"]
    pixel_goal_records = [r for r in records if r["has_pixel_goal"]]
    pixel_goal_valid = [r for r in pixel_goal_records if r["feasibility"]["status"] == "valid"]
    summary = {
        "run_dir": str(run_dir),
        "trajectory_steps": str(steps_path),
        "traversability_map": str(Path(args.traversability_map).expanduser().resolve()),
        "traversability_meta": str(Path(args.traversability_meta).expanduser().resolve()),
        "output_jsonl": str(out_jsonl),
        "total_steps": len(steps),
        "evaluated_steps": len(evaluated),
        "steps_with_pixel_goal": len(pixel_goal_records),
        "pixel_goal_valid_steps": len(pixel_goal_valid),
        "pixel_goal_valid_rate": float(len(pixel_goal_valid) / len(pixel_goal_records)) if pixel_goal_records else None,
        "mean_feasibility_score": float(np.mean(valid_scores)) if valid_scores else None,
        "status_counts": counts,
        "parameters": {
            "goal_source": args.goal_source,
            "only_steps_with_pixel_goal": bool(args.only_steps_with_pixel_goal),
            "min_clearance_m": float(args.min_clearance_m),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Directory containing trajectory_steps.jsonl.")
    parser.add_argument("--trajectory-steps", default=None, help="Explicit trajectory_steps.jsonl path.")
    parser.add_argument("--traversability-map", required=True)
    parser.add_argument("--traversability-meta", required=True)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument(
        "--goal-source",
        choices=["trajectory_world_endpoint", "odom_plus_local_endpoint"],
        default="trajectory_world_endpoint",
        help="Proxy used for the world-frame navigation target associated with the pixel goal.",
    )
    parser.add_argument("--min-clearance-m", type=float, default=0.20)
    parser.set_defaults(only_steps_with_pixel_goal=True)
    parser.add_argument("--only-steps-with-pixel-goal", dest="only_steps_with_pixel_goal", action="store_true")
    parser.add_argument("--all-steps", dest="only_steps_with_pixel_goal", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
