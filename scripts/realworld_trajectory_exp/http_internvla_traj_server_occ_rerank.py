import argparse
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent
from internnav.model.utils.vln_utils import S2Output, traj_to_actions
from internnav.utils.sim_occ_goal_correction import (
    append_jsonl,
    build_local_traversability_from_depth,
    rerank_trajectories_with_local_depth,
)


def patch_trajectory_inference(model, args):
    original_generate_traj = model.generate_traj

    def generate_traj_inference_mode(*call_args, **kwargs):
        kwargs.setdefault("predict_step_nums", args.traj_predict_steps)
        kwargs.setdefault("num_inference_steps", args.traj_inference_steps)
        kwargs.setdefault("num_sample_trajs", args.traj_num_samples)
        with torch.inference_mode():
            trajectories = original_generate_traj(*call_args, **kwargs)
        if torch.is_tensor(trajectories):
            return trajectories.clone()
        return trajectories

    model.generate_traj = generate_traj_inference_mode


def trajectory_to_discrete_fallback(trajectory):
    if trajectory is None:
        return None
    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] == 0:
        return None
    goal = traj[-1]
    x = float(goal[0]) if goal.shape[0] > 0 else 0.0
    y = float(goal[1]) if goal.shape[0] > 1 else 0.0
    if abs(y) > max(0.08, abs(x) * 0.35):
        return [2] if y > 0 else [3]
    if x > 0.08:
        return [1]
    return [0]


def _as_rgb_uint8(image):
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return arr[:, :, :3]


def _depth_to_color(depth_m, min_depth_m=0.05, max_depth_m=4.0):
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    norm = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth, min_depth_m, max_depth_m)
        norm[valid] = ((clipped[valid] - min_depth_m) / max(1e-6, max_depth_m - min_depth_m) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[~valid] = (255, 255, 255)
    return color


def _camera_ground_to_pixel(x_right, z_forward, floor_y, fx, fy, cx, cy, width, height):
    if z_forward <= 1e-4:
        return None
    u = int(round(fx * x_right / z_forward + cx))
    v = int(round(fy * floor_y / z_forward + cy))
    if u < 0 or u >= width or v < 0 or v >= height:
        return None
    return u, v


def _draw_robot_trajectory(image_bgr, trajectory, floor_y, intrinsic, color, label=None, thickness=3):
    if trajectory is None:
        return
    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] == 0:
        return
    h, w = image_bgr.shape[:2]
    fx, fy, cx, cy = float(intrinsic[0, 0]), float(intrinsic[1, 1]), float(intrinsic[0, 2]), float(intrinsic[1, 2])
    pixels = []
    for point in traj:
        forward = float(point[0])
        left = float(point[1]) if point.shape[0] > 1 else 0.0
        px = _camera_ground_to_pixel(-left, forward, floor_y, fx, fy, cx, cy, w, h)
        if px is not None:
            pixels.append(px)
    if len(pixels) < 2:
        return
    for p0, p1 in zip(pixels[:-1], pixels[1:]):
        cv2.line(image_bgr, p0, p1, color, thickness, lineType=cv2.LINE_AA)
    cv2.circle(image_bgr, pixels[-1], 7, color, -1, lineType=cv2.LINE_AA)
    if label:
        cv2.putText(image_bgr, label, (pixels[-1][0] + 8, pixels[-1][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def _put_label(image_bgr, label):
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (image_bgr.shape[1], 34), (0, 0, 0), -1)
    image_bgr[:] = cv2.addWeighted(overlay, 0.55, image_bgr, 0.45, 0)
    cv2.putText(image_bgr, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)



def _draw_pixel_goal(image_bgr, pixel_goal, color=(0, 255, 0)):
    if pixel_goal is None:
        return
    try:
        pg = tuple(int(x) for x in pixel_goal[:2])
    except Exception:
        return
    cv2.drawMarker(image_bgr, pg, color, cv2.MARKER_CROSS, 24, 3, line_type=cv2.LINE_AA)
    cv2.circle(image_bgr, pg, 9, color, 2, lineType=cv2.LINE_AA)
    cv2.putText(image_bgr, "goal", (pg[0] + 10, pg[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def _dp_actions_to_robot_paths(trajectories):
    if trajectories is None:
        return []
    if torch.is_tensor(trajectories):
        arr = trajectories.detach().float().cpu().numpy()
    else:
        arr = np.asarray(trajectories, dtype=np.float32)
    if arr.ndim == 2:
        return [arr[:, :2].astype(np.float32)]
    if arr.ndim != 3 or arr.shape[0] == 0:
        return []
    return [path.astype(np.float32) for path in np.cumsum(arr[:, :, :2] / 4.0, axis=1)]


def _draw_candidate_trajectories(image_bgr, candidate_paths, floor_y, intrinsic, selected_indices=None, draw_labels=False):
    selected = set()
    if selected_indices is not None:
        if isinstance(selected_indices, int):
            selected = {selected_indices}
        else:
            try:
                selected = {int(v) for v in selected_indices}
            except Exception:
                selected = set()
    palette = [(180, 180, 180), (255, 150, 80), (80, 180, 255), (180, 255, 80), (255, 80, 180), (180, 80, 255)]
    for idx, path in enumerate(candidate_paths):
        color = palette[idx % len(palette)]
        thickness = 1
        label = None
        if idx in selected:
            color = (0, 210, 255)
            thickness = 3
            label = f"sel{idx}" if draw_labels else None
        elif draw_labels:
            label = f"c{idx}"
        _draw_robot_trajectory(image_bgr, path, floor_y, intrinsic, color, label, thickness=thickness)


def _make_eight_panel_debug(
    rgb_bgr,
    depth_color,
    mask_bgr,
    occ,
    intrinsic,
    pixel_goal,
    candidate_paths,
    before_trajectory,
    final_trajectory,
    record,
    label,
):
    selected = record.get("selected_index") if isinstance(record, dict) else None
    p1 = rgb_bgr.copy()
    _draw_pixel_goal(p1, pixel_goal)

    p2 = depth_color.copy()

    p3 = rgb_bgr.copy()
    _draw_candidate_trajectories(p3, candidate_paths, occ["floor_y"], intrinsic, draw_labels=True)
    _draw_pixel_goal(p3, pixel_goal)

    p4 = rgb_bgr.copy()
    _draw_candidate_trajectories(p4, candidate_paths, occ["floor_y"], intrinsic)
    _draw_robot_trajectory(p4, before_trajectory, occ["floor_y"], intrinsic, (255, 0, 255), "before", thickness=3)
    _draw_pixel_goal(p4, pixel_goal)

    p5 = mask_bgr.copy()

    p6 = mask_bgr.copy()
    _draw_candidate_trajectories(p6, candidate_paths, occ["floor_y"], intrinsic, selected_indices=selected, draw_labels=True)
    _draw_robot_trajectory(p6, final_trajectory, occ["floor_y"], intrinsic, (0, 255, 255), "after", thickness=3)
    _draw_pixel_goal(p6, pixel_goal)

    rgb_occ = cv2.addWeighted(rgb_bgr, 0.62, mask_bgr, 0.38, 0)
    p7 = rgb_occ.copy()
    _draw_candidate_trajectories(p7, candidate_paths, occ["floor_y"], intrinsic, selected_indices=selected)
    _draw_robot_trajectory(p7, before_trajectory, occ["floor_y"], intrinsic, (255, 0, 255), "before", thickness=2)
    _draw_robot_trajectory(p7, final_trajectory, occ["floor_y"], intrinsic, (0, 255, 255), "after", thickness=3)
    _draw_pixel_goal(p7, pixel_goal)

    p8 = rgb_occ.copy()
    _draw_robot_trajectory(p8, final_trajectory, occ["floor_y"], intrinsic, (0, 255, 255), "final", thickness=4)
    _draw_pixel_goal(p8, pixel_goal)

    panels = [p1, p2, p3, p4, p5, p6, p7, p8]
    titles = [
        "1 RGB + goal",
        "2 depth",
        "3 RGB + all diffusion candidates",
        "4 RGB + all candidates + before mean",
        "5 OCC mask: blue=free red=occupied gray=other",
        "6 OCC mask + all candidates + selected/final",
        "7 RGB + OCC + candidates + before/after",
        "8 RGB + OCC + final route",
    ]
    for panel, title in zip(panels, titles):
        _put_label(panel, title)
    row1 = np.concatenate(panels[:4], axis=1)
    row2 = np.concatenate(panels[4:], axis=1)
    banner = np.full((44, row1.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(banner, label[:220], (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([banner, row1, row2], axis=0)


def _save_occ_rerank_debug(image, depth_m, intrinsic, trajectory, pixel_goal, record, save_dir, step_idx, args, before_trajectory=None, candidate_trajectories=None):
    if not getattr(args, "save_traj_occ_debug", True):
        return None
    every = max(1, int(getattr(args, "traj_occ_debug_every", 1)))
    if int(step_idx) % every != 0:
        return None

    intr = np.asarray(intrinsic, dtype=np.float32)
    fx, fy, cx, cy = float(intr[0, 0]), float(intr[1, 1]), float(intr[0, 2]), float(intr[1, 2])
    rgb = _as_rgb_uint8(image)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_color = _depth_to_color(
        depth_m,
        min_depth_m=float(args.traj_occ_min_depth_m),
        max_depth_m=float(args.traj_occ_max_depth_m),
    )
    occ = build_local_traversability_from_depth(
        depth_m,
        fx,
        fy,
        cx,
        cy,
        min_depth_m=float(args.traj_occ_min_depth_m),
        max_depth_m=float(args.traj_occ_max_depth_m),
        stride=int(args.traj_occ_stride),
        inflation_radius_m=float(args.traj_occ_robot_radius_m) + float(args.traj_occ_safety_margin_m),
        floor_percentile=float(args.traj_occ_floor_percentile),
        free_height_tol_m=float(args.traj_occ_free_height_tol_m),
        occupied_height_m=float(args.traj_occ_occupied_height_m),
    )
    if occ is None:
        return None

    h, w = rgb.shape[:2]
    mask_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    points = occ["points"]
    u = points["u"].astype(np.int32)
    v = points["v"].astype(np.int32)
    z = points["z"].astype(np.float32)
    free = points["free_mask"]
    occupied = points["occupied_mask"]
    order = np.argsort(z)[::-1]
    radius = max(1, int(getattr(args, "traj_occ_debug_point_radius", 2)))
    for i in order:
        if u[i] < 0 or u[i] >= w or v[i] < 0 or v[i] >= h:
            continue
        color = (80, 80, 80)
        if free[i]:
            color = (255, 140, 40)
        if occupied[i]:
            color = (0, 0, 255)
        cv2.circle(mask_bgr, (int(u[i]), int(v[i])), radius, color, -1, lineType=cv2.LINE_AA)

    route_bgr = mask_bgr.copy()
    overlay_bgr = cv2.addWeighted(rgb_bgr, 0.62, mask_bgr, 0.38, 0)
    status = "no_record"
    reason = ""
    selected = None
    if isinstance(record, dict):
        status = str(record.get("status", "unknown"))
        reason = str(record.get("reason", ""))
        selected = record.get("selected_index")
    label = f"step={step_idx} {status} {reason} selected={selected}"
    _draw_robot_trajectory(route_bgr, before_trajectory, occ["floor_y"], intr, (255, 0, 255), "before", thickness=2)
    _draw_robot_trajectory(overlay_bgr, before_trajectory, occ["floor_y"], intr, (255, 0, 255), "before", thickness=2)
    _draw_robot_trajectory(route_bgr, trajectory, occ["floor_y"], intr, (0, 255, 255), "after", thickness=3)
    _draw_robot_trajectory(overlay_bgr, trajectory, occ["floor_y"], intr, (0, 255, 255), "after", thickness=3)
    if pixel_goal is not None:
        try:
            pg = tuple(int(x) for x in pixel_goal[:2])
            cv2.drawMarker(route_bgr, pg, (0, 255, 0), cv2.MARKER_CROSS, 24, 3, line_type=cv2.LINE_AA)
            cv2.drawMarker(overlay_bgr, pg, (0, 255, 0), cv2.MARKER_CROSS, 24, 3, line_type=cv2.LINE_AA)
        except Exception:
            pass

    panels = [rgb_bgr.copy(), depth_color, route_bgr, overlay_bgr]
    labels = ["RGB raw", f"Depth {float(args.traj_occ_min_depth_m):.2f}-{float(args.traj_occ_max_depth_m):.2f}m", "OCC + routes: magenta=before yellow=after", "RGB + routes: magenta=before yellow=after"]
    for panel, panel_label in zip(panels, labels):
        _put_label(panel, panel_label)
    top = np.concatenate(panels, axis=1)
    banner = np.full((44, top.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(banner, label[:220], (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    panel = np.concatenate([banner, top], axis=0)

    debug_dir = Path(save_dir) / "occ_rerank_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"step_{int(step_idx):06d}"
    panel_path = debug_dir / f"{prefix}_panel.png"
    eight_dir = Path(save_dir) / "occ_rerank_eight_panel_debug"
    eight_dir.mkdir(parents=True, exist_ok=True)
    eight_panel_path = eight_dir / f"{prefix}_eight_panel.png"
    rgb_path = debug_dir / f"{prefix}_rgb.png"
    depth_path = debug_dir / f"{prefix}_depth.png"
    occ_path = debug_dir / f"{prefix}_occ_route.png"
    overlay_path = debug_dir / f"{prefix}_rgb_occ_route.png"
    cv2.imwrite(str(panel_path), panel)
    candidate_paths = _dp_actions_to_robot_paths(candidate_trajectories)
    eight_panel = _make_eight_panel_debug(
        rgb_bgr,
        depth_color,
        mask_bgr,
        occ,
        intr,
        pixel_goal,
        candidate_paths,
        before_trajectory,
        trajectory,
        record if isinstance(record, dict) else {},
        label,
    )
    cv2.imwrite(str(eight_panel_path), eight_panel)
    cv2.imwrite(str(rgb_path), rgb_bgr)
    cv2.imwrite(str(depth_path), depth_color)
    cv2.imwrite(str(occ_path), route_bgr)
    cv2.imwrite(str(overlay_path), overlay_bgr)
    return str(eight_panel_path)


class OccRerankRealworldAgent(InternVLAN1AsyncAgent):
    """Real-world InternVLA agent with optional traj_occ_rerank v2.

    This intentionally leaves System2/pixel-goal generation untouched.  The
    only intervention is between NavDP candidate generation and the legacy
    traj_to_actions() ensemble reduction.
    """

    def reset(self):
        super().reset()
        self.realworld_traj_occ_log = Path(self.save_dir) / "realworld_traj_occ_rerank.jsonl"

    def _rerank_trajs_if_enabled(self, trajectories, depth_m, intrinsic, phase):
        if not getattr(self.args, "enable_traj_occ_rerank", False):
            return trajectories, None
        if trajectories is None:
            return trajectories, {"status": "fallback", "reason": "missing_trajectories", "phase": phase}

        intr = np.asarray(intrinsic, dtype=np.float32)
        try:
            fx, fy, cx, cy = float(intr[0, 0]), float(intr[1, 1]), float(intr[0, 2]), float(intr[1, 2])
        except Exception as exc:
            return trajectories, {
                "status": "fallback",
                "reason": "invalid_intrinsic",
                "phase": phase,
                "error": str(exc),
            }

        try:
            reranked, record = rerank_trajectories_with_local_depth(
                trajectories,
                depth_m,
                fx,
                fy,
                cx,
                cy,
                min_depth_m=self.args.traj_occ_min_depth_m,
                max_depth_m=self.args.traj_occ_max_depth_m,
                stride=self.args.traj_occ_stride,
                floor_percentile=self.args.traj_occ_floor_percentile,
                free_height_tol_m=self.args.traj_occ_free_height_tol_m,
                occupied_height_m=self.args.traj_occ_occupied_height_m,
                robot_radius_m=self.args.traj_occ_robot_radius_m,
                safety_margin_m=self.args.traj_occ_safety_margin_m,
                support_radius_m=self.args.traj_occ_support_radius_m,
                min_support_fraction=self.args.traj_occ_min_support_fraction,
                horizon_steps=self.args.traj_occ_horizon_steps,
            )
            record.update(
                {
                    "phase": phase,
                    "episode_idx": int(self.episode_idx),
                    "pixel_goal": self.output_pixel.tolist()
                    if hasattr(self.output_pixel, "tolist")
                    else self.output_pixel,
                }
            )
            append_jsonl(str(self.realworld_traj_occ_log), record)
            print(
                "realworld_traj_occ",
                record.get("status"),
                record.get("reason"),
                "safe",
                record.get("safe_candidate_count"),
                "supported",
                record.get("supported_safe_candidate_count"),
                flush=True,
            )
            return reranked, record
        except Exception as exc:
            record = {
                "status": "fallback",
                "reason": "rerank_exception",
                "phase": phase,
                "episode_idx": int(self.episode_idx),
                "error": repr(exc),
            }
            append_jsonl(str(self.realworld_traj_occ_log), record)
            print("realworld_traj_occ fallback rerank_exception", repr(exc), flush=True)
            return trajectories, record

    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        dual_sys_output = S2Output()
        no_output_flag = self.output_action is None and self.output_latent is None
        run_s2 = (self.episode_idx - self.last_s2_idx > self.PLAN_STEP_GAP) or look_down or no_output_flag
        if run_s2:
            self.output_action, self.output_latent, self.output_pixel = self.step_s2(
                rgb, depth, pose, instruction, intrinsic, look_down
            )
            self.last_s2_idx = self.episode_idx
            dual_sys_output.output_pixel = self.output_pixel
            self.pixel_goal_rgb = copy.deepcopy(rgb)
            self.pixel_goal_depth = copy.deepcopy(depth)
        else:
            self.step_no_infer(rgb, depth, pose)

        if getattr(self.args, "disable_trajectory", False) and self.output_latent is not None:
            pixel_x = self.output_pixel[0] if self.output_pixel is not None else rgb.shape[1] // 2
            image_center = rgb.shape[1] / 2.0
            deadband = rgb.shape[1] * getattr(self.args, "pixel_deadband_ratio", 0.18)
            if pixel_x < image_center - deadband:
                dual_sys_output.output_action = [2]
            elif pixel_x > image_center + deadband:
                dual_sys_output.output_action = [3]
            else:
                dual_sys_output.output_action = [1]
            self.output_action = None
            self.output_latent = None
            return dual_sys_output

        if self.output_action is not None:
            dual_sys_output.output_action = copy.deepcopy(self.output_action)
            self.output_action = None
        elif self.output_latent is not None:
            processed_pixel_rgb = np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255
            processed_pixel_depth = np.array(Image.fromarray(self.pixel_goal_depth).resize((224, 224)))
            processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
            processed_depth = np.array(Image.fromarray(depth).resize((224, 224)))
            rgbs = (
                torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
                .unsqueeze(0)
                .to(self.device)
            )
            depths = (
                torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )
            trajectories = self.step_s1(self.output_latent, rgbs, depths)
            pre_rerank_trajectory = traj_to_actions(trajectories, use_discrate_action=False)
            raw_trajectories = trajectories
            trajectories, rerank_record = self._rerank_trajs_if_enabled(
                trajectories,
                depth,
                intrinsic,
                "new_pixel_goal" if run_s2 else "local_replan",
            )
            dual_sys_output.output_trajectory = traj_to_actions(trajectories, use_discrate_action=False)
            try:
                debug_path = _save_occ_rerank_debug(
                    rgb,
                    depth,
                    intrinsic,
                    dual_sys_output.output_trajectory.detach().cpu().numpy()
                    if torch.is_tensor(dual_sys_output.output_trajectory)
                    else np.asarray(dual_sys_output.output_trajectory),
                    self.output_pixel,
                    rerank_record,
                    self.save_dir,
                    self.episode_idx,
                    self.args,
                    before_trajectory=pre_rerank_trajectory.detach().cpu().numpy()
                    if torch.is_tensor(pre_rerank_trajectory)
                    else np.asarray(pre_rerank_trajectory),
                    candidate_trajectories=raw_trajectories,
                )
                if debug_path:
                    print(f"realworld_traj_occ_debug {debug_path}", flush=True)
            except Exception as exc:
                print("realworld_traj_occ_debug failed", repr(exc), flush=True)

        return dual_sys_output


app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ""
agent = None
args = None


def _set_occ_rerank_enabled(enabled):
    """Flip the occ V2 re-rank switch at runtime (no server restart / weight reload)."""
    enabled = bool(enabled)
    if args is not None:
        args.enable_traj_occ_rerank = enabled
    if agent is not None and getattr(agent, "args", None) is not None:
        agent.args.enable_traj_occ_rerank = enabled
    print(f"[occ_rerank] runtime toggle -> enable_traj_occ_rerank={enabled}", flush=True)
    return enabled


@app.route("/occ_rerank", methods=["GET", "POST"])
def occ_rerank_switch():
    """GET returns the current state; POST {"enabled": true/false} flips it."""
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        if "enabled" not in data:
            return jsonify({"error": "POST body needs an 'enabled' boolean"}), 400
        _set_occ_rerank_enabled(data["enabled"])
    return jsonify({"enable_traj_occ_rerank": bool(getattr(args, "enable_traj_occ_rerank", False))})


@app.route("/eval_dual", methods=["POST"])
def eval_dual():
    global idx, output_dir, start_time
    start_time = time.time()

    image_file = request.files["image"]
    depth_file = request.files["depth"]
    json_data = request.form["json"]
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert("RGB")
    image = np.asarray(image)

    depth = Image.open(depth_file.stream)
    depth = depth.convert("I")
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    print(f"read http data cost {time.time() - start_time}")

    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    instruction = data.get("instruction") or args.instruction
    policy_init = data["reset"]
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = "output/runs" + datetime.now().strftime("%m-%d-%H%M")
        os.makedirs(output_dir, exist_ok=True)
        print("init reset model!!!")
        agent.reset()

    idx += 1

    look_down = False
    t0 = time.time()
    dual_sys_output = {}

    try:
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
        )
        if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
            look_down = True
            dual_sys_output = agent.step(
                image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
            )
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        agent.reset()
        return jsonify({"error": "CUDA_OOM", "detail": str(exc)}), 503
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_output = {}
    if dual_sys_output.output_action is not None:
        json_output["discrete_action"] = dual_sys_output.output_action
    else:
        trajectory = dual_sys_output.output_trajectory.tolist()
        json_output["trajectory"] = trajectory
        if args.return_discrete_fallback:
            fallback = trajectory_to_discrete_fallback(trajectory)
            if fallback is not None:
                json_output["discrete_action_fallback"] = fallback
        if dual_sys_output.output_pixel is not None:
            json_output["pixel_goal"] = dual_sys_output.output_pixel

    t1 = time.time()
    generate_time = t1 - t0
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return jsonify(json_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=8)
    parser.add_argument("--disable_trajectory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pixel_deadband_ratio", type=float, default=0.18)
    parser.add_argument("--return_discrete_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--port", type=int, default=5802)
    parser.add_argument("--traj_predict_steps", type=int, default=32)
    parser.add_argument("--traj_inference_steps", type=int, default=16)
    parser.add_argument("--traj_num_samples", type=int, default=32)
    parser.add_argument("--enable_traj_occ_rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--traj_occ_min_depth_m", type=float, default=0.05)
    parser.add_argument("--traj_occ_max_depth_m", type=float, default=8.0)
    parser.add_argument("--traj_occ_stride", type=int, default=4)
    parser.add_argument("--traj_occ_floor_percentile", type=float, default=90.0)
    parser.add_argument("--traj_occ_free_height_tol_m", type=float, default=0.08)
    parser.add_argument("--traj_occ_occupied_height_m", type=float, default=0.18)
    parser.add_argument("--traj_occ_robot_radius_m", type=float, default=0.24)
    parser.add_argument("--traj_occ_safety_margin_m", type=float, default=0.08)
    parser.add_argument("--traj_occ_support_radius_m", type=float, default=0.35)
    parser.add_argument("--traj_occ_min_support_fraction", type=float, default=0.55)
    parser.add_argument("--traj_occ_horizon_steps", type=int, default=12)
    parser.add_argument("--save_traj_occ_debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--traj_occ_debug_every", type=int, default=1)
    parser.add_argument("--traj_occ_debug_point_radius", type=int, default=2)
    parser.add_argument(
        "--instruction",
        type=str,
        default="Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor",
    )
    parser.add_argument("--instruction_file", type=str, default=None)
    args = parser.parse_args()
    if args.instruction_file:
        with open(args.instruction_file, "r") as f:
            args.instruction = f.read().strip()

    args.camera_intrinsic = np.array(
        [
            [386.5, 0.0, 328.9, 0.0],
            [0.0, 386.5, 244, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    agent = OccRerankRealworldAgent(args)
    patch_trajectory_inference(agent.model, args)
    os.makedirs(agent.save_dir, exist_ok=True)
    agent.step(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640), dtype=np.float32),
        np.eye(4),
        "hello",
        intrinsic=args.camera_intrinsic,
    )
    agent.reset()

    app.run(host="0.0.0.0", port=args.port)
