"""Depth-derived local traversability checks for simulation pixel goals."""

import json
from pathlib import Path

import cv2
import numpy as np


UNKNOWN = 0
FREE = 1
OCCUPIED = 2
INFLATED = 3


def _nearest_valid_depth(depth_m, u, v, radius=5):
    h, w = depth_m.shape
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0.02)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _pixel_to_camera(u, v, z, fx, fy, cx, cy):
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def _camera_to_pixel(point, fx, fy, cx, cy, width, height):
    x, y, z = [float(v) for v in point]
    if z <= 1e-4:
        return None
    u = int(round(fx * x / z + cx))
    v = int(round(fy * y / z + cy))
    if u < 0 or u >= width or v < 0 or v >= height:
        return None
    return [u, v]


def _grid_index(x, z, origin_xz, resolution, width, height):
    gx = int(np.floor((x - origin_xz[0]) / resolution))
    gy = int(np.floor((z - origin_xz[1]) / resolution))
    if gx < 0 or gx >= width or gy < 0 or gy >= height:
        return None
    return gx, gy


def _dilate(mask, radius_cells):
    if radius_cells <= 0 or not np.any(mask):
        return mask.copy()
    size = radius_cells * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def build_local_traversability_from_depth(
    depth_m,
    fx,
    fy,
    cx,
    cy,
    resolution_m=0.05,
    min_depth_m=0.05,
    max_depth_m=5.0,
    stride=4,
    inflation_radius_m=0.20,
    floor_percentile=82.0,
    free_height_tol_m=0.18,
    occupied_height_m=0.22,
):
    """Build a local X-Z traversability grid in the camera optical frame.

    The grid is intentionally lightweight for online evaluation. It marks
    floor-like depth points as free and points above that floor estimate as
    occupied. It is not a replacement for nvblox; full sequences are exported
    separately so the native nvblox pipeline can refine the map offline.
    """

    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[ys, xs].astype(np.float32)
    valid = np.isfinite(z) & (z > float(min_depth_m)) & (z <= max_depth_m)
    if np.count_nonzero(valid) < 32:
        return None

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    near = (z > max(float(min_depth_m), 0.20)) & (z < max_depth_m)
    floor_y = float(np.percentile(y[near] if np.any(near) else y, floor_percentile))
    free_mask = np.abs(y - floor_y) <= free_height_tol_m
    occ_mask = y < (floor_y - occupied_height_m)

    all_x = x
    all_z = z
    padding = 0.35
    min_x, max_x = float(all_x.min() - padding), float(all_x.max() + padding)
    min_z, max_z = 0.0, float(all_z.max() + padding)
    width = max(1, int(np.ceil((max_x - min_x) / resolution_m)) + 1)
    height = max(1, int(np.ceil((max_z - min_z) / resolution_m)) + 1)
    grid = np.zeros((height, width), dtype=np.uint8)
    origin_xz = np.array([min_x, min_z], dtype=np.float32)

    free_points = np.stack([x[free_mask], z[free_mask]], axis=1)
    occ_points = np.stack([x[occ_mask], z[occ_mask]], axis=1)
    for px, pz in free_points:
        idx = _grid_index(px, pz, origin_xz, resolution_m, width, height)
        if idx is not None:
            grid[idx[1], idx[0]] = FREE

    occupied = np.zeros_like(grid, dtype=bool)
    for px, pz in occ_points:
        idx = _grid_index(px, pz, origin_xz, resolution_m, width, height)
        if idx is not None:
            occupied[idx[1], idx[0]] = True

    inflated = _dilate(occupied, int(np.ceil(inflation_radius_m / resolution_m)))
    grid[inflated & (grid != OCCUPIED)] = INFLATED
    grid[occupied] = OCCUPIED
    return {
        "grid": grid,
        "origin_xz": origin_xz,
        "resolution_m": float(resolution_m),
        "floor_y": floor_y,
        "points": {
            "u": xs.astype(np.int32),
            "v": ys.astype(np.int32),
            "z": z,
            "y": y,
            "free_mask": free_mask,
            "occupied_mask": occ_mask,
        },
        "encoding": {"unknown": UNKNOWN, "free": FREE, "occupied": OCCUPIED, "inflated": INFLATED},
    }


def rerank_trajectories_with_local_depth(
    dp_actions,
    depth_m,
    fx,
    fy,
    cx,
    cy,
    min_depth_m=0.05,
    max_depth_m=5.0,
    stride=4,
    floor_percentile=90.0,
    free_height_tol_m=0.08,
    occupied_height_m=0.18,
    robot_radius_m=0.20,
    safety_margin_m=0.08,
    support_radius_m=0.30,
    min_support_fraction=0.55,
    horizon_steps=12,
):
    """Conservatively replace an unsafe ensemble-mean trajectory.

    The diffusion output uses robot coordinates ``x=forward, y=left`` while
    the depth cloud uses optical-camera coordinates ``x=right, z=forward``.
    System-2's answer and latent are intentionally left untouched.  If the
    original ensemble mean is collision-free and sufficiently supported by
    observed floor points, the input tensor is returned unchanged so the
    legacy averaging behavior is exactly preserved by ``traj_to_actions``.
    """

    actions_np = dp_actions.detach().float().cpu().numpy()
    if actions_np.ndim != 3 or actions_np.shape[0] == 0:
        return dp_actions, {"status": "fallback", "reason": "invalid_trajectory_shape"}

    occ = build_local_traversability_from_depth(
        depth_m,
        fx,
        fy,
        cx,
        cy,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        stride=stride,
        inflation_radius_m=0.0,
        floor_percentile=floor_percentile,
        free_height_tol_m=free_height_tol_m,
        occupied_height_m=occupied_height_m,
    )
    if occ is None:
        return dp_actions, {"status": "fallback", "reason": "insufficient_depth_for_rerank"}

    points = occ["points"]
    point_x = (points["u"].astype(np.float32) - float(cx)) * points["z"] / float(fx)
    point_z = points["z"].astype(np.float32)
    free_points = np.stack([point_x[points["free_mask"]], point_z[points["free_mask"]]], axis=1)
    obstacle_points = np.stack(
        [point_x[points["occupied_mask"]], point_z[points["occupied_mask"]]], axis=1
    )
    if len(free_points) < 16:
        return dp_actions, {"status": "fallback", "reason": "insufficient_floor_support"}

    # Model deltas are normalized by 4.  Convert [forward, left] paths to
    # camera-ground [right, forward] paths for comparison with the depth cloud.
    paths_robot = np.cumsum(actions_np[:, :, :2] / 4.0, axis=1)
    mean_robot = paths_robot.mean(axis=0)

    def to_camera_ground(path_robot):
        return np.stack([-path_robot[:, 1], path_robot[:, 0]], axis=1)

    collision_radius = float(robot_radius_m) + float(safety_margin_m)

    def path_metrics(path_robot):
        checked_steps = max(1, min(int(horizon_steps), len(path_robot)))
        path = to_camera_ground(path_robot[:checked_steps])
        active = path[:, 1] > 0.10
        active_path = path[active]
        if len(active_path) == 0:
            return {
                "collision_count": 0,
                "support_fraction": 1.0,
                "min_obstacle_clearance_m": None,
                "behind_count": int(np.count_nonzero(path[:, 1] < -0.05)),
                "path_length_m": 0.0,
            }

        free_dist = np.sqrt(((active_path[:, None, :] - free_points[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        support_fraction = float(np.mean(free_dist <= float(support_radius_m)))
        if len(obstacle_points):
            obstacle_dist = np.sqrt(
                ((active_path[:, None, :] - obstacle_points[None, :, :]) ** 2).sum(axis=2)
            ).min(axis=1)
            collision_count = int(np.count_nonzero(obstacle_dist < collision_radius))
            min_clearance = float(obstacle_dist.min())
        else:
            collision_count = 0
            min_clearance = None
        segments = np.diff(np.concatenate([np.zeros((1, 2), dtype=path.dtype), path], axis=0), axis=0)
        return {
            "collision_count": collision_count,
            "support_fraction": support_fraction,
            "min_obstacle_clearance_m": min_clearance,
            "behind_count": int(np.count_nonzero(path[:, 1] < -0.05)),
            "path_length_m": float(np.linalg.norm(segments, axis=1).sum()),
        }

    def is_collision_safe(metrics):
        # Sparse single-view floor observations are not reliable enough to
        # declare an otherwise clear trajectory unsafe.  Unknown support is
        # retained as a preference when selecting a replacement, but only an
        # observed obstacle (or a backwards path) triggers intervention.
        return metrics["collision_count"] == 0 and metrics["behind_count"] == 0

    mean_metrics = path_metrics(mean_robot)
    record = {
        "status": "kept_mean",
        "reason": "ensemble_mean_is_safe",
        "selected_index": None,
        "num_candidates": int(actions_np.shape[0]),
        "mean_metrics": mean_metrics,
        "params": {
            "min_depth_m": float(min_depth_m),
            "max_depth_m": float(max_depth_m),
            "stride": int(stride),
            "floor_percentile": float(floor_percentile),
            "free_height_tol_m": float(free_height_tol_m),
            "occupied_height_m": float(occupied_height_m),
            "robot_radius_m": float(robot_radius_m),
            "safety_margin_m": float(safety_margin_m),
            "collision_radius_m": collision_radius,
            "support_radius_m": float(support_radius_m),
            "min_support_fraction": float(min_support_fraction),
            "horizon_steps": int(horizon_steps),
        },
    }
    if is_collision_safe(mean_metrics):
        return dp_actions, record

    candidate_metrics = [path_metrics(path) for path in paths_robot]
    fidelity = np.sqrt(((paths_robot - mean_robot[None, :, :]) ** 2).sum(axis=2)).mean(axis=1)
    collision_safe_indices = [
        idx for idx, metrics in enumerate(candidate_metrics) if is_collision_safe(metrics)
    ]
    supported_safe_indices = [
        idx
        for idx in collision_safe_indices
        if candidate_metrics[idx]["support_fraction"] >= float(min_support_fraction)
    ]
    if supported_safe_indices:
        # Average over every candidate that is both collision-safe and
        # sufficiently floor-supported, instead of collapsing to a single
        # closest-to-mean sample. This keeps traj_to_actions's own
        # np.mean(all_trajectory, axis=0) (vln_utils.py:131) acting as a
        # real ensemble average over the safe subset, rather than silently
        # disabling the model's ensembling exactly when the rerank fires
        # (previously happened on every trigger, ~20% of decision points
        # in the scene10pct_seed20260715 run).
        selected_indices = supported_safe_indices
        reason = "unsafe_mean_replaced_by_supported_safe_subset_mean"
    elif collision_safe_indices:
        # Every candidate here is already collision-safe (no observed obstacle
        # contact, no backward motion) -- they only fall short of the floor-
        # support confidence threshold used above. None of them touch an
        # obstacle, so averaging across the whole subset carries none of the
        # "average of two ways around a wall crosses the wall" risk that
        # blocks doing the same thing in the fallback branch below. This used
        # to be a single best-of-subset pick, which (like the original bug)
        # disabled traj_to_actions's ensemble averaging on this branch --
        # ~30% of trigger events in the full 1839-episode / real-lookdown run.
        selected_indices = collision_safe_indices
        reason = "unsafe_mean_replaced_by_collision_safe_subset_mean"
    else:
        # No candidate is collision-safe at all here: the surviving paths can
        # disagree on which side of an obstacle to pass, so averaging them
        # (unlike the two branches above) risks producing a composite path
        # that cuts straight through what each individual candidate was
        # avoiding. Single least-risk selection is kept deliberately.
        def risk_key(idx):
            metrics = candidate_metrics[idx]
            return (
                metrics["collision_count"],
                metrics["behind_count"],
                -metrics["support_fraction"],
                float(fidelity[idx]),
            )

        best = min(range(len(candidate_metrics)), key=risk_key)
        selected_indices = [best]
        reason = "unsafe_mean_no_fully_safe_candidate_minimum_risk_fallback"

    record.update(
        {
            "status": "reranked",
            "reason": reason,
            "selected_index": selected_indices,
            "selected_count": len(selected_indices),
            "selected_mean_support_fraction": float(
                np.mean([candidate_metrics[i]["support_fraction"] for i in selected_indices])
            ),
            "selected_fidelity_m": float(np.mean(fidelity[selected_indices])),
            "safe_candidate_count": int(len(collision_safe_indices)),
            "supported_safe_candidate_count": int(len(supported_safe_indices)),
        }
    )
    return dp_actions[selected_indices], record


def _height_colors(y_values):
    physical_h = -y_values
    if len(physical_h) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo, hi = np.percentile(physical_h, [2, 98])
    if hi <= lo:
        hi = lo + 1e-3
    t = np.clip((physical_h - lo) / (hi - lo), 0.0, 1.0)
    return cv2.applyColorMap((t * 255.0).astype(np.uint8), cv2.COLORMAP_JET)[:, 0, :]


def _draw_pixel_goal_pair(image_bgr, raw_uv, corrected_uv):
    ru, rv = [int(v) for v in raw_uv]
    corrected_uv = corrected_uv if corrected_uv is not None else raw_uv
    cu, cv = [int(v) for v in corrected_uv]
    if (ru, rv) != (cu, cv):
        cv2.line(image_bgr, (ru, rv), (cu, cv), (0, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.drawMarker(image_bgr, (cu, cv), (0, 255, 0), cv2.MARKER_CROSS, 24, 3, line_type=cv2.LINE_AA)
    cv2.circle(image_bgr, (cu, cv), 10, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    cv2.putText(image_bgr, "corr", (cu + 12, cv - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(image_bgr, (ru, rv), (255, 0, 255), cv2.MARKER_CROSS, 24, 3, line_type=cv2.LINE_AA)
    cv2.circle(image_bgr, (ru, rv), 10, (255, 0, 255), 3, lineType=cv2.LINE_AA)
    cv2.putText(image_bgr, "raw", (ru + 12, rv + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)


def _add_label(image_bgr, label):
    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    image_bgr[:] = cv2.addWeighted(overlay, 0.58, image_bgr, 0.42, 0)
    cv2.putText(image_bgr, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)


def save_rgb_projected_occ_debug(
    output_dir,
    record_prefix,
    current_rgb,
    occ,
    raw_uv,
    corrected_uv,
    status,
    params,
    point_radius=1,
    large_point_radius=2,
    image_mode="all",
):
    if output_dir is None or record_prefix is None or current_rgb is None:
        return {}
    image_mode = str(image_mode)
    if image_mode == "none":
        return {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(current_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return {}
    base_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    points = occ.get("points", {})
    u = points.get("u")
    v = points.get("v")
    z = points.get("z")
    y = points.get("y")
    free_mask = points.get("free_mask")
    occupied_mask = points.get("occupied_mask")
    if u is None or v is None or z is None or y is None:
        return {}
    order = np.argsort(z)[::-1]
    colors = _height_colors(y)
    other_mask = ~(free_mask | occupied_mask)

    def draw_height(radius, suffix):
        image = base_bgr.copy()
        for idx in order:
            cv2.circle(
                image,
                (int(u[idx]), int(v[idx])),
                radius,
                tuple(int(value) for value in colors[idx].tolist()),
                -1,
                lineType=cv2.LINE_AA,
            )
        _draw_pixel_goal_pair(image, raw_uv, corrected_uv)
        label = (
            f"{record_prefix} | {status} | maxD={params['max_depth_m']} stride={params['stride']} "
            f"floorP={params['floor_percentile']} freeTol={params['free_height_tol_m']} infl={params['inflation_radius_m']}"
        )
        _add_label(image, label)
        path = out_dir / f"{record_prefix}_rgb_projected_occ_goals_{suffix}.png"
        cv2.imwrite(str(path), image)
        return str(path)

    def draw_ground_gray(radius, suffix):
        image = base_bgr.copy()
        for idx in order:
            if other_mask[idx]:
                color = (78, 78, 78)
                point_radius = max(1, radius - 1)
            elif free_mask[idx]:
                color = (205, 205, 205)
                point_radius = radius
            else:
                color = tuple(int(value) for value in colors[idx].tolist())
                point_radius = radius
            cv2.circle(image, (int(u[idx]), int(v[idx])), point_radius, color, -1, lineType=cv2.LINE_AA)
        _draw_pixel_goal_pair(image, raw_uv, corrected_uv)
        label = (
            f"{record_prefix} | {status} | ground={int(np.count_nonzero(free_mask))} "
            f"occ={int(np.count_nonzero(occupied_mask))} floor_y={occ['floor_y']:.3f} tol={params['free_height_tol_m']}"
        )
        _add_label(image, label)
        path = out_dir / f"{record_prefix}_rgb_projected_ground_gray_{suffix}.png"
        cv2.imwrite(str(path), image)
        return str(path)

    def draw_redmean_distance_transform():
        if not params.get("local_occ_robot_x_from_high_clearance", False):
            return None
        include_unknown = bool(params.get("local_occ_include_unknown_as_free", False))
        target_clearance = float(params.get("local_occ_target_clearance_px", 75.0))
        robot_pixel = params.get("robot_pixel", [rgb.shape[1] // 2, rgb.shape[0] - 1])
        free_rgb_mask = _make_rgb_free_mask(
            occ,
            rgb.shape,
            max(1, int(point_radius)),
            include_unknown=include_unknown,
        )
        if np.any(free_rgb_mask):
            dist = cv2.distanceTransform(free_rgb_mask.astype(np.uint8), cv2.DIST_L2, 5)
            values = dist[free_rgb_mask]
            threshold = max(target_clearance, float(np.percentile(values, 90.0)))
            high = free_rgb_mask & (dist >= threshold)
            if np.count_nonzero(high) < 16:
                threshold = max(target_clearance * 0.5, float(np.percentile(values, 80.0)))
                high = free_rgb_mask & (dist >= threshold)
        else:
            dist = np.zeros(free_rgb_mask.shape, dtype=np.float32)
            high = np.zeros_like(free_rgb_mask, dtype=bool)
            threshold = target_clearance

        if np.max(dist) > 1e-6:
            vis = cv2.applyColorMap(np.uint8(np.clip(dist / np.max(dist), 0.0, 1.0) * 255), cv2.COLORMAP_TURBO)
        else:
            vis = np.zeros_like(base_bgr)
        vis[~free_rgb_mask] = (18, 18, 18)

        ys_high, xs_high = np.nonzero(high)
        for px, py in zip(xs_high, ys_high):
            cv2.circle(vis, (int(px), int(py)), 1, (245, 245, 245), -1, lineType=cv2.LINE_AA)

        rb = (int(robot_pixel[0]), int(robot_pixel[1]))
        cv2.circle(vis, rb, 8, (0, 210, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(vis, "robot", (rb[0] + 9, rb[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 2)
        _draw_pixel_goal_pair(vis, raw_uv, corrected_uv)
        label = (
            f"{record_prefix} | distance transform for redmean robot-x | "
            f"white dots={int(np.count_nonzero(high))} threshold={threshold:.1f}"
        )
        _add_label(vis, label)
        path = out_dir / f"{record_prefix}_rgb_distance_transform_redmean.png"
        cv2.imwrite(str(path), vis)
        return str(path)

    result = {
        "rgb_projected_free_points": int(np.count_nonzero(free_mask)),
        "rgb_projected_occupied_points": int(np.count_nonzero(occupied_mask)),
        "rgb_projected_other_points": int(np.count_nonzero(other_mask)),
    }
    redmean_distance_path = draw_redmean_distance_transform()
    if redmean_distance_path is not None:
        result["rgb_distance_transform_redmean"] = redmean_distance_path
    if image_mode == "large_only":
        result.update(
            {
                "rgb_projected_occ_goals_large": draw_height(large_point_radius, "large_points"),
                "rgb_projected_ground_gray_large": draw_ground_gray(large_point_radius, "large_points"),
            }
        )
    else:
        result.update(
            {
                "rgb_projected_occ_goals": draw_height(point_radius, "small_points"),
                "rgb_projected_occ_goals_large": draw_height(large_point_radius, "large_points"),
                "rgb_projected_ground_gray": draw_ground_gray(point_radius, "small_points"),
                "rgb_projected_ground_gray_large": draw_ground_gray(large_point_radius, "large_points"),
            }
        )
    return result


def _last_free_on_ray(grid, start, goal):
    sx, sy = start
    gx, gy = goal
    steps = max(abs(gx - sx), abs(gy - sy), 1)
    last_free = None
    first_blocked = None
    for i in range(steps + 1):
        t = i / float(steps)
        x = int(round(sx + (gx - sx) * t))
        y = int(round(sy + (gy - sy) * t))
        if x < 0 or x >= grid.shape[1] or y < 0 or y >= grid.shape[0]:
            break
        value = int(grid[y, x])
        if value == FREE:
            last_free = (x, y)
        elif value in (UNKNOWN, OCCUPIED, INFLATED):
            first_blocked = (x, y, value)
            if last_free is not None:
                break
    return last_free, first_blocked


def _make_rgb_free_mask(occ, image_shape, radius_px, include_unknown=False):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    points = occ.get("points", {})
    u = points.get("u")
    v = points.get("v")
    free_mask = points.get("free_mask")
    occupied_mask = points.get("occupied_mask")
    if u is None or v is None or free_mask is None:
        return mask.astype(bool)
    if include_unknown and occupied_mask is not None:
        draw_mask = ~occupied_mask
    else:
        draw_mask = free_mask
    radius_px = max(1, int(radius_px))
    for px, py in zip(u[draw_mask], v[draw_mask]):
        if 0 <= int(px) < w and 0 <= int(py) < h:
            cv2.circle(mask, (int(px), int(py)), radius_px, 1, -1, lineType=cv2.LINE_AA)
    return mask.astype(bool)


def _clean_and_select_rgb_component(seed_mask, robot_uv, close_px=9, dilate_px=3):
    mask = seed_mask.astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, k, iterations=1)

    h, w = mask.shape
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return mask.astype(bool), None, {"component_label": None, "component_count": 0}

    robot = np.array([float(robot_uv[0]), float(robot_uv[1])], dtype=np.float32)
    best_label = None
    best_score = None
    for label in range(1, num):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 64:
            continue
        cx, cy = centroids[label]
        dist = np.linalg.norm(np.array([cx, cy], dtype=np.float32) - robot)
        bottom_bonus = cy / max(1.0, h)
        score = dist / np.sqrt(w * w + h * h) - 0.28 * bottom_bonus - 0.08 * np.log1p(area)
        if best_score is None or score < best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return mask.astype(bool), None, {"component_label": None, "component_count": int(num - 1)}
    component = labels == best_label
    return component, int(best_label), {
        "component_label": int(best_label),
        "component_count": int(num - 1),
        "component_pixels": int(np.count_nonzero(component)),
    }


def _first_free_on_image_ray(free_mask, raw_uv, robot_uv):
    h, w = free_mask.shape
    ru, rv = raw_uv
    bu, bv = robot_uv
    steps = max(abs(int(bu) - int(ru)), abs(int(bv) - int(rv)), 1)
    samples = []
    for i in range(steps + 1):
        t = i / float(steps)
        u = int(round(ru + (bu - ru) * t))
        v = int(round(rv + (bv - rv) * t))
        if u < 0 or u >= w or v < 0 or v >= h:
            samples.append([i, u, v, -1])
            continue
        value = int(free_mask[v, u])
        samples.append([i, u, v, value])
        if value:
            return [u, v], samples
    return None, samples


def _robot_x_from_high_clearance(free_mask, fallback_robot_uv, min_clearance_px):
    h, w = free_mask.shape
    fallback_u, fallback_v = [int(v) for v in fallback_robot_uv]
    if not np.any(free_mask):
        return [int(np.clip(fallback_u, 0, w - 1)), int(np.clip(fallback_v, 0, h - 1))], {
            "method": "fallback_no_free",
            "high_clearance_count": 0,
        }
    dist = cv2.distanceTransform(free_mask.astype(np.uint8), cv2.DIST_L2, 5)
    values = dist[free_mask]
    threshold = max(float(min_clearance_px), float(np.percentile(values, 90.0)))
    high = free_mask & (dist >= threshold)
    if np.count_nonzero(high) < 16:
        threshold = float(np.percentile(values, 90.0))
        high = free_mask & (dist >= threshold)
    if np.count_nonzero(high) < 16:
        threshold = float(np.percentile(values, 80.0))
        high = free_mask & (dist >= threshold)
    if np.count_nonzero(high) == 0:
        return [int(np.clip(fallback_u, 0, w - 1)), int(np.clip(fallback_v, 0, h - 1))], {
            "method": "fallback_no_high_clearance",
            "high_clearance_count": 0,
            "high_clearance_threshold_px": float(threshold),
        }
    xs = np.nonzero(high)[1]
    robot_u = int(round(float(xs.mean())))
    return [int(np.clip(robot_u, 0, w - 1)), int(np.clip(fallback_v, 0, h - 1))], {
        "method": "mean_high_clearance_x",
        "high_clearance_count": int(np.count_nonzero(high)),
        "high_clearance_threshold_px": float(threshold),
        "fallback_robot_pixel": [int(fallback_u), int(fallback_v)],
        "min_x": int(xs.min()),
        "max_x": int(xs.max()),
        "mean_x": float(xs.mean()),
    }


def _ray_clearance_band_goal(
    free_mask,
    raw_uv,
    robot_uv,
    target_clearance_px,
    clearance_band_px,
    max_shift_px,
    strict_band=False,
):
    h, w = free_mask.shape
    ru, rv = [int(v) for v in raw_uv]
    bu, bv = [int(v) for v in robot_uv]
    dist = cv2.distanceTransform(free_mask.astype(np.uint8), cv2.DIST_L2, 5)
    steps = max(abs(bu - ru), abs(bv - rv), 1)
    target = float(target_clearance_px)
    band = float(clearance_band_px)
    max_shift = float(max_shift_px)
    samples = []
    best = None
    best_score = None
    best_in_or_above_band = None
    best_in_or_above_band_score = None

    for i in range(steps + 1):
        t = i / float(steps)
        u = int(round(ru + (bu - ru) * t))
        v = int(round(rv + (bv - rv) * t))
        if u < 0 or u >= w or v < 0 or v >= h or not free_mask[v, u]:
            continue
        clearance = float(dist[v, u])
        raw_distance = float(np.hypot(u - ru, v - rv))
        if raw_distance > max_shift:
            continue
        score = abs(clearance - target) / max(1.0, band) + 0.08 * raw_distance / max(1.0, max_shift)
        sample = [int(i), int(u), int(v), float(clearance), float(raw_distance), float(score)]
        samples.append(sample)
        if target - band <= clearance <= target + band:
            return [u, v], {
                "reason": "ray_clearance_band_first_hit",
                "target_clearance_px": target,
                "clearance_band_px": band,
                "clearance_px": clearance,
                "raw_distance_px": raw_distance,
                "candidate_count": len(samples),
                "ray_clearance_samples": samples,
            }
        if clearance >= target - band:
            high_score = abs(clearance - target) / max(1.0, band) + 0.08 * raw_distance / max(1.0, max_shift)
            if best_in_or_above_band_score is None or high_score < best_in_or_above_band_score:
                best_in_or_above_band_score = high_score
                best_in_or_above_band = [u, v, clearance, raw_distance, high_score]
        if best_score is None or score < best_score:
            best_score = score
            best = [u, v, clearance, raw_distance, score]

    if best is None:
        return None, {
            "reason": "ray_clearance_band_no_candidate",
            "target_clearance_px": target,
            "clearance_band_px": band,
            "candidate_count": len(samples),
            "ray_clearance_samples": samples,
        }
    if strict_band:
        if best_in_or_above_band is None:
            return None, {
                "reason": "ray_clearance_band_no_yellow_green_candidate",
                "target_clearance_px": target,
                "clearance_band_px": band,
                "candidate_count": len(samples),
                "ray_clearance_samples": samples,
            }
        best = best_in_or_above_band
        return [int(best[0]), int(best[1])], {
            "reason": "ray_clearance_band_strict_nearest_above_lower",
            "target_clearance_px": target,
            "clearance_band_px": band,
            "clearance_px": float(best[2]),
            "raw_distance_px": float(best[3]),
            "score": float(best[4]),
            "candidate_count": len(samples),
            "ray_clearance_samples": samples,
        }
    return [int(best[0]), int(best[1])], {
        "reason": "ray_clearance_band_nearest",
        "target_clearance_px": target,
        "clearance_band_px": band,
        "clearance_px": float(best[2]),
        "raw_distance_px": float(best[3]),
        "score": float(best[4]),
        "candidate_count": len(samples),
        "ray_clearance_samples": samples,
    }


def _vertical_clearance_band_goal(
    free_mask,
    raw_uv,
    robot_uv,
    target_clearance_px,
    clearance_band_px,
    max_shift_px,
):
    """Like _ray_clearance_band_goal, but only ever moves along the raw
    prediction's own column (u fixed), sweeping v toward the robot's row.

    _ray_clearance_band_goal interpolates between robot_uv and raw_uv, which
    are generally different columns -- so its "same ray" framing is only
    approximate; the search direction actually drifts sideways as it looks
    for a nearer candidate. For a roughly planar floor with a fixed camera
    pitch, a fixed column corresponds much more closely to "same bearing" in
    the real world, with v acting as a proxy for distance along that bearing
    (larger v = lower in the image = nearer the camera/floor). This keeps
    the correction from silently changing the direction the model actually
    pointed at, which the diagonal-ray version does not guarantee.

    Known limitation: this column-is-bearing / row-is-distance approximation
    assumes a flat floor and stable camera pitch. It degrades on stairs/
    multi-level scenes, where a fixed column does not correspond to a
    constant real-world bearing as v changes.
    """
    h, w = free_mask.shape
    ru, rv = [int(v) for v in raw_uv]
    _, bv = [int(v) for v in robot_uv]
    if ru < 0 or ru >= w:
        return None, {"reason": "vertical_clearance_band_column_out_of_bounds", "candidate_count": 0}

    dist = cv2.distanceTransform(free_mask.astype(np.uint8), cv2.DIST_L2, 5)
    steps = max(abs(bv - rv), 1)
    target = float(target_clearance_px)
    band = float(clearance_band_px)
    max_shift = float(max_shift_px)
    samples = []
    best = None
    best_score = None

    for i in range(steps + 1):
        t = i / float(steps)
        v = int(round(rv + (bv - rv) * t))
        if v < 0 or v >= h or not free_mask[v, ru]:
            continue
        clearance = float(dist[v, ru])
        raw_distance = float(abs(v - rv))
        if raw_distance > max_shift:
            continue
        score = abs(clearance - target) / max(1.0, band) + 0.08 * raw_distance / max(1.0, max_shift)
        sample = [int(i), int(ru), int(v), float(clearance), float(raw_distance), float(score)]
        samples.append(sample)
        if target - band <= clearance <= target + band:
            return [ru, v], {
                "reason": "vertical_clearance_band_first_hit",
                "target_clearance_px": target,
                "clearance_band_px": band,
                "clearance_px": clearance,
                "raw_distance_px": raw_distance,
                "candidate_count": len(samples),
                "vertical_clearance_samples": samples,
            }
        if best_score is None or score < best_score:
            best_score = score
            best = [ru, v, clearance, raw_distance, score]

    if best is None:
        return None, {
            "reason": "vertical_clearance_band_no_candidate",
            "target_clearance_px": target,
            "clearance_band_px": band,
            "candidate_count": len(samples),
            "vertical_clearance_samples": samples,
        }
    return [int(best[0]), int(best[1])], {
        "reason": "vertical_clearance_band_nearest",
        "target_clearance_px": target,
        "clearance_band_px": band,
        "clearance_px": float(best[2]),
        "raw_distance_px": float(best[3]),
        "score": float(best[4]),
        "candidate_count": len(samples),
        "vertical_clearance_samples": samples,
    }


def _vertical_nearest_free_goal(free_mask, raw_uv, robot_uv, max_shift_px):
    """Same column-fixed / row-only search direction as
    _vertical_clearance_band_goal, but with no target clearance to hit --
    just returns the first (i.e. nearest to the raw prediction) walkable
    pixel found while sweeping from raw_uv's row toward the robot's row on
    the raw prediction's own column.
    """
    h, w = free_mask.shape
    ru, rv = [int(v) for v in raw_uv]
    _, bv = [int(v) for v in robot_uv]
    if ru < 0 or ru >= w:
        return None, {"reason": "vertical_nearest_free_column_out_of_bounds", "candidate_count": 0}

    steps = max(abs(bv - rv), 1)
    max_shift = float(max_shift_px)
    samples = []

    for i in range(steps + 1):
        t = i / float(steps)
        v = int(round(rv + (bv - rv) * t))
        raw_distance = float(abs(v - rv))
        if raw_distance > max_shift:
            # raw_distance is monotonically non-decreasing in i (v moves
            # monotonically from rv toward bv), so nothing further along
            # the sweep can come back within budget.
            break
        if v < 0 or v >= h:
            continue
        samples.append([int(i), int(ru), int(v), raw_distance])
        if free_mask[v, ru]:
            return [ru, v], {
                "reason": "vertical_nearest_free_first_hit",
                "raw_distance_px": raw_distance,
                "candidate_count": len(samples),
                "vertical_nearest_free_samples": samples,
            }

    return None, {
        "reason": "vertical_nearest_free_no_candidate",
        "candidate_count": len(samples),
        "vertical_nearest_free_samples": samples,
    }


def save_traversability_preview(path, grid, raw_grid=None, corrected_grid=None):
    colors = np.array(
        [
            [35, 35, 35],
            [232, 232, 232],
            [220, 45, 35],
            [245, 160, 35],
        ],
        dtype=np.uint8,
    )
    image = colors[np.clip(grid, 0, len(colors) - 1)]
    image = np.flipud(image)
    image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    def draw_grid_point(grid_xy, color):
        if grid_xy is None:
            return
        gx, gy = grid_xy
        px = int(gx * 4 + 2)
        py = int((grid.shape[0] - 1 - gy) * 4 + 2)
        cv2.circle(image, (px, py), 5, color, -1)

    draw_grid_point(raw_grid, (255, 0, 255))
    draw_grid_point(corrected_grid, (0, 255, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def correct_pixel_goal_with_local_occ(
    depth_m,
    pixel_goal,
    fx,
    fy,
    cx,
    cy,
    current_rgb=None,
    output_dir=None,
    record_prefix=None,
    resolution_m=0.05,
    max_depth_m=5.0,
    stride=4,
    inflation_radius_m=0.20,
    floor_percentile=82.0,
    free_height_tol_m=0.18,
    occupied_height_m=0.22,
    rgb_point_radius_px=1,
    correction_mode="topdown_ray",
    robot_pixel=None,
    debug_image_mode="all",
    local_occ_include_unknown_as_free=False,
    local_occ_target_clearance_px=75.0,
    local_occ_clearance_band_px=12.0,
    local_occ_max_shift_px=220.0,
    local_occ_robot_x_from_high_clearance=False,
):
    h, w = depth_m.shape
    # BUG FIX (verified 2026-07-23, pixel-perfect against occ_goal_debug preview
    # images): the caller (habitat_vln_evaluator.py) passes `pixel_goal` as
    # [v, u] i.e. [row, col] -- confirmed empirically (first number in the
    # model's raw "A B" text output reaches 593, which is impossible for a
    # <=480-tall row index, so it must be the width-bounded column; the
    # evaluator's `pixel_goal = [coord[1], coord[0]]` then puts that column
    # value second). This function assumed [u, v] i.e. [col, row], so raw_u/
    # raw_v were transposed (and raw_v silently clipped to h-1 whenever the
    # true column exceeded 479) for every call site in this project (Method5,
    # Method7, redmean75, and their "latent" variants). See
    # Simulation_evaluation/notes/experiment_log.md ("Latent 重灌实验" section)
    # for the pixel-exact verification (occ_goal_debug marker landed at
    # exactly (423, 479) -- 479 being h-1, the clipped column value).
    raw_u = int(np.clip(pixel_goal[1], 0, w - 1))
    raw_v = int(np.clip(pixel_goal[0], 0, h - 1))
    occ = build_local_traversability_from_depth(
        depth_m,
        fx,
        fy,
        cx,
        cy,
        resolution_m=resolution_m,
        max_depth_m=max_depth_m,
        stride=stride,
        inflation_radius_m=inflation_radius_m,
        floor_percentile=floor_percentile,
        free_height_tol_m=free_height_tol_m,
        occupied_height_m=occupied_height_m,
    )
    if occ is None:
        return {
            "status": "unknown",
            "reason": "insufficient_depth_for_occ",
            "raw_pixel_goal": [raw_u, raw_v],
            "corrected_pixel_goal": [raw_u, raw_v],
            "used_corrected_goal": False,
        }

    z = _nearest_valid_depth(depth_m, raw_u, raw_v)
    if z is None:
        return {
            "status": "unknown",
            "reason": "no_depth_at_pixel_goal",
            "raw_pixel_goal": [raw_u, raw_v],
            "corrected_pixel_goal": [raw_u, raw_v],
            "used_corrected_goal": False,
        }

    raw_point = _pixel_to_camera(raw_u, raw_v, z, fx, fy, cx, cy)
    grid = occ["grid"]
    raw_grid = _grid_index(
        raw_point[0],
        raw_point[2],
        occ["origin_xz"],
        occ["resolution_m"],
        grid.shape[1],
        grid.shape[0],
    )
    robot_grid = _grid_index(
        0.0,
        0.0,
        occ["origin_xz"],
        occ["resolution_m"],
        grid.shape[1],
        grid.shape[0],
    )
    raw_value = int(grid[raw_grid[1], raw_grid[0]]) if raw_grid is not None else None
    first_blocked = None
    image_ray_samples = []
    rgb_free_mask = None
    robot_pixel = robot_pixel if robot_pixel is not None else [w // 2, h - 1]

    arrive_gate = False
    corrected_depth_m = None

    if correction_mode in (
        "rgb_bottom_ray",
        "rgb_ray_clearance_band",
        "rgb_ray_clearance_band_strict",
        "rgb_ray_clearance_band_harmed100",
        "rgb_ray_clearance_band_method5_new",
        "vertical_clearance_band_method5_arrive",
        "vertical_nearest_free_method5_arrive",
    ):
        rgb_seed_mask = _make_rgb_free_mask(
            occ,
            depth_m.shape,
            rgb_point_radius_px,
            include_unknown=local_occ_include_unknown_as_free,
        )
        if correction_mode in (
            "rgb_ray_clearance_band_harmed100",
            "rgb_ray_clearance_band_method5_new",
            "vertical_clearance_band_method5_arrive",
            "vertical_nearest_free_method5_arrive",
        ):
            rgb_free_mask, component_label, component_info = _clean_and_select_rgb_component(
                rgb_seed_mask,
                robot_pixel,
                close_px=9,
                dilate_px=3,
            )
        else:
            rgb_free_mask = rgb_seed_mask
            component_info = None
        robot_pixel_info = None
        if local_occ_robot_x_from_high_clearance:
            robot_pixel, robot_pixel_info = _robot_x_from_high_clearance(
                rgb_free_mask,
                robot_pixel,
                local_occ_target_clearance_px,
            )
        raw_is_free = bool(rgb_free_mask[raw_v, raw_u])
        corrected_grid = raw_grid
        if raw_is_free:
            status = "valid"
            corrected_pixel = [raw_u, raw_v]
            reason = "raw_goal_in_rgb_free_mask"
            used = False
        elif correction_mode == "vertical_clearance_band_method5_arrive":
            # method_5_arrive: keep the raw prediction's own column fixed (does
            # not drift sideways the way the diagonal ray search does -- see
            # _vertical_clearance_band_goal's docstring) and only search for a
            # walkable distance along that column. If no such point exists at
            # all, fall back to the original diagonal ray search rather than
            # declaring the goal invalid -- but only the vertical-search
            # success case sets arrive_gate, since only that case is
            # guaranteed to preserve the model's own predicted direction.
            corrected_pixel, vert_info = _vertical_clearance_band_goal(
                rgb_free_mask,
                [raw_u, raw_v],
                [int(robot_pixel[0]), int(robot_pixel[1])],
                local_occ_target_clearance_px,
                local_occ_clearance_band_px,
                local_occ_max_shift_px,
            )
            if component_info is not None:
                vert_info["rgb_component_info"] = component_info
            image_ray_samples = vert_info.get("vertical_clearance_samples", [])
            if corrected_pixel is not None:
                status = "corrected"
                reason = vert_info["reason"]
                used = corrected_pixel != [raw_u, raw_v]
                arrive_gate = used
            else:
                fallback_pixel, ray_info = _ray_clearance_band_goal(
                    rgb_free_mask,
                    [raw_u, raw_v],
                    [int(robot_pixel[0]), int(robot_pixel[1])],
                    local_occ_target_clearance_px,
                    local_occ_clearance_band_px,
                    local_occ_max_shift_px,
                    strict_band=False,
                )
                ray_info["vertical_search_reason"] = vert_info["reason"]
                if component_info is not None:
                    ray_info["rgb_component_info"] = component_info
                image_ray_samples = ray_info.get("ray_clearance_samples", [])
                if fallback_pixel is None:
                    status = "invalid"
                    corrected_pixel = [raw_u, raw_v]
                    reason = f"vertical_no_candidate_fallback_{ray_info['reason']}"
                    used = False
                else:
                    status = "corrected"
                    corrected_pixel = fallback_pixel
                    reason = f"vertical_no_candidate_fallback_{ray_info['reason']}"
                    used = corrected_pixel != [raw_u, raw_v]
                    # arrive_gate deliberately left False: this is the old
                    # diagonal-ray method's own point, which does not carry
                    # the same "direction preserved" guarantee.
        elif correction_mode == "vertical_nearest_free_method5_arrive":
            # Same fixed-column / row-only search direction as
            # vertical_clearance_band_method5_arrive, but instead of hunting
            # for a specific target clearance, just walk to the nearest
            # walkable point on that column. Falls back to the same
            # unmodified diagonal-ray method when no vertical candidate
            # exists (identical to the clearance-band variant above).
            corrected_pixel, vert_info = _vertical_nearest_free_goal(
                rgb_free_mask,
                [raw_u, raw_v],
                [int(robot_pixel[0]), int(robot_pixel[1])],
                local_occ_max_shift_px,
            )
            if component_info is not None:
                vert_info["rgb_component_info"] = component_info
            image_ray_samples = vert_info.get("vertical_nearest_free_samples", [])
            if corrected_pixel is not None:
                status = "corrected"
                reason = vert_info["reason"]
                used = corrected_pixel != [raw_u, raw_v]
                arrive_gate = used
            else:
                fallback_pixel, ray_info = _ray_clearance_band_goal(
                    rgb_free_mask,
                    [raw_u, raw_v],
                    [int(robot_pixel[0]), int(robot_pixel[1])],
                    local_occ_target_clearance_px,
                    local_occ_clearance_band_px,
                    local_occ_max_shift_px,
                    strict_band=False,
                )
                ray_info["vertical_search_reason"] = vert_info["reason"]
                if component_info is not None:
                    ray_info["rgb_component_info"] = component_info
                image_ray_samples = ray_info.get("ray_clearance_samples", [])
                if fallback_pixel is None:
                    status = "invalid"
                    corrected_pixel = [raw_u, raw_v]
                    reason = f"vertical_no_candidate_fallback_{ray_info['reason']}"
                    used = False
                else:
                    status = "corrected"
                    corrected_pixel = fallback_pixel
                    reason = f"vertical_no_candidate_fallback_{ray_info['reason']}"
                    used = corrected_pixel != [raw_u, raw_v]
                    # arrive_gate deliberately left False: same rationale as
                    # the clearance-band variant's fallback case.
        elif correction_mode in (
            "rgb_ray_clearance_band",
            "rgb_ray_clearance_band_strict",
            "rgb_ray_clearance_band_harmed100",
            "rgb_ray_clearance_band_method5_new",
        ):
            corrected_pixel, ray_info = _ray_clearance_band_goal(
                rgb_free_mask,
                [raw_u, raw_v],
                [int(robot_pixel[0]), int(robot_pixel[1])],
                local_occ_target_clearance_px,
                local_occ_clearance_band_px,
                local_occ_max_shift_px,
                strict_band=correction_mode == "rgb_ray_clearance_band_strict",
            )
            if component_info is not None:
                ray_info["rgb_component_info"] = component_info
            image_ray_samples = ray_info.get("ray_clearance_samples", [])
            if corrected_pixel is None:
                status = "invalid"
                corrected_pixel = [raw_u, raw_v]
                reason = ray_info["reason"]
                used = False
            else:
                status = "corrected"
                reason = ray_info["reason"]
                used = corrected_pixel != [raw_u, raw_v]
        else:
            corrected_pixel, image_ray_samples = _first_free_on_image_ray(
                rgb_free_mask,
                [raw_u, raw_v],
                [int(robot_pixel[0]), int(robot_pixel[1])],
            )
            if corrected_pixel is None:
                status = "invalid"
                corrected_pixel = [raw_u, raw_v]
                reason = "no_rgb_free_pixel_on_raw_to_bottom_ray"
                used = False
            else:
                status = "corrected"
                reason = "clipped_to_first_rgb_free_on_raw_to_bottom_ray"
                used = corrected_pixel != [raw_u, raw_v]
        if robot_pixel_info is not None:
            image_ray_samples = {
                "robot_pixel_info": robot_pixel_info,
                "samples": image_ray_samples,
            }
            if component_info is not None:
                image_ray_samples["rgb_component_info"] = component_info
    elif raw_grid is None or robot_grid is None:
        status = "invalid"
        corrected_pixel = [raw_u, raw_v]
        corrected_grid = None
        reason = "goal_or_robot_outside_local_occ"
        used = False
    elif raw_value == FREE:
        status = "valid"
        corrected_pixel = [raw_u, raw_v]
        corrected_grid = raw_grid
        reason = "raw_goal_in_free"
        used = False
    else:
        corrected_grid, first_blocked = _last_free_on_ray(grid, robot_grid, raw_grid)
        if corrected_grid is None:
            status = "invalid"
            corrected_pixel = [raw_u, raw_v]
            reason = "no_free_cell_on_robot_goal_ray"
            used = False
        else:
            cx_grid, cz_grid = corrected_grid
            x = occ["origin_xz"][0] + (cx_grid + 0.5) * occ["resolution_m"]
            z_corr = occ["origin_xz"][1] + (cz_grid + 0.5) * occ["resolution_m"]
            corrected_point = np.array([x, occ["floor_y"], z_corr], dtype=np.float32)
            projected = _camera_to_pixel(corrected_point, fx, fy, cx, cy, w, h)
            corrected_pixel = projected if projected is not None else [raw_u, raw_v]
            status = "corrected" if projected is not None else "invalid"
            reason = "clipped_to_last_free_on_robot_goal_ray"
            used = projected is not None

    if used and corrected_pixel is not None:
        corrected_depth_m = _nearest_valid_depth(depth_m, int(corrected_pixel[0]), int(corrected_pixel[1]))

    preview_file = None
    debug_images = {}
    params = {
        "resolution_m": float(resolution_m),
        "max_depth_m": float(max_depth_m),
        "stride": int(stride),
        "inflation_radius_m": float(inflation_radius_m),
        "floor_percentile": float(floor_percentile),
        "free_height_tol_m": float(free_height_tol_m),
        "occupied_height_m": float(occupied_height_m),
        "correction_mode": str(correction_mode),
        "robot_pixel": [int(robot_pixel[0]), int(robot_pixel[1])],
        "local_occ_include_unknown_as_free": bool(local_occ_include_unknown_as_free),
        "local_occ_target_clearance_px": float(local_occ_target_clearance_px),
        "local_occ_clearance_band_px": float(local_occ_clearance_band_px),
        "local_occ_max_shift_px": float(local_occ_max_shift_px),
        "local_occ_robot_x_from_high_clearance": bool(local_occ_robot_x_from_high_clearance),
    }
    if isinstance(image_ray_samples, dict) and image_ray_samples.get("robot_pixel_info") is not None:
        params["robot_pixel_info"] = image_ray_samples["robot_pixel_info"]
        if image_ray_samples.get("rgb_component_info") is not None:
            params["rgb_component_info"] = image_ray_samples["rgb_component_info"]
    if output_dir is not None and record_prefix is not None and str(debug_image_mode) != "none":
        if str(debug_image_mode) == "all":
            preview_file = str(Path(output_dir) / f"{record_prefix}_local_occ.png")
            save_traversability_preview(preview_file, grid, raw_grid=raw_grid, corrected_grid=corrected_grid)
        debug_images = save_rgb_projected_occ_debug(
            output_dir=output_dir,
            record_prefix=record_prefix,
            current_rgb=current_rgb,
            occ=occ,
            raw_uv=[raw_u, raw_v],
            corrected_uv=corrected_pixel,
            status=status,
            params=params,
            point_radius=max(1, int(rgb_point_radius_px)),
            large_point_radius=max(2, int(rgb_point_radius_px) + 1),
            image_mode=debug_image_mode,
        )

    return {
        "status": status,
        "reason": reason,
        "raw_pixel_goal": [raw_u, raw_v],
        "corrected_pixel_goal": corrected_pixel,
        "used_corrected_goal": bool(used),
        "raw_grid": list(raw_grid) if raw_grid is not None else None,
        "robot_grid": list(robot_grid) if robot_grid is not None else None,
        "corrected_grid": list(corrected_grid) if corrected_grid is not None else None,
        "raw_grid_value": raw_value,
        "first_blocked": list(first_blocked) if first_blocked is not None else None,
        "image_ray_samples": image_ray_samples,
        "rgb_free_at_raw": bool(rgb_free_mask[raw_v, raw_u]) if rgb_free_mask is not None else None,
        "preview_file": preview_file,
        "local_occ_params": params,
        "floor_y": float(occ["floor_y"]),
        "arrive_gate": bool(arrive_gate),
        "corrected_depth_m": float(corrected_depth_m) if corrected_depth_m is not None else None,
        **debug_images,
    }


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
