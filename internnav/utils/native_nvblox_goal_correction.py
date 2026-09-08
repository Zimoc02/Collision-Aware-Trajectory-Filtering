"""Native nvblox current-map pixel-goal correction for Habitat eval debug runs."""

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

try:
    import open3d as o3d
except Exception:  # pragma: no cover - optional until native nvblox mode is enabled.
    o3d = None


FREE = 1
BLOCKING = {2, 3}


def _run(cmd, env=None):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(map(str, cmd)) + "\n" + result.stdout)
    return result.stdout


def _nearest_valid_depth(depth_m, u, v, radius=5):
    h, w = depth_m.shape
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0.05)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _pixel_to_camera(u, v, z, fx, fy, cx, cy):
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z, 1.0], dtype=np.float32)


def _camera_to_pixel(point_c, fx, fy, cx, cy, width, height):
    x, y, z = [float(v) for v in point_c[:3]]
    if z <= 1e-4:
        return None
    u = int(round(fx * x / z + cx))
    v = int(round(fy * y / z + cy))
    if 0 <= u < width and 0 <= v < height:
        return [u, v]
    return None


def _world_to_grid_xy(point_w, meta):
    origin = np.asarray(meta["origin_xy"], dtype=np.float32)
    res = float(meta["resolution_m"])
    gx = int(np.floor((float(point_w[0]) - origin[0]) / res))
    gy = int(np.floor((float(point_w[1]) - origin[1]) / res))
    if gx < 0 or gy < 0 or gx >= int(meta["width"]) or gy >= int(meta["height"]):
        return None
    return [gx, gy]


def _grid_to_world_xy(grid_xy, meta, z=0.0):
    origin = np.asarray(meta["origin_xy"], dtype=np.float32)
    res = float(meta["resolution_m"])
    x = origin[0] + (grid_xy[0] + 0.5) * res
    y = origin[1] + (grid_xy[1] + 0.5) * res
    return np.array([x, y, z, 1.0], dtype=np.float32)


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
        if x < 0 or y < 0 or x >= grid.shape[1] or y >= grid.shape[0]:
            break
        value = int(grid[y, x])
        if value == FREE:
            last_free = [x, y]
        elif value in BLOCKING or value == 0:
            first_blocked = [x, y, value]
            if last_free is not None:
                break
    return last_free, first_blocked


def _nearest_free_cell(grid, center, radius_cells):
    cx, cy = center
    h, w = grid.shape
    x0, x1 = max(0, cx - radius_cells), min(w - 1, cx + radius_cells)
    y0, y1 = max(0, cy - radius_cells), min(h - 1, cy + radius_cells)
    candidates = []
    radius_sq = radius_cells * radius_cells
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if int(grid[y, x]) != FREE:
                continue
            dist_sq = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if dist_sq <= radius_sq:
                candidates.append((dist_sq, x, y))
    if not candidates:
        return None, None
    dist_sq, x, y = min(candidates, key=lambda item: item[0])
    return [x, y], float(np.sqrt(dist_sq))


def _load_ply(path):
    if o3d is None:
        raise RuntimeError("open3d is required for native nvblox RGB-mask correction")
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    colors = np.asarray(cloud.colors, dtype=np.float32)
    if len(colors) != len(points):
        colors = np.tile(np.array([[0.65, 0.65, 0.65]], dtype=np.float32), (len(points), 1))
    return points, np.clip(colors * 255.0, 0, 255).astype(np.uint8)


def _project_points(points_w, colors, t_w_c, intr, width, height, max_depth_m):
    t_c_w = np.linalg.inv(t_w_c)
    homo = np.concatenate([points_w, np.ones((len(points_w), 1), dtype=np.float32)], axis=1)
    points_c = (t_c_w @ homo.T).T[:, :3]
    z = points_c[:, 2]
    valid = np.isfinite(z) & (z > 0.05) & (z <= max_depth_m)
    points_c = points_c[valid]
    colors = colors[valid]
    z = z[valid]
    fx, fy, cx, cy = intr
    u = np.round(fx * points_c[:, 0] / z + cx).astype(np.int32)
    v = np.round(fy * points_c[:, 1] / z + cy).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside], v[inside], z[inside], colors[inside]


def _zbuffer_projection(u, v, z, colors, width, height):
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    order = np.argsort(z)
    for idx in order:
        x, y = int(u[idx]), int(v[idx])
        if z[idx] < zbuf[y, x]:
            zbuf[y, x] = z[idx]
            image[y, x] = colors[idx]
    mask = np.isfinite(zbuf)
    return image, zbuf, mask


def _expand_points(image, mask, radius):
    if radius <= 0:
        return image, mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    expanded = image.copy()
    for channel in range(3):
        expanded[:, :, channel] = cv2.dilate(image[:, :, channel], kernel, iterations=1)
    return expanded, expanded_mask


def _project_cloud_to_image_mask(ply_path, t_w_c, intr, width, height, depth_m, max_depth_m, occlusion_tol_m, radius_px):
    points, colors = _load_ply(ply_path)
    u, v, z, colors = _project_points(points, colors, t_w_c, intr, width, height, max_depth_m)
    image, zbuf, mask = _zbuffer_projection(u, v, z, colors, width, height)
    valid_depth = np.isfinite(depth_m) & (depth_m > 0.05)
    mask = mask & valid_depth & (zbuf <= depth_m + occlusion_tol_m)
    image = np.where(mask[:, :, None], image, 0).astype(np.uint8)
    return _expand_points(image, mask, radius_px)


def _depth_discontinuity_mask(depth_m, threshold_m=0.35):
    valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 9.95)
    depth = np.where(valid, depth_m, 0.0).astype(np.float32)
    dx = np.zeros_like(depth, dtype=np.float32)
    dy = np.zeros_like(depth, dtype=np.float32)
    dx[:, :-1] = np.abs(depth[:, 1:] - depth[:, :-1])
    dy[:-1, :] = np.abs(depth[1:, :] - depth[:-1, :])
    valid_x = np.zeros_like(valid)
    valid_y = np.zeros_like(valid)
    valid_x[:, :-1] = valid[:, 1:] & valid[:, :-1]
    valid_y[:-1, :] = valid[1:, :] & valid[:-1, :]
    edge = ((dx > threshold_m) & valid_x) | ((dy > threshold_m) & valid_y) | (~valid)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(edge.astype(np.uint8), kernel, iterations=1).astype(bool)


def _filled_visible_free_mask(
    raw_free_mask,
    obstacle_mask,
    depth_m,
    close_kernel=31,
    close_iterations=2,
    dilate_kernel=21,
    obstacle_dilate_kernel=9,
    depth_edge_threshold_m=0.35,
):
    height, width = raw_free_mask.shape
    raw_u8 = raw_free_mask.astype(np.uint8) * 255
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(raw_u8, cv2.MORPH_CLOSE, close_k, iterations=close_iterations) > 0
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    morphed = cv2.dilate(closed.astype(np.uint8), dilate_k, iterations=1).astype(bool)

    hull_mask = np.zeros((height, width), dtype=np.uint8)
    ys, xs = np.nonzero(morphed)
    if len(xs) > 8:
        points = np.stack([xs, ys], axis=1).astype(np.int32)
        keep = points[:, 1] > int(height * 0.28)
        if np.count_nonzero(keep) > 8:
            hull = cv2.convexHull(points[keep])
            cv2.fillConvexPoly(hull_mask, hull, 255)
    hull_mask = cv2.erode(
        hull_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    ).astype(bool)

    near_morphed = cv2.dilate(
        morphed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)),
        iterations=1,
    ).astype(bool)
    depth_edges = _depth_discontinuity_mask(depth_m, depth_edge_threshold_m)
    obstacle_block = cv2.dilate(
        obstacle_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (obstacle_dilate_kernel, obstacle_dilate_kernel)),
        iterations=1,
    ).astype(bool)
    blocking = obstacle_block | depth_edges
    candidate = (morphed | (hull_mask & near_morphed)) & (~blocking)
    candidate[: int(height * 0.22), :] &= raw_free_mask[: int(height * 0.22), :]

    seed0 = np.array([width // 2, height - 28], dtype=np.int32)
    yy, xx = np.nonzero(candidate)
    seed = None
    if len(xx) > 0:
        weights = (xx - seed0[0]) * (xx - seed0[0]) + 2.5 * (yy - seed0[1]) * (yy - seed0[1])
        lower = yy > int(height * 0.45)
        if np.any(lower):
            lower_indices = np.flatnonzero(lower)
            best = int(lower_indices[int(np.argmin(weights[lower]))])
        else:
            best = int(np.argmin(weights))
        seed = [int(xx[best]), int(yy[best])]

    if seed is None:
        filled = candidate
    else:
        flood = candidate.astype(np.uint8) * 255
        mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(flood, mask, tuple(seed), 128)
        filled = flood == 128

    debug = {
        "morphed_mask": morphed,
        "depth_edge_mask": depth_edges,
        "obstacle_block_mask": obstacle_block,
        "candidate_mask": candidate,
        "seed_pixel": seed,
    }
    return filled, debug


def _save_rgbmask_filled_debug(goal_dir, rgb, raw_uv, old_mask, filled_mask, debug_masks, old_uv=None, new_uv=None):
    debug_dir = Path(goal_dir) / "rgbmask_filled_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    def overlay(mask, color, alpha=0.55):
        image = np.asarray(rgb).copy().astype(np.float32)
        image[mask] = (1.0 - alpha) * image[mask] + alpha * np.asarray(color, dtype=np.float32)
        return np.clip(image, 0, 255).astype(np.uint8)

    def draw_points(image):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        raw = tuple(int(v) for v in raw_uv)
        cv2.circle(image, raw, 9, (255, 0, 255), 2)
        cv2.drawMarker(image, raw, (255, 0, 255), cv2.MARKER_CROSS, 22, 2)
        if old_uv is not None:
            old = tuple(int(v) for v in old_uv)
            cv2.circle(image, old, 8, (255, 255, 0), 2)
            cv2.drawMarker(image, old, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.line(image, raw, old, (255, 255, 0), 2)
        if new_uv is not None:
            new = tuple(int(v) for v in new_uv)
            cv2.circle(image, new, 8, (0, 255, 0), 2)
            cv2.drawMarker(image, new, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.line(image, raw, new, (0, 255, 0), 2)
        return image

    cv2.imwrite(str(debug_dir / "01_raw_free_overlay.png"), draw_points(overlay(old_mask, (210, 210, 210), 0.65)))
    cv2.imwrite(
        str(debug_dir / "02_morph_close_dilate_overlay.png"),
        draw_points(overlay(debug_masks["morphed_mask"], (180, 230, 255), 0.55)),
    )
    boundary = np.asarray(rgb).copy().astype(np.float32)
    boundary[debug_masks["depth_edge_mask"]] = (
        0.45 * boundary[debug_masks["depth_edge_mask"]] + 0.55 * np.array([0, 80, 255])
    )
    boundary[debug_masks["obstacle_block_mask"]] = (
        0.35 * boundary[debug_masks["obstacle_block_mask"]] + 0.65 * np.array([255, 40, 20])
    )
    cv2.imwrite(str(debug_dir / "03_obstacle_depth_boundaries.png"), draw_points(np.clip(boundary, 0, 255).astype(np.uint8)))
    cv2.imwrite(str(debug_dir / "04_flood_filled_overlay.png"), draw_points(overlay(filled_mask, (120, 255, 120), 0.55)))
    cv2.imwrite(str(debug_dir / "raw_free_mask.png"), old_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(debug_dir / "morphed_mask.png"), debug_masks["morphed_mask"].astype(np.uint8) * 255)
    cv2.imwrite(str(debug_dir / "boundary_mask.png"), (debug_masks["depth_edge_mask"] | debug_masks["obstacle_block_mask"]).astype(np.uint8) * 255)
    cv2.imwrite(str(debug_dir / "filled_mask.png"), filled_mask.astype(np.uint8) * 255)
    return debug_dir


def _nearest_mask_pixel(mask, raw_uv, radius_px):
    raw_u, raw_v = [int(v) for v in raw_uv]
    h, w = mask.shape
    raw_u = int(np.clip(raw_u, 0, w - 1))
    raw_v = int(np.clip(raw_v, 0, h - 1))
    if bool(mask[raw_v, raw_u]):
        return [raw_u, raw_v], 0.0
    x0, x1 = max(0, raw_u - radius_px), min(w - 1, raw_u + radius_px)
    y0, y1 = max(0, raw_v - radius_px), min(h - 1, raw_v + radius_px)
    yy, xx = np.nonzero(mask[y0 : y1 + 1, x0 : x1 + 1])
    if len(xx) == 0:
        return None, None
    xx = xx + x0
    yy = yy + y0
    dist_sq = (xx - raw_u) * (xx - raw_u) + (yy - raw_v) * (yy - raw_v)
    in_radius = dist_sq <= radius_px * radius_px
    if not np.any(in_radius):
        return None, None
    valid_indices = np.flatnonzero(in_radius)
    best = int(valid_indices[int(np.argmin(dist_sq[in_radius]))])
    return [int(xx[best]), int(yy[best])], float(np.sqrt(float(dist_sq[best])))


def _draw_goal_overlay(path, rgb, raw_uv, world_uv=None, rgb_uv=None, free_mask=None):
    image = cv2.cvtColor(np.asarray(rgb).copy(), cv2.COLOR_RGB2BGR)
    if free_mask is not None:
        image[free_mask] = (0.42 * image[free_mask].astype(np.float32) + 0.58 * np.array([210, 210, 210])).astype(
            np.uint8
        )
    raw = tuple(int(v) for v in raw_uv)
    cv2.circle(image, raw, 9, (255, 0, 255), 2)
    cv2.drawMarker(image, raw, (255, 0, 255), cv2.MARKER_CROSS, 22, 2)
    if world_uv is not None:
        world_uv = tuple(int(v) for v in world_uv)
        cv2.circle(image, world_uv, 8, (0, 255, 0), 2)
        cv2.drawMarker(image, world_uv, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 18, 2)
    if rgb_uv is not None:
        rgb_uv = tuple(int(v) for v in rgb_uv)
        cv2.circle(image, rgb_uv, 10, (255, 255, 0), 2)
        cv2.drawMarker(image, rgb_uv, (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.line(image, raw, rgb_uv, (255, 255, 0), 2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _write_native_pack(goal_dir, occ_frames, frames_dir, rows, cols, fx, fy, cx, cy):
    pack_dir = goal_dir / "native_occ_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for frame in occ_frames:
        depth = np.load(Path(frames_dir) / frame["depth"]).astype(np.float32)
        depth_name = f"depth_{int(frame['idx']):06d}.f32"
        depth_path = pack_dir / depth_name
        depth.tofile(depth_path)
        t = np.asarray(frame["t_w_c"], dtype=np.float32).reshape(4, 4).reshape(-1)
        lines.append(" ".join([f"native_occ_pack/{depth_name}"] + [f"{float(v):.9g}" for v in t]))
    config_path = goal_dir / "native_occ_frames.txt"
    header = f"{rows} {cols} {fx:.9g} {fy:.9g} {cx:.9g} {cy:.9g} {len(lines)}\n"
    config_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def correct_pixel_goal_with_native_nvblox(
    occ_frames,
    frames_dir,
    current_rgb,
    current_depth_m,
    pixel_goal,
    t_w_c,
    fx,
    fy,
    cx,
    cy,
    output_dir,
    record_prefix,
    mode,
    runner_bin,
    filter_script,
    nvblox_build,
    cuda_root="/usr/local/cuda-12.8",
    voxel_size_m=0.05,
    nearest_free_radius_m=0.6,
    rgb_search_radius_px=140,
    rgb_point_radius_px=2,
    rgb_max_depth_m=8.0,
    rgb_depth_occlusion_tol_m=0.35,
    python_bin=None,
):
    if not occ_frames:
        raise RuntimeError("native nvblox correction requires at least one saved RGBD frame")
    mode = str(mode).lower()
    if mode not in {"world2d", "rgbmask", "rgbmask_filled"}:
        raise ValueError(f"Unsupported native nvblox goal mode: {mode}")

    output_dir = Path(output_dir)
    goal_dir = output_dir / record_prefix
    goal_dir.mkdir(parents=True, exist_ok=True)
    h, w = current_depth_m.shape
    # Same u/v transposition bug as sim_occ_goal_correction.py (fixed there
    # 2026-07-23, see experiment_log.md) -- the caller passes pixel_goal as
    # [v, u], not [u, v]. This backend isn't exercised by any current config
    # (occ_goal_correction_backend is "local_depth" everywhere), so the fix
    # here is unverified against a real run, but it's the same bug.
    raw_u = int(np.clip(pixel_goal[1], 0, w - 1))
    raw_v = int(np.clip(pixel_goal[0], 0, h - 1))

    config_path = _write_native_pack(goal_dir, occ_frames, frames_dir, h, w, fx, fy, cx, cy)
    out_ply = goal_dir / "native_nvblox_current_occupancy.ply"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{nvblox_build}/nvblox:{cuda_root}/lib64:{nvblox_build}/_deps/ext_stdgpu-build/bin:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    _run([str(runner_bin), str(config_path), str(out_ply), str(voxel_size_m)], env=env)
    filter_dir = goal_dir / "habitat_known_ground_filter"
    _run([python_bin or "python", str(filter_script), "--occupancy-ply", str(out_ply), "--output-dir", str(filter_dir)])

    t_w_c = np.asarray(t_w_c, dtype=np.float32).reshape(4, 4)
    t_c_w = np.linalg.inv(t_w_c)
    intr = (float(fx), float(fy), float(cx), float(cy))
    record = {
        "backend": "native_nvblox",
        "mode": mode,
        "status": "unknown",
        "reason": "not_evaluated",
        "raw_pixel_goal": [raw_u, raw_v],
        "corrected_pixel_goal": [raw_u, raw_v],
        "used_corrected_goal": False,
        "goal_dir": str(goal_dir),
        "current_occupancy_ply": str(out_ply),
    }

    meta = json.loads((filter_dir / "traversability_meta.json").read_text(encoding="utf-8"))
    grid = np.load(filter_dir / "traversability_map.npy")
    z = _nearest_valid_depth(current_depth_m, raw_u, raw_v)
    raw_world = None
    raw_grid = None
    raw_value = None
    world_corrected_uv = None
    world_corrected_grid = None
    nearest_dist_cells = None
    first_blocked = None
    robot_grid = _world_to_grid_xy(t_w_c[:3, 3], meta)
    if z is not None:
        raw_world = t_w_c @ _pixel_to_camera(raw_u, raw_v, z, fx, fy, cx, cy)
        raw_grid = _world_to_grid_xy(raw_world, meta)
        if raw_grid is not None and robot_grid is not None:
            raw_value = int(grid[raw_grid[1], raw_grid[0]])
            if raw_value == FREE:
                world_corrected_grid = raw_grid
                world_corrected_uv = [raw_u, raw_v]
            else:
                radius_cells = max(1, int(round(nearest_free_radius_m / float(meta["resolution_m"]))))
                world_corrected_grid, nearest_dist_cells = _nearest_free_cell(grid, raw_grid, radius_cells)
                if world_corrected_grid is None:
                    world_corrected_grid, first_blocked = _last_free_on_ray(grid, robot_grid, raw_grid)
                if world_corrected_grid is not None:
                    corrected_world = _grid_to_world_xy(world_corrected_grid, meta, z=0.0)
                    world_corrected_uv = _camera_to_pixel(t_c_w @ corrected_world, fx, fy, cx, cy, w, h)

    record.update(
        {
            "raw_depth_m": z,
            "raw_world_xyz": raw_world[:3].astype(float).tolist() if raw_world is not None else None,
            "robot_world_xyz": t_w_c[:3, 3].astype(float).tolist(),
            "raw_grid": raw_grid,
            "robot_grid": robot_grid,
            "raw_grid_value": raw_value,
            "world_corrected_pixel_goal": world_corrected_uv,
            "world_corrected_grid": world_corrected_grid,
            "first_blocked": first_blocked,
            "nearest_free_radius_m": nearest_free_radius_m,
            "nearest_free_distance_m": (
                float(nearest_dist_cells * float(meta["resolution_m"])) if nearest_dist_cells is not None else None
            ),
        }
    )

    free_mask = None
    rgb_corrected_uv = None
    rgb_dist = None
    if mode in {"rgbmask", "rgbmask_filled"}:
        _, free_mask = _project_cloud_to_image_mask(
            filter_dir / "native_nvblox_ground_known_z0.ply",
            t_w_c,
            intr,
            w,
            h,
            current_depth_m,
            rgb_max_depth_m,
            rgb_depth_occlusion_tol_m,
            rgb_point_radius_px,
        )
        raw_rgb_free_mask = free_mask
        if mode == "rgbmask_filled":
            _, obstacle_mask = _project_cloud_to_image_mask(
                filter_dir / "native_nvblox_planning_obstacles.ply",
                t_w_c,
                intr,
                w,
                h,
                current_depth_m,
                rgb_max_depth_m,
                rgb_depth_occlusion_tol_m,
                max(3, rgb_point_radius_px),
            )
            free_mask, filled_debug = _filled_visible_free_mask(raw_rgb_free_mask, obstacle_mask, current_depth_m)
            record.update(
                {
                    "rgbmask_filled_raw_free_pixels": int(np.count_nonzero(raw_rgb_free_mask)),
                    "rgbmask_filled_morphed_pixels": int(np.count_nonzero(filled_debug["morphed_mask"])),
                    "rgbmask_filled_obstacle_block_pixels": int(np.count_nonzero(filled_debug["obstacle_block_mask"])),
                    "rgbmask_filled_depth_edge_pixels": int(np.count_nonzero(filled_debug["depth_edge_mask"])),
                    "rgbmask_filled_candidate_pixels": int(np.count_nonzero(filled_debug["candidate_mask"])),
                    "rgbmask_filled_pixels": int(np.count_nonzero(free_mask)),
                    "rgbmask_filled_seed_pixel": filled_debug["seed_pixel"],
                }
            )
        rgb_corrected_uv, rgb_dist = _nearest_mask_pixel(free_mask, [raw_u, raw_v], rgb_search_radius_px)
        if rgb_corrected_uv is None:
            record.update({"status": "invalid", "reason": "no_rgb_free_mask_pixel_near_raw_goal"})
        elif rgb_dist <= 0.0:
            record.update({"status": "valid", "reason": "raw_goal_in_projected_rgb_free_mask"})
        else:
            record.update(
                {
                    "status": "corrected",
                    "reason": "snapped_to_nearest_projected_rgb_free_mask_pixel",
                    "corrected_pixel_goal": rgb_corrected_uv,
                    "used_corrected_goal": True,
                }
            )
        record.update(
            {
                "rgb_corrected_pixel_goal": rgb_corrected_uv,
                "rgb_corrected_distance_px": rgb_dist,
                "rgb_search_radius_px": int(rgb_search_radius_px),
                "rgb_free_pixels": int(np.count_nonzero(free_mask)) if free_mask is not None else 0,
            }
        )
        if mode == "rgbmask_filled":
            debug_dir = _save_rgbmask_filled_debug(
                goal_dir,
                current_rgb,
                [raw_u, raw_v],
                raw_rgb_free_mask,
                free_mask,
                filled_debug,
                old_uv=record.get("world_corrected_pixel_goal"),
                new_uv=rgb_corrected_uv,
            )
            record["rgbmask_filled_debug_dir"] = str(debug_dir)
    else:
        if z is None:
            record.update({"status": "unknown", "reason": "no_depth_at_pixel_goal"})
        elif raw_grid is None or robot_grid is None:
            record.update({"status": "invalid", "reason": "goal_or_robot_outside_current_nvblox_map"})
        elif raw_value == FREE:
            record.update({"status": "valid", "reason": "raw_goal_in_current_nvblox_free"})
        elif world_corrected_uv is None:
            record.update({"status": "invalid", "reason": "no_nearby_or_ray_free_cell_in_current_nvblox"})
        else:
            record.update(
                {
                    "status": "corrected",
                    "reason": "snapped_to_nearest_current_nvblox_free_cell",
                    "corrected_pixel_goal": world_corrected_uv,
                    "used_corrected_goal": True,
                }
            )

    overlay = goal_dir / "native_nvblox_goal_overlay.png"
    _draw_goal_overlay(
        overlay,
        current_rgb,
        [raw_u, raw_v],
        world_uv=world_corrected_uv,
        rgb_uv=rgb_corrected_uv,
        free_mask=free_mask,
    )
    record["overlay"] = str(overlay)
    (goal_dir / "native_nvblox_goal_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
