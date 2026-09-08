# Unitree Go2W trajectory-filter deployment

This directory contains the physical-robot integration used for the controlled
Baseline/V2 study. It is separate from the upstream discrete-action deployment
under `scripts/realworld/`.

Run the trajectory server on the GPU workstation from the repository root:

```bash
python scripts/realworld_trajectory_exp/http_internvla_traj_server_occ_rerank.py \
  --model_path checkpoints/InternVLA-N1-DualVLN \
  --device cuda:0 \
  --instruction_file realworld_instruction.txt \
  --port 5802 \
  --traj_num_samples 32 \
  --traj_inference_steps 16 \
  --traj_predict_steps 32 \
  --enable_traj_occ_rerank \
  --traj_occ_min_depth_m 0.05 \
  --traj_occ_max_depth_m 8.0 \
  --traj_occ_stride 4 \
  --traj_occ_floor_percentile 90 \
  --traj_occ_free_height_tol_m 0.08 \
  --traj_occ_occupied_height_m 0.18 \
  --traj_occ_robot_radius_m 0.24 \
  --traj_occ_safety_margin_m 0.08 \
  --traj_occ_support_radius_m 0.35 \
  --traj_occ_min_support_fraction 0.55 \
  --traj_occ_horizon_steps 12
```

The collision-query radius is `0.24 + 0.08 = 0.32 m`. Disable the filter with
`--no-enable_traj_occ_rerank` for the matched Baseline condition.

Robot-side clients require site-specific network addresses, SDK paths, and the
`GO2W_PASSWORD` environment variable. The probe is dry-run by default and only
sends movement commands when `--execute` is supplied.

The occupancy-map utility can be checked independently with:

```bash
python scripts/realworld_trajectory_exp/smoke_test_occ_eval.py
```
