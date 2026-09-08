# Collision-Aware Geometric Trajectory Filtering for Vision-and-Language Navigation

Code release for Zimo Chen's MSc project at University College London.

This repository implements a training-free geometric safety filter for the
InternVLA-N1-DualVLN navigation pipeline. The filter operates after System 1
has sampled local trajectories: it back-projects the current depth image,
checks candidate paths for observed collision risk and ground support, and
passes a safer candidate subset to the original trajectory aggregation and
execution pipeline. The learned navigation models are left unchanged.

## Main contribution

The final method is the **Conservative Safe-Subset Filter (V2)**:

1. Generate 32 candidate local trajectories with the original diffusion policy.
2. Convert valid depth observations into local free-ground and obstacle point sets.
3. Check the first 12 positions of the original mean trajectory.
4. If that mean is safe, preserve the original aggregation unchanged.
5. Otherwise, retain and average the candidates that satisfy the collision and
   backward-motion checks, preferring candidates with stronger observed ground
   support.
6. If no candidate satisfies the hard checks, use one lexicographic
   minimum-risk candidate as a fallback.

The repository also contains two development variants used in the thesis:

- **Baseline — Original Aggregation:** unmodified DualVLN aggregation over all
  sampled candidates.
- **V1 — High-Support Subset Filter:** an early variant whose intervention
  branches often return a single candidate.
- **V2 — Conservative Safe-Subset Filter:** the final method described above.
- **V3 — Direction-Clustered Fallback Filter:** a V2 ablation that changes only
  the no-safe-candidate fallback.

The central implementation is in
[`internnav/utils/sim_occ_goal_correction.py`](internnav/utils/sim_occ_goal_correction.py).
Simulation integration is in
[`internnav/habitat_extensions/vln/habitat_vln_evaluator.py`](internnav/habitat_extensions/vln/habitat_vln_evaluator.py),
and the physical-robot server is in
[`scripts/realworld_trajectory_exp/http_internvla_traj_server_occ_rerank.py`](scripts/realworld_trajectory_exp/http_internvla_traj_server_occ_rerank.py).

## Reported results

On the paired R2R VLN-CE `val_unseen` evaluation (1,839 episodes), V2 reduced
the total number of simulator collision events from 25,812 to 17,271, a 33.1%
reduction. Success rate changed from 60.2% to 59.3%; the paired McNemar test did
not indicate a significant success-rate difference (`p = 0.37`).

Under the no-sliding diagnostic condition, success rate increased from 28.1%
to 47.4% and collision events decreased by 57.0%.

In the controlled Unitree Go2W study (two scenarios, five trials per method in
each scenario), the Baseline completed 8/10 trials with contact in 6/10 trials;
V2 completed 9/10 trials with contact in 2/10 trials. These physical results are
small-scale descriptive evidence rather than population-level estimates.

## Final experimental parameters

| Parameter | Simulation | Physical robot |
| --- | ---: | ---: |
| Candidate trajectories | 32 | 32 |
| Predicted trajectory steps | 32 | 32 |
| Diffusion inference steps | 16 | 16 |
| Checked horizon | 12 | 12 |
| Depth-image stride | 4 | 4 |
| Filter depth range | 0.15–5.0 m | 0.05–8.0 m |
| Floor percentile | 90 | 90 |
| Free-ground height tolerance | 0.08 m | 0.08 m |
| Occupied-height threshold | 0.18 m | 0.18 m |
| Collision-query radius | 0.28 m | 0.32 m |
| Support-query radius | 0.30 m | 0.35 m |
| Minimum support fraction | 0.55 | 0.55 |

The physical collision-query radius is represented in the launch interface as
a 0.22 m robot radius plus a 0.10 m safety margin.

## Repository structure

```text
internnav/
  habitat_extensions/vln/    paired simulator evaluation and collision logging
  utils/                      geometric checks, candidate filtering, map utilities
scripts/
  eval/                       Habitat evaluation entry points and example configs
  realworld_trajectory_exp/   Go2W server, client, diagnostics, and smoke test
requirements/                 dependency groups inherited from InternNav
tests/                        upstream unit-test scaffolding
```

Generated logs, model checkpoints, datasets, videos, captured depth arrays,
robot credentials, and machine-specific paths are intentionally excluded from
this public source release.

## Installation

The software stack is GPU-oriented and follows the InternNav environment. A
typical installation is:

```bash
git clone --recurse-submodules https://github.com/Zimoc02/Collision-Aware-Trajectory-Filtering.git
cd Collision-Aware-Trajectory-Filtering
conda create -n collision-aware-vln python=3.10 -y
conda activate collision-aware-vln
pip install -e '.[habitat,internvla_n1]'
```

InternVLA-N1-DualVLN checkpoints, R2R annotations, and Matterport3D scene data
must be obtained separately under their respective access conditions. Place
the model at `checkpoints/InternVLA-N1-DualVLN`, or update `model_path` in the
evaluation config.

## Simulation evaluation

Set the external dataset locations before running:

```bash
export VLN_SCENES_DIR=/path/to/matterport3d
export VLN_R2R_DATA_PATH='/path/to/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz'
python scripts/eval/eval.py \
  --config scripts/eval/configs/habitat_dual_system_v2_cfg.py
```

Use `habitat_dual_system_baseline_cfg.py` for the matched Baseline. Both example
configs use deterministic episode seeding and differ in the V2 enable flag.
Output is written below `logs/habitat/` and is ignored by Git.

## Physical-robot server

The final physical configuration can be launched on the GPU workstation with:

```bash
python scripts/realworld_trajectory_exp/http_internvla_traj_server_occ_rerank.py \
  --model_path checkpoints/InternVLA-N1-DualVLN \
  --device cuda:0 \
  --port 5802 \
  --traj_num_samples 32 \
  --traj_inference_steps 16 \
  --traj_predict_steps 32 \
  --enable_traj_occ_rerank \
  --traj_occ_min_depth_m 0.05 \
  --traj_occ_max_depth_m 8.0
```

Robot-side addresses, interfaces, SDK paths, and credentials must be supplied
for the local deployment. Keep the client in dry-run mode until the complete
robot control and emergency-stop chain has been checked.

The lightweight occupancy evaluation can be checked without a robot:

```bash
python scripts/realworld_trajectory_exp/smoke_test_occ_eval.py
```

## Attribution and licence

This project is a modification of
[InternRobotics/InternNav](https://github.com/InternRobotics/InternNav) and uses
the InternVLA-N1/DualVLN and NavDP software stack. The original InternNav
copyright notice and MIT licence are retained in [`LICENSE`](LICENSE). The
geometric filtering, evaluation integration, collision logging, and Go2W
experimental additions in this repository were developed for the MSc project.

If you use this repository, cite both this project and the relevant upstream
InternNav, InternVLA-N1/DualVLN, and NavDP publications or repositories.
