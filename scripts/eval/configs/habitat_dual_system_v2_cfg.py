from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg


eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name="internvla_n1",
        model_settings={
            "mode": "dual_system",
            "model_path": "checkpoints/InternVLA-N1-DualVLN",
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
            "attn_implementation": "sdpa",
            "vis_debug": False,
        },
    ),
    env=EnvCfg(
        env_type="habitat",
        env_settings={"config_path": "scripts/eval/configs/vln_r2r_local_navila.yaml"},
    ),
    eval_type="habitat_vln",
    eval_settings={
        "output_path": "./logs/habitat/v2_val_unseen",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2333",
        "dist_url": "env://",
        "episode_seed_offset": 0,
        "enable_traj_occ_rerank": True,
        "traj_occ_max_depth_m": 5.0,
        "traj_occ_stride": 4,
        "traj_occ_floor_percentile": 90.0,
        "traj_occ_free_height_tol_m": 0.08,
        "traj_occ_occupied_height_m": 0.18,
        "traj_occ_robot_radius_m": 0.20,
        "traj_occ_safety_margin_m": 0.08,
        "traj_occ_support_radius_m": 0.30,
        "traj_occ_min_support_fraction": 0.55,
        "traj_occ_horizon_steps": 12,
    },
)
