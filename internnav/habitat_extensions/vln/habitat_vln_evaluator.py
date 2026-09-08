import argparse
import json
import math
import os
import sys
import zlib
from enum import IntEnum

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import random
import re
from collections import OrderedDict

import cv2
import habitat
import imageio
import numpy as np
import quaternion
import torch
import tqdm
from depth_camera_filtering import filter_depth
try:
    from habitat.config.default import get_agent_config
except ImportError:

    def get_agent_config(simulator_config):
        return simulator_config.agents.main_agent
try:
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
except ImportError:
    CollisionsMeasurementConfig = None
    FogOfWarConfig = None
    TopDownMapMeasurementConfig = None
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.habitat_extensions.vln.utils import (
    get_axis_align_matrix,
    get_intrinsic_matrix,
    pixel_to_gps,
    preprocess_depth_image_v2,
    xyz_yaw_pitch_to_tf_matrix,
)
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions
from internnav.utils.native_nvblox_goal_correction import correct_pixel_goal_with_native_nvblox
from internnav.utils.sim_occ_goal_correction import (
    append_jsonl,
    correct_pixel_goal_with_local_occ,
    rerank_trajectories_with_local_depth,
)

# Import for Habitat registry side effects — do not remove
import internnav.habitat_extensions.vln.measures  # noqa: F401 # isort: skip


DEFAULT_IMAGE_TOKEN = "<image>"

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4

# method_5_arrive: must match habitat.simulator.forward_step_size in the eval
# yaml (all current configs use 0.25m). Used to convert a corrected goal's
# depth into an estimated forward-step budget, so the agent isn't cut off by
# the fixed MAX_STEPS tether before it can actually reach a corrected point
# that legitimately requires more than MAX_STEPS forward actions.
FORWARD_STEP_SIZE_M = 0.25

# method_5_arrive (2026-07-26 fix): arrive_gate_remaining is a FORWARD-only
# budget (see arrive_forward_moves below) -- turn-in-place actions produce no
# displacement toward the corrected goal's estimated depth and must not
# consume it (diagnosed via TbHJrupSAjP episode 1270: 21-step budget spent as
# 11 forward + 10 turns, so the agent covered <50% of the intended distance
# before being cut off). This multiplier still bounds total wall-clock time
# when a route needs a lot of turning, independently of the forward budget.
ARRIVE_GATE_TOTAL_STEP_MULTIPLIER = 3


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


def log_step_collision(evaluator, step_collision_log, scene_id, episode_id, step_id, action):
    # Must be called right after a real self.env.step() call (not the noop_
    # lookdown branch, which never touches the simulator) so that
    # info['collisions']['is_collision'] reflects that specific step -- a
    # stale get_metrics() read shared across multiple loop passes (e.g. the
    # lookdown no-op iterations that precede the real motion action) would
    # double-count the same underlying collision event.
    info = evaluator.env.get_metrics()
    collisions = info.get('collisions')
    if collisions is None:
        return
    append_jsonl(
        step_collision_log,
        {
            "scene_id": scene_id,
            "episode_id": int(episode_id),
            "step_id": int(step_id),
            "action": int(action),
            "is_collision": bool(collisions.get('is_collision')),
            "collision_count_so_far": collisions.get('count'),
        },
    )


@Evaluator.register('habitat_vln')
class HabitatVLNEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.save_video = args.save_video
        self.save_all_videos = bool(getattr(args, "save_all_videos", False))
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.output_path = args.output_path

        # create habitat config
        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        if TopDownMapMeasurementConfig is not None and hasattr(habitat.config, "read_write"):
            with habitat.config.read_write(self.config):
                self.config.habitat.task.measurements.update(
                    {
                        "top_down_map": TopDownMapMeasurementConfig(
                            map_padding=3,
                            map_resolution=1024,
                            draw_source=True,
                            draw_border=True,
                            draw_shortest_path=True,
                            draw_view_points=True,
                            draw_goal_positions=True,
                            draw_goal_aabbs=True,
                            fog_of_war=FogOfWarConfig(
                                draw=True,
                                visibility_dist=5.0,
                                fov=90,
                            ),
                        ),
                        "collisions": CollisionsMeasurementConfig(),
                    }
                )
        cfg.env.env_settings['habitat_config'] = self.config
        cfg.env.env_settings['output_path'] = self.output_path

        # init agent and env
        super().__init__(cfg, init_agent=False)

        # ------------------------------------- model ------------------------------------------
        self.model_args = argparse.Namespace(**cfg.agent.model_settings)
        self.vis_debug = bool(getattr(self.model_args, "vis_debug", False))
        self.vis_debug_path = getattr(self.model_args, "vis_debug_path", os.path.join(self.output_path, "vis_debug"))
        self.attn_implementation = getattr(self.model_args, "attn_implementation", "flash_attention_2")
        # Reproducibility: generate_traj()'s diffusion sampling (randn_tensor(...,
        # generator=None)) and the conjunction-phrase prompt text
        # (random.choice(self.conjunctions)) both draw from the global,
        # never-seeded RNG state -- so the same episode can behave differently
        # depending on how many prior random draws happened earlier in the same
        # process (see experiment_log.md: the same 184 episodes gave a 5+ point
        # SR swing and ~23% per-episode flip rate run standalone vs embedded in
        # the full 1839-集 run). Default (None) leaves this untouched -- fully
        # random, original behavior, for whenever genuine run-to-run variance is
        # what you actually want (e.g. estimating a method's variance across
        # repeated trials). Setting `episode_seed_offset` in eval_settings makes
        # each episode's randomness a pure function of (scene_id, episode_id,
        # offset) via zlib.crc32 (deterministic across processes, unlike
        # Python's built-in hash()) -- so two different configs (e.g. baseline
        # vs a correction method) run with the *same* offset see identical
        # per-episode random draws, and any outcome difference is attributable
        # to the actual method difference rather than independent RNG noise.
        # Different offsets give independent-but-reproducible trials for
        # variance estimation.
        self.episode_seed_offset = getattr(args, "episode_seed_offset", None)
        # method_5_arrive ("no_arrive" A/B variant, 2026-07-27): isolates
        # whether a correction method's SR difference vs baseline comes from
        # picking a different pixel goal, or from arrive_gate_remaining's
        # step-budget widening itself (a side channel baseline never has,
        # since it never computes occ_record). Default False preserves all
        # existing behavior; True forces arrive_gate_remaining to stay 0
        # always, so the step-cap logic is identical to baseline
        # (forward_action > MAX_STEPS, never widened) regardless of what a
        # correction reports.
        self.disable_arrive_gate_step_budget = bool(getattr(args, "disable_arrive_gate_step_budget", False))
        self.enable_occ_goal_correction = bool(getattr(args, "enable_occ_goal_correction", False))
        self.save_occ_sequence = bool(getattr(args, "save_occ_sequence", self.enable_occ_goal_correction))
        self.noop_lookdown = bool(getattr(args, "noop_lookdown", False))
        self.occ_resolution_m = float(getattr(args, "occ_resolution_m", 0.05))
        self.occ_max_depth_m = float(getattr(args, "occ_max_depth_m", 5.0))
        self.local_occ_stride = int(getattr(args, "local_occ_stride", 4))
        self.local_occ_inflation_radius_m = float(getattr(args, "local_occ_inflation_radius_m", 0.20))
        self.local_occ_floor_percentile = float(getattr(args, "local_occ_floor_percentile", 82.0))
        self.local_occ_free_height_tol_m = float(getattr(args, "local_occ_free_height_tol_m", 0.18))
        self.local_occ_occupied_height_m = float(getattr(args, "local_occ_occupied_height_m", 0.22))
        self.local_occ_rgb_point_radius_px = int(getattr(args, "local_occ_rgb_point_radius_px", 1))
        self.local_occ_correction_mode = str(getattr(args, "local_occ_correction_mode", "topdown_ray"))
        self.local_occ_include_unknown_as_free = bool(getattr(args, "local_occ_include_unknown_as_free", False))
        self.local_occ_target_clearance_px = float(getattr(args, "local_occ_target_clearance_px", 75.0))
        self.local_occ_clearance_band_px = float(getattr(args, "local_occ_clearance_band_px", 12.0))
        self.local_occ_max_shift_px = float(getattr(args, "local_occ_max_shift_px", 220.0))
        self.local_occ_robot_x_from_high_clearance = bool(
            getattr(args, "local_occ_robot_x_from_high_clearance", False)
        )
        self.occ_goal_debug_image_mode = str(getattr(args, "occ_goal_debug_image_mode", "all"))
        self.occ_goal_correction_backend = str(getattr(args, "occ_goal_correction_backend", "local_depth"))
        self.native_nvblox_goal_mode = str(getattr(args, "native_nvblox_goal_mode", "world2d"))
        self.native_nvblox_runner_bin = str(
            getattr(
                args,
                "native_nvblox_runner_bin",
                os.environ.get("NVBLOX_OCCUPANCY_RUNNER", "native_nvblox_occupancy_runner"),
            )
        )
        self.native_nvblox_filter_script = str(
            getattr(
                args,
                "native_nvblox_filter_script",
                os.environ.get("NVBLOX_FILTER_SCRIPT", "filter_habitat_nvblox_known_ground.py"),
            )
        )
        self.native_nvblox_build = str(
            getattr(args, "native_nvblox_build", os.environ.get("NVBLOX_BUILD_DIR", ""))
        )
        self.native_nvblox_cuda_root = str(getattr(args, "native_nvblox_cuda_root", "/usr/local/cuda-12.8"))
        self.native_nvblox_python_bin = str(
            getattr(args, "native_nvblox_python_bin", sys.executable)
        )
        self.native_nvblox_nearest_free_radius_m = float(getattr(args, "native_nvblox_nearest_free_radius_m", 0.6))
        self.native_nvblox_rgb_search_radius_px = int(getattr(args, "native_nvblox_rgb_search_radius_px", 140))
        self.native_nvblox_rgb_point_radius_px = int(getattr(args, "native_nvblox_rgb_point_radius_px", 2))
        self.native_nvblox_rgb_max_depth_m = float(getattr(args, "native_nvblox_rgb_max_depth_m", 8.0))
        self.native_nvblox_rgb_depth_occlusion_tol_m = float(
            getattr(args, "native_nvblox_rgb_depth_occlusion_tol_m", 0.35)
        )
        self.enable_traj_occ_rerank = bool(getattr(args, "enable_traj_occ_rerank", False))
        self.traj_occ_max_depth_m = float(getattr(args, "traj_occ_max_depth_m", 5.0))
        self.traj_occ_stride = int(getattr(args, "traj_occ_stride", 4))
        self.traj_occ_floor_percentile = float(getattr(args, "traj_occ_floor_percentile", 90.0))
        self.traj_occ_free_height_tol_m = float(getattr(args, "traj_occ_free_height_tol_m", 0.08))
        self.traj_occ_occupied_height_m = float(getattr(args, "traj_occ_occupied_height_m", 0.18))
        self.traj_occ_robot_radius_m = float(getattr(args, "traj_occ_robot_radius_m", 0.20))
        self.traj_occ_safety_margin_m = float(getattr(args, "traj_occ_safety_margin_m", 0.08))
        self.traj_occ_support_radius_m = float(getattr(args, "traj_occ_support_radius_m", 0.30))
        self.traj_occ_min_support_fraction = float(getattr(args, "traj_occ_min_support_fraction", 0.55))
        self.traj_occ_horizon_steps = int(getattr(args, "traj_occ_horizon_steps", 12))

        processor = AutoProcessor.from_pretrained(self.model_args.model_path)
        processor.tokenizer.padding_side = 'left'

        device = torch.device(f"cuda:{self.local_rank}")
        if self.model_args.mode == 'dual_system':
            model = InternVLAN1ForCausalLM.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.attn_implementation,
                device_map={"": device},
            )
        elif self.model_args.mode == 'system2':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.attn_implementation,
                device_map={"": device},
            )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        model.eval()
        self.device = device

        self.model = model
        self.processor = processor

        # refactor: this part used in three places
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.num_history = self.model_args.num_history

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        # Old behavior was something like:
        # sucs, spls, oss, nes, ep_num = self.eval_action(self.rank)
        # Now just implement the actual eval here and return dict.

        if self.model_args.mode == 'dual_system':
            sucs, spls, oss, nes, ndtws = self._run_eval_dual_system()
        elif self.model_args.mode == 'system2':
            sucs, spls, oss, nes, ndtws = self._run_eval_system2()
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        result = {
            "sucs": sucs,  # shape [N_local]
            "spls": spls,  # shape [N_local]
            "oss": oss,  # shape [N_local]
            "nes": nes,  # shape [N_local]
        }

        if ndtws is not None:
            result["ndtws"] = ndtws  # shape [N_local]
        return result

    def calc_metrics(self, global_metrics: dict) -> dict:
        """
        global_metrics["sucs"] etc. are global 1-D CPU tensors with all episodes.
        """
        sucs_all = global_metrics["sucs"]
        spls_all = global_metrics["spls"]
        oss_all = global_metrics["oss"]
        nes_all = global_metrics["nes"]

        # avoid /0 if no episodes
        denom = max(len(sucs_all), 1)

        # clean NaN in spls, treat as 0.0
        torch.nan_to_num(spls_all, nan=0.0, posinf=0.0, neginf=0.0, out=spls_all)

        # clean inf in nes, only fiinite nes are counted
        nes_finite_mask = torch.isfinite(nes_all)
        nes_all = nes_all[nes_finite_mask]

        result_all = {
            "sucs_all": float(sucs_all.mean().item()) if denom > 0 else 0.0,
            "spls_all": float(spls_all.mean().item()) if denom > 0 else 0.0,
            "oss_all": float(oss_all.mean().item()) if denom > 0 else 0.0,
            "nes_all": float(nes_all.mean().item()) if denom > 0 else 0.0,
            # "length" will be filled by base class
        }

        if "ndtws" in global_metrics:
            ndtws_all = global_metrics["ndtws"]
            result_all["ndtws_all"] = float(ndtws_all.mean().item()) if denom > 0 else 0.0

        return result_all

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def resume_from_output_path(self) -> None:
        sucs, spls, oss, nes, ndtw = [], [], [], [], []
        if self.rank != 0:
            return sucs, spls, oss, nes, ndtw

        # resume from previous results
        if os.path.exists(os.path.join(self.output_path, 'progress.json')):
            with open(os.path.join(self.output_path, 'progress.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    sucs.append(res['success'])
                    spls.append(res['spl'])
                    oss.append(res['os'])
                    nes.append(res['ne'])
                    if 'ndtw' in res:
                        ndtw.append(res['ndtw'])
        return sucs, spls, oss, nes, ndtw

    def _run_eval_dual_system(self) -> tuple:  # noqa: C901
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)

            if self.episode_seed_offset is not None:
                # deterministic across processes (unlike builtin hash(), which
                # is salted per-process via PYTHONHASHSEED) -- see the
                # episode_seed_offset comment in __init__.
                seed_str = f"{scene_id}_{episode_id}_{self.episode_seed_offset}"
                episode_seed = zlib.crc32(seed_str.encode("utf-8")) % (2**31)
                torch.manual_seed(episode_seed)
                random.seed(episode_seed)

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            action = None
            messages = []
            local_actions = []

            done = False
            flag = False
            pixel_goal = None
            arrive_gate_remaining = 0  # method_5_arrive: forward-step budget override, see FORWARD_STEP_SIZE_M
            arrive_forward_moves = 0  # method_5_arrive: FORWARD-only count against arrive_gate_remaining
            initial_height = self.env._env.sim.get_agent_state().position[1]
            intrinsic_matrix = get_intrinsic_matrix(
                self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
            )
            occ_episode_dir = os.path.join(self.output_path, "occ_sequences", f"{scene_id}_{episode_id:04d}")
            occ_frames_dir = os.path.join(occ_episode_dir, "frames")
            occ_goal_dir = os.path.join(self.output_path, "occ_goal_debug", f"{scene_id}_{episode_id:04d}")
            occ_goal_log = os.path.join(self.output_path, "occ_pixel_goal_eval.jsonl")
            traj_occ_log = os.path.join(self.output_path, "traj_occ_rerank.jsonl")
            step_collision_log = os.path.join(self.output_path, "step_collisions.jsonl")
            occ_frames = []
            occ_frame_idx = 0
            occ_goal_idx = 0

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                camera_yaw = observations["compass"][0]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000
                front_depth_m = (depth / 1000.0).astype(np.float32)
                front_rgb = np.asarray(rgb).copy()

                agent_state = self.env._env.sim.get_agent_state()
                height = agent_state.position[1] - initial_height
                camera_position = np.array([x, -y, self._camera_height + height])
                tf_camera_to_episodic = (
                    xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, 0.0) @ get_axis_align_matrix()
                )

                if self.save_occ_sequence:
                    os.makedirs(occ_frames_dir, exist_ok=True)
                    rgb_name = f"rgb_{occ_frame_idx:06d}.png"
                    depth_name = f"depth_{occ_frame_idx:06d}.npy"
                    Image.fromarray(front_rgb).save(os.path.join(occ_frames_dir, rgb_name))
                    np.save(os.path.join(occ_frames_dir, depth_name), front_depth_m)
                    occ_frames.append(
                        {
                            "idx": occ_frame_idx,
                            "step_id": int(step_id),
                            "rgb": rgb_name,
                            "depth": depth_name,
                            "pose_source": "habitat_sim",
                            "t_w_c": tf_camera_to_episodic.astype(float).tolist(),
                            "depth_nonzero": int(np.count_nonzero(front_depth_m)),
                            "depth_min": float(np.nanmin(front_depth_m)),
                            "depth_max": float(np.nanmax(front_depth_m)),
                        }
                    )
                    occ_frame_idx += 1

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                    if self.noop_lookdown:
                        look_down_image = Image.fromarray(front_rgb).convert('RGB')
                        depth = front_depth_m * 1000
                    else:
                        # Habitat asserts on step() once an episode is already
                        # over, so each of these extra lookdown/lookup calls
                        # (which don't count toward step_id / max_steps_per_episode)
                        # must be skipped once `done` fires mid-sequence,
                        # otherwise a later call here crashes the whole run.
                        down_observations, _, done, _ = self.env.step(action_code.LOOKDOWN)
                        if not done:
                            down_observations, _, done, _ = self.env.step(action_code.LOOKDOWN)

                        look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                        depth = down_observations["depth"]
                        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                        depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                        depth = depth * 1000
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0

                    if not self.noop_lookdown and not done:
                        _, _, done, _ = self.env.step(action_code.LOOKUP)
                        if not done:
                            _, _, done, _ = self.env.step(action_code.LOOKUP)

                # The lookdown/lookup calls above can end the episode mid-sequence
                # (habitat's own step budget, not step_id/max_steps_per_episode).
                # Every later call in this iteration -- including the main
                # self.env.step(action) call further down -- assumes the episode
                # is still active, so bail out now and let the while-loop condition
                # (`while (not done) and ...`) end the episode cleanly instead of
                # crashing on a step() call against an already-over episode.
                if done:
                    continue

                if len(action_seq) == 0 and pixel_goal is None:
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                        ).sequences

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    raw_llm_outputs = llm_outputs
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        arrive_gate_remaining = 0
                        arrive_forward_moves = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]
                        raw_pixel_goal = list(pixel_goal)
                        draw_pixel_goal = True
                        occ_record = None
                        if self.enable_occ_goal_correction:
                            if self.occ_goal_correction_backend == "native_nvblox":
                                occ_record = correct_pixel_goal_with_native_nvblox(
                                    occ_frames=occ_frames,
                                    frames_dir=occ_frames_dir,
                                    current_rgb=front_rgb,
                                    current_depth_m=front_depth_m,
                                    pixel_goal=pixel_goal,
                                    t_w_c=tf_camera_to_episodic,
                                    fx=intrinsic_matrix[0, 0],
                                    fy=intrinsic_matrix[1, 1],
                                    cx=intrinsic_matrix[0, 2],
                                    cy=intrinsic_matrix[1, 2],
                                    output_dir=occ_goal_dir,
                                    record_prefix=f"step_{step_id:04d}_goal_{occ_goal_idx:04d}_{self.native_nvblox_goal_mode}",
                                    mode=self.native_nvblox_goal_mode,
                                    runner_bin=self.native_nvblox_runner_bin,
                                    filter_script=self.native_nvblox_filter_script,
                                    nvblox_build=self.native_nvblox_build,
                                    cuda_root=self.native_nvblox_cuda_root,
                                    voxel_size_m=self.occ_resolution_m,
                                    nearest_free_radius_m=self.native_nvblox_nearest_free_radius_m,
                                    rgb_search_radius_px=self.native_nvblox_rgb_search_radius_px,
                                    rgb_point_radius_px=self.native_nvblox_rgb_point_radius_px,
                                    rgb_max_depth_m=self.native_nvblox_rgb_max_depth_m,
                                    rgb_depth_occlusion_tol_m=self.native_nvblox_rgb_depth_occlusion_tol_m,
                                    python_bin=self.native_nvblox_python_bin,
                                )
                            else:
                                occ_record = correct_pixel_goal_with_local_occ(
                                    front_depth_m,
                                    pixel_goal,
                                    intrinsic_matrix[0, 0],
                                    intrinsic_matrix[1, 1],
                                    intrinsic_matrix[0, 2],
                                    intrinsic_matrix[1, 2],
                                    current_rgb=front_rgb,
                                    output_dir=occ_goal_dir,
                                    record_prefix=f"step_{step_id:04d}_goal_{occ_goal_idx:04d}",
                                    resolution_m=self.occ_resolution_m,
                                    max_depth_m=self.occ_max_depth_m,
                                    stride=self.local_occ_stride,
                                    inflation_radius_m=self.local_occ_inflation_radius_m,
                                    floor_percentile=self.local_occ_floor_percentile,
                                    free_height_tol_m=self.local_occ_free_height_tol_m,
                                    occupied_height_m=self.local_occ_occupied_height_m,
                                    rgb_point_radius_px=self.local_occ_rgb_point_radius_px,
                                    correction_mode=self.local_occ_correction_mode,
                                    debug_image_mode=self.occ_goal_debug_image_mode,
                                    local_occ_include_unknown_as_free=self.local_occ_include_unknown_as_free,
                                    local_occ_target_clearance_px=self.local_occ_target_clearance_px,
                                    local_occ_clearance_band_px=self.local_occ_clearance_band_px,
                                    local_occ_max_shift_px=self.local_occ_max_shift_px,
                                    local_occ_robot_x_from_high_clearance=self.local_occ_robot_x_from_high_clearance,
                                )
                            corrected_pixel = occ_record["corrected_pixel_goal"]
                            if occ_record.get("used_corrected_goal"):
                                pixel_goal = [int(corrected_pixel[0]), int(corrected_pixel[1])]
                                # NOTE (latent-refeed fix): the corrected coordinate is
                                # reordered back to the raw text order (training data stores
                                # the answer as f"{action[0]} {action[1]}"; pixel_goal here is
                                # stored [v, u] i.e. reversed relative to that, per the parsing
                                # a few lines up), tokenized fresh, and -- critically -- closed
                                # with the same eos/end-of-turn token
                                # (self.processor.tokenizer.eos_token_id, "<|im_end|>") that a
                                # genuine model.generate() call always ends the assistant turn
                                # with. The previous version omitted this token, so
                                # generate_latents() below was run on a sequence structurally
                                # different from anything the model produced or saw in
                                # training (see retrain_system1/design_doc.md section 1 /
                                # Simulation_evaluation/notes/experiment_log.md Method5/7
                                # results). output_ids is rebuilt from inputs.input_ids (the
                                # original prompt, already ending at the
                                # add_generation_prompt=True marker) + this properly-closed
                                # answer, so the corrected goal flows through the same
                                # text -> self-attention -> latent path the model was trained
                                # to use, instead of a separate untrained channel.
                                corrected_text = f"{pixel_goal[1]} {pixel_goal[0]}"
                                answer_ids = self.processor.tokenizer(
                                    corrected_text,
                                    return_tensors="pt",
                                    add_special_tokens=False,
                                ).input_ids.to(inputs.input_ids.device)
                                eos_id = torch.tensor(
                                    [[self.processor.tokenizer.eos_token_id]],
                                    device=inputs.input_ids.device,
                                    dtype=answer_ids.dtype,
                                )
                                answer_ids = torch.cat([answer_ids, eos_id], dim=1)
                                output_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
                                llm_outputs = corrected_text
                                # method_5_arrive: only the vertical-column search's own
                                # success sets arrive_gate (the diagonal-ray fallback does
                                # not -- see sim_occ_goal_correction.py). When set, don't
                                # let the fixed MAX_STEPS tether cut the agent off before
                                # it can plausibly reach this specific corrected point;
                                # forward_action's cap is widened below via
                                # max(MAX_STEPS, arrive_gate_remaining).
                                # disable_arrive_gate_step_budget ("no_arrive" A/B
                                # variant): skip this widening entirely so the step-cap
                                # logic matches baseline exactly, isolating "did the
                                # corrected pixel goal help" from "did the widened step
                                # budget itself help".
                                if (
                                    occ_record.get("arrive_gate")
                                    and occ_record.get("corrected_depth_m")
                                    and not self.disable_arrive_gate_step_budget
                                ):
                                    arrive_gate_remaining = math.ceil(
                                        occ_record["corrected_depth_m"] / FORWARD_STEP_SIZE_M
                                    )
                            occ_record.update(
                                {
                                    "scene_id": scene_id,
                                    "episode_id": int(episode_id),
                                    "step_id": int(step_id),
                                    "goal_index": int(occ_goal_idx),
                                    "occ_goal_correction_backend": self.occ_goal_correction_backend,
                                    "native_nvblox_goal_mode": self.native_nvblox_goal_mode,
                                    "llm_output_raw": raw_llm_outputs,
                                    "llm_output_active": llm_outputs,
                                    "raw_pixel_goal": raw_pixel_goal,
                                    "active_pixel_goal": list(pixel_goal),
                                    "noop_lookdown": bool(self.noop_lookdown),
                                    "pixel_goal_image_source": "current_rgbd" if self.noop_lookdown else "habitat_lookdown",
                                    "rgb_frame": occ_frames[-1]["rgb"] if occ_frames else None,
                                    "depth_frame": occ_frames[-1]["depth"] if occ_frames else None,
                                }
                            )
                            append_jsonl(occ_goal_log, occ_record)
                            occ_goal_idx += 1
                            print(
                                "occ_goal",
                                occ_record["status"],
                                "raw",
                                raw_pixel_goal,
                                "active",
                                pixel_goal,
                                "reason",
                                occ_record["reason"],
                                flush=True,
                            )

                        # look down --> horizontal
                        if not self.noop_lookdown:
                            self.env.step(action_code.LOOKUP)
                            self.env.step(action_code.LOOKUP)

                        local_actions = []
                        pixel_values = inputs.pixel_values
                        image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                        with torch.no_grad():
                            traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                        # prepocess align with navdp
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                        pix_goal_image = copy.copy(image_dp)
                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                        pix_goal_depth = copy.copy(depth_dp)
                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)

                        if self.enable_traj_occ_rerank:
                            dp_actions, traj_occ_record = rerank_trajectories_with_local_depth(
                                dp_actions,
                                front_depth_m,
                                intrinsic_matrix[0, 0],
                                intrinsic_matrix[1, 1],
                                intrinsic_matrix[0, 2],
                                intrinsic_matrix[1, 2],
                                max_depth_m=self.traj_occ_max_depth_m,
                                stride=self.traj_occ_stride,
                                floor_percentile=self.traj_occ_floor_percentile,
                                free_height_tol_m=self.traj_occ_free_height_tol_m,
                                occupied_height_m=self.traj_occ_occupied_height_m,
                                robot_radius_m=self.traj_occ_robot_radius_m,
                                safety_margin_m=self.traj_occ_safety_margin_m,
                                support_radius_m=self.traj_occ_support_radius_m,
                                min_support_fraction=self.traj_occ_min_support_fraction,
                                horizon_steps=self.traj_occ_horizon_steps,
                            )
                            traj_occ_record.update(
                                {
                                    "scene_id": scene_id,
                                    "episode_id": int(episode_id),
                                    "step_id": int(step_id),
                                    "phase": "new_pixel_goal",
                                }
                            )
                            append_jsonl(traj_occ_log, traj_occ_record)

                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]

                        action = local_actions[0]
                        if action == action_code.STOP:
                            pixel_goal = None
                            output_ids = None
                            arrive_gate_remaining = 0
                            arrive_forward_moves = 0
                            action = action_code.LEFT
                            observations, _, done, _ = self.env.step(action)
                            log_step_collision(self, step_collision_log, scene_id, episode_id, step_id, action)
                            step_id += 1
                            messages = []
                            continue
                        print('predicted goal', pixel_goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif pixel_goal is not None:
                    if len(local_actions) == 0:
                        # navdp
                        local_actions = []
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255

                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)

                        if self.enable_traj_occ_rerank:
                            dp_actions, traj_occ_record = rerank_trajectories_with_local_depth(
                                dp_actions,
                                front_depth_m,
                                intrinsic_matrix[0, 0],
                                intrinsic_matrix[1, 1],
                                intrinsic_matrix[0, 2],
                                intrinsic_matrix[1, 2],
                                max_depth_m=self.traj_occ_max_depth_m,
                                stride=self.traj_occ_stride,
                                floor_percentile=self.traj_occ_floor_percentile,
                                free_height_tol_m=self.traj_occ_free_height_tol_m,
                                occupied_height_m=self.traj_occ_occupied_height_m,
                                robot_radius_m=self.traj_occ_robot_radius_m,
                                safety_margin_m=self.traj_occ_safety_margin_m,
                                support_radius_m=self.traj_occ_support_radius_m,
                                min_support_fraction=self.traj_occ_min_support_fraction,
                                horizon_steps=self.traj_occ_horizon_steps,
                            )
                            traj_occ_record.update(
                                {
                                    "scene_id": scene_id,
                                    "episode_id": int(episode_id),
                                    "step_id": int(step_id),
                                    "phase": "local_replan",
                                }
                            )
                            append_jsonl(traj_occ_log, traj_occ_record)

                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]
                        print("local_actions", local_actions)
                        action = local_actions.pop(0)
                    else:
                        action = local_actions.pop(0)

                    forward_action += 1
                    if action == action_code.FORWARD:
                        arrive_forward_moves += 1
                    # method_5_arrive: when a vertical-search correction set
                    # arrive_gate_remaining, don't let a corrected goal that
                    # genuinely needs more than MAX_STEPS forward actions be
                    # abandoned before the agent can reach it -- but gate on
                    # arrive_forward_moves (FORWARD actions only), not the
                    # combined forward_action count, since turn-in-place
                    # actions produce no displacement toward the corrected
                    # goal's estimated depth and must not consume this budget
                    # (see ARRIVE_GATE_TOTAL_STEP_MULTIPLIER's docstring for
                    # the diagnosed case this fixes). A separate, more
                    # generous total-action cap still bounds wall-clock time
                    # if the route needs a lot of turning. Falls back to the
                    # original fixed MAX_STEPS whenever arrive_gate_remaining
                    # is 0 (no arrive-gated correction active this cycle).
                    if arrive_gate_remaining > 0:
                        budget_exceeded = arrive_forward_moves > arrive_gate_remaining
                        safety_cap_exceeded = forward_action > arrive_gate_remaining * ARRIVE_GATE_TOTAL_STEP_MULTIPLIER
                    else:
                        budget_exceeded = forward_action > MAX_STEPS
                        safety_cap_exceeded = False
                    if budget_exceeded or safety_cap_exceeded:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        arrive_gate_remaining = 0
                        arrive_forward_moves = 0
                        local_actions = []
                        continue
                    if action == action_code.STOP:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        arrive_gate_remaining = 0
                        arrive_forward_moves = 0
                        local_actions = []
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if pixel_goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if pixel_goal is not None:
                        if draw_pixel_goal:
                            cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)

                if action == action_code.LOOKDOWN:
                    if self.noop_lookdown:
                        observations, done = observations, done
                    else:
                        self.env.step(action)
                        observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    log_step_collision(self, step_collision_log, scene_id, episode_id, step_id, action)
                    step_id += 1
                    messages = []
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode progress.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
            }
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']
            # Ground-truth physics collision count from Habitat's own
            # CollisionsMeasurement (self._sim.previous_step_collided,
            # registered in __init__ via CollisionsMeasurementConfig) --
            # independent of our own occupancy-grid safety judgments, so this
            # is the actual metric to use for "did correction make things
            # safer", not a proxy for it.
            if 'collisions' in metrics and metrics['collisions'] is not None:
                result['collisions'] = metrics['collisions'].get('count')

            # save current progress
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")

            # save video
            if self.save_video and (self.save_all_videos or metrics['success'] == 1.0):
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()
            if self.save_occ_sequence:
                os.makedirs(occ_episode_dir, exist_ok=True)
                summary = {
                    "source": "habitat_sim",
                    "scene_id": scene_id,
                    "episode_id": int(episode_id),
                    "num_saved": len(occ_frames),
                    "intrinsics": [
                        [float(intrinsic_matrix[0, 0]), 0.0, float(intrinsic_matrix[0, 2])],
                        [0.0, float(intrinsic_matrix[1, 1]), float(intrinsic_matrix[1, 2])],
                        [0.0, 0.0, 1.0],
                    ],
                    "frames": occ_frames,
                }
                with open(os.path.join(occ_episode_dir, "manifest.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                with open(os.path.join(occ_episode_dir, "nvblox_sequence_summary.json"), "w", encoding="utf-8") as f:
                    json.dump({"intrinsics": summary["intrinsics"], "num_saved": len(occ_frames)}, f, indent=2)

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )

    def _run_eval_system2(self) -> tuple:
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)

            if self.episode_seed_offset is not None:
                seed_str = f"{scene_id}_{episode_id}_{self.episode_seed_offset}"
                episode_seed = zlib.crc32(seed_str.encode("utf-8")) % (2**31)
                torch.manual_seed(episode_seed)
                random.seed(episode_seed)

            agent_state = self.env._env.sim.get_agent_state()
            rotation = agent_state.rotation
            translation = agent_state.position
            rotation_matrix = quaternion.as_rotation_matrix(rotation)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            agent = ShortestPathFollower(self.env._env.sim, 0.25, False)

            intrinsic_matrix = get_intrinsic_matrix(
                self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
            )

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )
            initial_height = self.env._env.sim.get_agent_state().position[1]

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            goal = None
            action = None
            messages = []

            done = False
            flag = False

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                camera_yaw = observations["compass"][0]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                agent_state = self.env._env.sim.get_agent_state()
                height = agent_state.position[1] - initial_height  # Habitat GPS makes west negative, so flip y
                camera_position = np.array([x, -y, self._camera_height + height])
                tf_camera_to_episodic = (
                    xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30)) @ get_axis_align_matrix()
                )

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                if len(action_seq) == 0 and goal is None:
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                        ).sequences

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]
                        draw_pixel_goal = True

                        # look down --> horizontal
                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                        goal = pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)

                        goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                        if not self.env._env.sim.pathfinder.is_navigable(np.array(goal)):
                            goal = np.array(self.env._env.sim.pathfinder.snap_point(np.array(goal)))

                        action = agent.get_next_action(goal)
                        if action == action_code.STOP:
                            goal = None
                            output_ids = None
                            action = action_code.LEFT  # random action to avoid deadlock
                            observations, _, done, _ = self.env.step(action)
                            step_id += 1
                            messages = []
                            continue
                        print('predicted goal', pixel_goal, goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif goal is not None:
                    action = agent.get_next_action(goal)
                    action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                    action = action[0] if hasattr(action, "__len__") else action

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                    if action == action_code.STOP:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if draw_pixel_goal:
                        cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)

                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode result.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
            }
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']
            if 'collisions' in metrics and metrics['collisions'] is not None:
                result['collisions'] = metrics['collisions'].get('count')

            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")
            if self.save_video and (self.save_all_videos or metrics['success'] == 1.0):
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )
