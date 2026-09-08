import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent


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

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''


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


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time
    start_time = time.time()

    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    print(f"read http data cost {time.time() - start_time}")

    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    instruction = data.get('instruction') or args.instruction
    policy_init = data['reset']
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
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
        json_output['discrete_action'] = dual_sys_output.output_action
    else:
        trajectory = dual_sys_output.output_trajectory.tolist()
        json_output['trajectory'] = trajectory
        if args.return_discrete_fallback:
            fallback = trajectory_to_discrete_fallback(trajectory)
            if fallback is not None:
                json_output['discrete_action_fallback'] = fallback
        if dual_sys_output.output_pixel is not None:
            json_output['pixel_goal'] = dual_sys_output.output_pixel

    t1 = time.time()
    generate_time = t1 - t0
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return jsonify(json_output)


if __name__ == '__main__':

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
    parser.add_argument("--traj_predict_steps", type=int, default=16)
    parser.add_argument("--traj_inference_steps", type=int, default=4)
    parser.add_argument("--traj_num_samples", type=int, default=4)
    parser.add_argument(
        "--instruction",
        type=str,
        default="Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor",
    )
    parser.add_argument("--instruction_file", type=str, default=None)
    args = parser.parse_args()
    if args.instruction_file:
        with open(args.instruction_file, 'r') as f:
            args.instruction = f.read().strip()

    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    agent = InternVLAN1AsyncAgent(args)
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

    app.run(host='0.0.0.0', port=args.port)
