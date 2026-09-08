#!/usr/bin/env python3
"""Run one Go2W InternNav step with local-PC Unitree control.

The Go2W only captures RealSense frames and asks the InternNav HTTP server for
an action. SportClient commands are sent from this PC, matching the older
OmniVLA path that was confirmed to move the robot reliably.
"""

import argparse
import ipaddress
import json
import os
import posixpath
import re
import shlex
import sys
import time
from pathlib import Path

import paramiko


REMOTE_CAPTURE_SCRIPT = r'''#!/usr/bin/env python3
import io
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import requests
import rospy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from PIL import Image as PILImage
from sensor_msgs.msg import Image

SERVER_URL = sys.argv[1]
OUT_ROOT = sys.argv[2]
INSTRUCTION = sys.argv[3]
STEP_IDX = int(sys.argv[4]) if len(sys.argv) > 4 else 0
RESET = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True
RGB_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'


class CaptureInferOnce:
    def __init__(self):
        self.bridge = CvBridge()
        self.done = False
        self.out_dir = os.path.join(
            OUT_ROOT, datetime.now().strftime('%Y%m%d_%H%M%S') + '_capture_infer_once'
        )
        os.makedirs(self.out_dir, exist_ok=True)
        rospy.loginfo('Saving capture/infer result to %s', self.out_dir)
        rgb_sub = Subscriber(RGB_TOPIC, Image)
        depth_sub = Subscriber(DEPTH_TOPIC, Image)
        self.sync = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.2)
        self.sync.registerCallback(self.callback)

    def callback(self, rgb_msg, depth_msg):
        if self.done:
            return
        self.done = True
        started = time.time()
        result = {
            'mode': 'CAPTURE_INFER_ONLY',
            'instruction': INSTRUCTION,
            'server_url': SERVER_URL,
            'step_idx': STEP_IDX,
            'reset': RESET,
        }
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) / 1000.0
            else:
                depth_m = depth.astype(np.float32)
            depth_for_server = np.clip(depth_m * 10000.0, 0, 65535).astype(np.uint16)

            PILImage.fromarray(rgb).save(os.path.join(self.out_dir, 'rgb.jpg'), format='JPEG')
            PILImage.fromarray(depth_for_server).save(
                os.path.join(self.out_dir, 'depth_server_units.png'), format='PNG'
            )
            np.save(os.path.join(self.out_dir, 'depth_meters.npy'), depth_m)

            rgb_buf = io.BytesIO()
            PILImage.fromarray(rgb).save(rgb_buf, format='JPEG')
            rgb_buf.seek(0)
            depth_buf = io.BytesIO()
            PILImage.fromarray(depth_for_server).save(depth_buf, format='PNG')
            depth_buf.seek(0)
            files = {
                'image': ('rgb.jpg', rgb_buf, 'image/jpeg'),
                'depth': ('depth.png', depth_buf, 'image/png'),
            }
            data = {'reset': RESET, 'idx': STEP_IDX, 'instruction': INSTRUCTION}
            rospy.loginfo('Posting frame to %s', SERVER_URL)
            response = requests.post(
                SERVER_URL, files=files, data={'json': json.dumps(data)}, timeout=180
            )
            result['http_status'] = response.status_code
            result['response_text'] = response.text[:1000]
            response.raise_for_status()
            result['response_json'] = response.json()
        except Exception as exc:
            result['exception'] = repr(exc)
        finally:
            result['elapsed_sec'] = time.time() - started
            with open(os.path.join(self.out_dir, 'capture_infer.json'), 'w') as f:
                json.dump(result, f, indent=2)
            print('REPORT_JSON=' + json.dumps(result, sort_keys=True))
            print('OUT_DIR=' + self.out_dir)
            rospy.signal_shutdown('capture/infer complete')


if __name__ == '__main__':
    rospy.init_node('internnav_capture_infer_once', anonymous=True)
    CaptureInferOnce()
    rospy.spin()
'''


def split_user_host(value):
    if "@" not in value:
        raise ValueError("--go2-host must look like user@host")
    user, host = value.split("@", 1)
    return user, host


def ssh_connect(args):
    user, host = split_user_host(args.go2_host)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        username=user,
        password=args.password,
        look_for_keys=False,
        allow_agent=False,
        timeout=8,
    )
    return client


def run_ssh(client, command, timeout=240):
    stdin, stdout, stderr = client.exec_command(command, get_pty=True, timeout=timeout)
    channel = stdout.channel
    chunks = []
    while True:
        if channel.recv_ready():
            data = channel.recv(4096)
            text = data.decode("utf-8", "replace")
            print(text, end="")
            chunks.append(text)
        if channel.exit_status_ready():
            while channel.recv_ready():
                data = channel.recv(4096)
                text = data.decode("utf-8", "replace")
                print(text, end="")
                chunks.append(text)
            break
        time.sleep(0.1)
    return channel.recv_exit_status(), "".join(chunks)


def upload_remote_script(client, path):
    sftp = client.open_sftp()
    try:
        with sftp.file(path, "w") as f:
            f.write(REMOTE_CAPTURE_SCRIPT)
        sftp.chmod(path, 0o755)
    finally:
        sftp.close()


def copy_remote_dir(client, remote_dir, local_root):
    local_dir = Path(local_root).expanduser() / ("go2w_" + posixpath.basename(remote_dir))
    local_dir.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        for name in sftp.listdir(remote_dir):
            remote_path = posixpath.join(remote_dir, name)
            local_path = local_dir / name
            try:
                sftp.get(remote_path, str(local_path))
            except OSError:
                pass
    finally:
        sftp.close()
    return local_dir


def command_for_action(actions, max_linear, max_angular, duration):
    if not actions:
        return None, "NO_ACTION"
    action = int(actions[0])
    if action == 0:
        return {"name": "STOP", "linear": 0.0, "angular": 0.0, "duration": 0.0}, None
    if action == 1:
        return {"name": "FORWARD", "linear": max_linear, "angular": 0.0, "duration": duration}, None
    if action == 2:
        return {"name": "TURN_LEFT", "linear": 0.0, "angular": max_angular, "duration": duration}, None
    if action == 3:
        return {"name": "TURN_RIGHT", "linear": 0.0, "angular": -max_angular, "duration": duration}, None
    return None, f"UNSAFE_ACTION_NOT_EXECUTED_{action}"


def init_sport_client(iface, sdk_path):
    sys.path.insert(0, sdk_path)
    import unitree_sdk2py.core.channel as channel
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    try:
        ipaddress.ip_address(iface)
    except ValueError:
        pass
    else:
        channel.ChannelConfigHasInterface = '''<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface address="$__IF_NAME__$" priority="default" multicast="default"/>
      </Interfaces>
    </General>
    <Tracing>
      <Verbosity>config</Verbosity>
      <OutputFile>/tmp/cdds_internnav.LOG</OutputFile>
    </Tracing>
  </Domain>
</CycloneDDS>'''

    ChannelFactoryInitialize(0, iface)
    client = SportClient()
    client.SetTimeout(3.0)
    client.Init()
    return client


def execute_move(client, linear, angular, duration, hz):
    if duration <= 0.0:
        print("Selected STOP; sending StopMove only.")
        print("SportClient.StopMove ret:", client.StopMove())
        return
    period = 1.0 / hz
    deadline = time.monotonic() + duration
    ret = None
    sent = 0
    while time.monotonic() < deadline:
        ret = client.Move(float(linear), 0.0, float(angular))
        sent += 1
        time.sleep(period)
    print(f"SportClient.Move ret: {ret} ({sent} command(s) sent)")
    print("SportClient.StopMove ret:", client.StopMove())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--go2-host", default=os.environ.get("GO2W_HOST", "unitree@ROBOT_IP"))
    parser.add_argument("--password", default=os.environ.get("GO2W_PASSWORD"))
    parser.add_argument("--server-url", default=os.environ.get("INTERNVLA_SERVER_URL", "http://SERVER_IP:5801/eval_dual"))
    parser.add_argument("--instruction", default="Turn slightly left in place, then stop.")
    parser.add_argument("--remote-out-root", default="/home/unitree/internnav_results")
    parser.add_argument("--local-result-root", default="results")
    parser.add_argument("--remote-script", default="/tmp/internnav_capture_infer_once.py")
    parser.add_argument("--iface", default=os.environ.get("UNITREE_NETWORK_INTERFACE", "SERVER_IP"))
    parser.add_argument("--sdk-path", default=os.environ.get("UNITREE_SDK2_PYTHON", "unitree_sdk2_python"))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-linear", type=float, default=0.08)
    parser.add_argument("--max-angular", type=float, default=0.03)
    parser.add_argument("--move-duration", type=float, default=0.20)
    parser.add_argument("--command-hz", type=float, default=10.0)
    parser.add_argument("--inter-step-delay", type=float, default=0.3)
    parser.add_argument("--reset-each-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    client = ssh_connect(args)
    try:
        upload_remote_script(client, args.remote_script)
        sport = None
        if args.execute:
            if not args.yes:
                confirm = input(
                    f'Type "GO" to execute up to {args.steps} low-speed step(s), or anything else to stop: '
                )
                if confirm != "GO":
                    print("Stopped by user.")
                    return
            sport = init_sport_client(args.iface, args.sdk_path)

        for step in range(args.steps):
            print(f"\n=== InternNav Go2W step {step + 1}/{args.steps} ===")
            command = (
                "bash -lc "
                + shlex.quote(
                    "source /opt/ros/noetic/setup.bash; "
                    f"python3 {shlex.quote(args.remote_script)} "
                    f"{shlex.quote(args.server_url)} "
                    f"{shlex.quote(args.remote_out_root)} "
                    f"{shlex.quote(args.instruction)} "
                    f"{step} "
                    f"{1 if args.reset_each_step or step == 0 else 0}"
                )
            )
            code, stdout = run_ssh(client, command)
            if code != 0:
                raise RuntimeError(f"remote capture/infer failed with exit code {code}")

            out_match = re.search(r"OUT_DIR=(\S+)", stdout)
            report_match = re.search(r"REPORT_JSON=(\{.*\})", stdout)
            if not out_match or not report_match:
                raise RuntimeError("remote output did not include OUT_DIR and REPORT_JSON")
            remote_dir = out_match.group(1)
            report = json.loads(report_match.group(1))
            local_dir = copy_remote_dir(client, remote_dir, args.local_result_root)

            response = report.get("response_json", {})
            actions = response.get("discrete_action", [])
            selected, blocked = command_for_action(actions, args.max_linear, args.max_angular, args.move_duration)
            report.update(
                {
                    "local_control_iface": args.iface,
                    "selected_command": selected,
                    "blocked_reason": blocked,
                    "move_sent": False,
                }
            )
            report_path = local_dir / "local_control_report.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(f"Local result dir: {local_dir}")
            print(f"Model actions: {actions}")
            print(f"Selected command: {selected}")

            if blocked is not None:
                print(f"Not executing: {blocked}")
                break
            if selected and selected["name"] == "STOP":
                print("Model selected STOP; stopping loop.")
                if sport is not None:
                    print("SportClient.StopMove ret:", sport.StopMove())
                    report["stopmove_sent"] = True
                    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                break
            if not args.execute:
                print("Dry-run: not sending command to Go2. Add --execute when ready.")
            else:
                execute_move(sport, selected["linear"], selected["angular"], selected["duration"], args.command_hz)
                report["move_sent"] = selected["duration"] > 0.0
                report["stopmove_sent"] = True
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                time.sleep(args.inter_step_delay)
    finally:
        client.close()
    if 'sport' in locals() and sport is not None:
        print("Final StopMove ret:", sport.StopMove())


if __name__ == "__main__":
    main()
