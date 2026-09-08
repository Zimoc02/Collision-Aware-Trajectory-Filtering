#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import stat
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import paramiko


RESULT_RE = re.compile(r"RESULT_DIR=(\S+)")


def set_server_occ_rerank(server_url, enabled):
    """Toggle occ V2 re-rank on the trajectory server before the run (no server restart)."""
    parts = urlsplit(server_url)
    toggle_url = urlunsplit((parts.scheme, parts.netloc, "/occ_rerank", "", ""))
    body = json.dumps({"enabled": bool(enabled)}).encode("utf-8")
    req = urllib.request.Request(
        toggle_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = resp.read().decode("utf-8", errors="replace").strip()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to toggle occ re-rank on server ({toggle_url}): {exc}")
    print(f"[server] occ V2 re-rank -> {'ON' if enabled else 'OFF'}  {payload}")


def parse_target(target):
    if "@" in target:
        username, host = target.split("@", 1)
    else:
        username, host = "unitree", target
    return username, host


def connect(target, password):
    username, host = parse_target(target)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        username=username,
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=10,
    )
    return client


def run_streaming(client, command):
    transport = client.get_transport()
    channel = transport.open_session()
    channel.exec_command(command)
    output = []
    while True:
        if channel.recv_ready():
            data = channel.recv(4096).decode("utf-8", errors="replace")
            output.append(data)
            sys.stdout.write(data)
            sys.stdout.flush()
        if channel.recv_stderr_ready():
            data = channel.recv_stderr(4096).decode("utf-8", errors="replace")
            output.append(data)
            sys.stderr.write(data)
            sys.stderr.flush()
        if channel.exit_status_ready():
            while channel.recv_ready():
                data = channel.recv(4096).decode("utf-8", errors="replace")
                output.append(data)
                sys.stdout.write(data)
            while channel.recv_stderr_ready():
                data = channel.recv_stderr(4096).decode("utf-8", errors="replace")
                output.append(data)
                sys.stderr.write(data)
            return channel.recv_exit_status(), "".join(output)


def run_capture(client, command):
    stdin, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out, end="")
    if err:
        print(err, end="", file=sys.stderr)
    return code, out + err


def sftp_download_dir(sftp, remote_dir, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in sftp.listdir_attr(remote_dir):
        remote_path = f"{remote_dir.rstrip('/')}/{item.filename}"
        local_path = local_dir / item.filename
        if stat.S_ISDIR(item.st_mode):
            count += sftp_download_dir(sftp, remote_path, local_path)
        else:
            sftp.get(remote_path, str(local_path))
            count += 1
    return count


def build_remote_client_command(args):
    mode = "--execute" if args.execute else "--dry-run"
    argv = [
        "/usr/bin/python3",
        "-u",
        "http_internvla_client_safe.py",
        "--server-url",
        args.server_url,
        "--instruction",
        args.instruction,
        mode,
        "--max-runtime-sec",
        str(args.max_runtime_sec),
        "--max-linear",
        str(args.max_linear),
        "--max-angular",
        str(args.max_angular),
        "--result-root",
        args.remote_result_root,
    ]
    command = " ".join(shlex.quote(p) for p in argv)
    return (
        "bash -lc "
        + shlex.quote(
            " && ".join(
                [
                    "cd /home/unitree/onboard_original",
                    "source ./setup_original_env.sh",
                    command,
                ]
            )
        )
    )


def important_files(local_dir):
    names = [
        "summary.json",
        "executed_path.png",
        "latest_plan.png",
        "plans_over_time.png",
        "cmd_vel_timeline.png",
        "trajectory_steps.jsonl",
        "cmd_vel.jsonl",
        "run_config.json",
    ]
    return [name for name in names if (local_dir / name).exists()]


def main():
    parser = argparse.ArgumentParser(description="Run the Go2W original client, pull results to PC, then clean Go2W.")
    parser.add_argument("--go2-host", default=os.environ.get("GO2W_HOST", "unitree@ROBOT_IP"))
    parser.add_argument("--password", default=os.environ.get("GO2W_PASSWORD"))
    parser.add_argument("--server-url", default=os.environ.get("INTERNVLA_SERVER_URL", "http://SERVER_IP:5802/eval_dual"))
    parser.add_argument("--instruction", default="Navigate to the red chair and stop.")
    parser.add_argument("--execute", action="store_true", help="Publish cmd_vel on Go2W. Default is dry-run.")
    parser.add_argument("--yes", action="store_true", help="Skip the local GO confirmation for --execute.")
    parser.add_argument("--max-runtime-sec", type=float, default=12.0)
    parser.add_argument("--max-linear", type=float, default=0.15)
    parser.add_argument("--max-angular", type=float, default=0.10)
    parser.add_argument(
        "--occ-rerank",
        dest="occ_rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Turn occ V2 re-rank ON (--occ-rerank) or OFF (--no-occ-rerank) on the server "
        "before this run, without restarting it. Omit to leave the server's current setting alone.",
    )
    parser.add_argument("--remote-result-root", default="/home/unitree/internnav_original_results")
    parser.add_argument("--local-result-root", default=str(Path.cwd() / "results"))
    args = parser.parse_args()

    if args.execute and not args.yes:
        print(
            "This will send low-speed movement commands through the current Go2W bridge.\n"
            f"server={args.server_url}\n"
            f"instruction={args.instruction}\n"
            f"limits: vx<={args.max_linear}, wz<={args.max_angular}, runtime={args.max_runtime_sec}s"
        )
        if input('Type "GO" to execute: ').strip() != "GO":
            print("Cancelled.")
            return 2

    if args.occ_rerank is not None:
        set_server_occ_rerank(args.server_url, args.occ_rerank)

    client = connect(args.go2_host, args.password)
    try:
        print("[go2w] Running original client...")
        code, output = run_streaming(client, build_remote_client_command(args))
        match = RESULT_RE.search(output)
        if not match:
            raise SystemExit("Could not find RESULT_DIR in Go2W output; nothing was deleted.")
        remote_result_dir = match.group(1)

        print(f"\n[go2w] Generating plots in {remote_result_dir} ...")
        plot_cmd = "bash -lc " + shlex.quote(
            "cd /home/unitree/onboard_original && /usr/bin/python3 plot_original_result.py "
            + shlex.quote(remote_result_dir)
        )
        plot_code, _ = run_capture(client, plot_cmd)
        if plot_code != 0:
            raise SystemExit("Plot generation failed; leaving Go2W result dir in place.")

        local_name = f"original_{Path(remote_result_dir).name}"
        local_dir = Path(args.local_result_root) / local_name
        print(f"[pc] Pulling results to {local_dir} ...")
        with client.open_sftp() as sftp:
            files = sftp_download_dir(sftp, remote_result_dir, local_dir)
        if files <= 0:
            raise SystemExit("Downloaded zero files; leaving Go2W result dir in place.")

        print(f"[go2w] Deleting remote result dir {remote_result_dir} ...")
        rm_code, _ = run_capture(client, "rm -rf -- " + shlex.quote(remote_result_dir))
        if rm_code != 0:
            raise SystemExit("Download succeeded, but remote cleanup failed.")

        print(f"\nLOCAL_RESULT_DIR={local_dir}")
        for name in important_files(local_dir):
            print(f"- {local_dir / name}")
        if code != 0:
            print(f"\nWarning: Go2W client exited with code {code}, but saved results were pulled.")
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
