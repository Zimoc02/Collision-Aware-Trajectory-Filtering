#!/usr/bin/env python3
"""Receive UDP cmd_vel from Go2W ROS2 bridge and execute with PC SportClient."""

import argparse
import ipaddress
import os
import socket
import struct
import sys
import time


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
      <OutputFile>/tmp/cdds_internnav_pc_udp_driver.LOG</OutputFile>
    </Tracing>
  </Domain>
</CycloneDDS>'''

    ChannelFactoryInitialize(0, iface)
    client = SportClient()
    client.SetTimeout(3.0)
    client.Init()
    return client


def clip(value, limit):
    return max(-limit, min(limit, float(value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--iface", default=os.environ.get("UNITREE_NETWORK_INTERFACE", "SERVER_IP"))
    parser.add_argument("--sdk-path", default=os.environ.get("UNITREE_SDK2_PYTHON", "unitree_sdk2_python"))
    parser.add_argument("--max-linear", type=float, default=0.08)
    parser.add_argument("--max-angular", type=float, default=0.12)
    parser.add_argument("--timeout-sec", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sport = None
    if not args.dry_run:
        sport = init_sport_client(args.iface, args.sdk_path)
        print(f"[SDK] SportClient ready on {args.iface}", flush=True)
    else:
        print("[DRY_RUN] Not sending SportClient commands", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.listen_ip, args.port))
    sock.setblocking(False)
    print(f"[Network] UDP listening on {args.listen_ip}:{args.port}", flush=True)

    vx = 0.0
    wz = 0.0
    last_packet = 0.0
    was_moving = False

    try:
        while True:
            got_packet = False
            try:
                while True:
                    data, addr = sock.recvfrom(1024)
                    raw_vx, raw_wz = struct.unpack("ff", data[:8])
                    vx = clip(raw_vx, args.max_linear)
                    wz = clip(raw_wz, args.max_angular)
                    got_packet = True
                    print(f"[UDP] from {addr[0]} vx={vx:.4f} wz={wz:.4f}", flush=True)
            except BlockingIOError:
                pass

            if got_packet:
                last_packet = time.monotonic()

            if time.monotonic() - last_packet > args.timeout_sec:
                vx = 0.0
                wz = 0.0

            moving = abs(vx) > 1e-3 or abs(wz) > 1e-3
            if args.dry_run:
                pass
            elif moving:
                if not was_moving:
                    print("[Control] UDP active", flush=True)
                sport.Move(vx, 0.0, wz)
            elif was_moving:
                print("[Control] UDP timeout/zero, StopMove", flush=True)
                sport.StopMove()

            was_moving = moving
            time.sleep(0.04)
    except KeyboardInterrupt:
        print("\n[Exit]", flush=True)
    finally:
        if sport is not None:
            print("[Exit] StopMove", flush=True)
            sport.StopMove()


if __name__ == "__main__":
    main()
