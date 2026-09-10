#!/usr/bin/env python3
"""Display raw pedal key events without starting collection or robot control."""

import argparse
from datetime import datetime
import errno
import fcntl
import os
import select
import signal
import struct
import time


INPUT_EVENT = struct.Struct("llHHi")
EVIOCGRAB = 0x40044590
KEY_LABELS = {code: str(code - 1) for code in range(2, 11)}
KEY_LABELS.update({11: "0", 30: "a", 48: "b", 46: "c"})


def interrupted(_signum, _frame):
    raise KeyboardInterrupt


def monitor(device):
    print("踏板按键检测：依次踩下四个键，观察显示顺序。", flush=True)
    print("仅显示输入，不启动采集或机械臂。Ctrl-C 或关闭窗口退出。", flush=True)
    print("检测期间独占踏板，按键只显示在此窗口。", flush=True)
    count = 0
    while True:
        print(f"等待踏板连接：{device}", flush=True)
        while True:
            try:
                fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
                break
            except FileNotFoundError:
                time.sleep(0.5)
        try:
            fcntl.ioctl(fd, EVIOCGRAB, 1)
            print("已连接，请踩踏板。", flush=True)
            pending = b""
            while True:
                if not select.select([fd], [], [], 0.5)[0]:
                    continue
                data = os.read(fd, INPUT_EVENT.size * 64)
                if not data:
                    break
                pending += data
                while len(pending) >= INPUT_EVENT.size:
                    raw, pending = pending[:INPUT_EVENT.size], pending[INPUT_EVENT.size:]
                    _sec, _usec, kind, code, value = INPUT_EVENT.unpack(raw)
                    if kind != 1 or value not in (0, 1):
                        continue
                    if value == 1:
                        count += 1
                    edge = "按下" if value == 1 else "松开"
                    label = KEY_LABELS.get(code, f"KEY_{code}")
                    print(f"{datetime.now():%H:%M:%S.%f}  {edge} [{label}]"
                          f"  键码={code}  累计按下={count}", flush=True)
        except OSError as exc:
            if exc.errno != errno.ENODEV:
                raise
            print("踏板已断开。", flush=True)
        finally:
            os.close(fd)  # Closing releases EVIOCGRAB, including on interruption.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/input/prometheus-pedal")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        monitor(args.device)
    except KeyboardInterrupt:
        print("\n检测已退出，踏板输入已释放。", flush=True)


if __name__ == "__main__":
    main()
