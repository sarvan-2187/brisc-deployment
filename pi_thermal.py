"""Shared SoC-temperature / throttle-status helpers for Raspberry Pi
scripts (benchmark.py, batch_infer.py). Degrades to no-ops off-Pi (no
/sys/class/thermal, no vcgencmd).
"""

import subprocess
import time
from pathlib import Path

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
THROTTLE_FLAGS = {
    0: "under-voltage now",
    1: "arm frequency capped now",
    2: "currently throttled",
    3: "soft temperature limit active now",
    16: "under-voltage has occurred",
    17: "arm frequency capping has occurred",
    18: "throttling has occurred",
    19: "soft temperature limit has occurred",
}


def read_temp_c():
    try:
        return int(THERMAL_ZONE.read_text().strip()) / 1000
    except (FileNotFoundError, ValueError):
        return None


def read_throttled_flags():
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2,
        )
        value = int(out.stdout.strip().split("=")[1], 16)
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        return None
    return [desc for bit, desc in THROTTLE_FLAGS.items() if value & (1 << bit)]


def wait_for_safe_temp(max_temp, poll_seconds=5):
    temp = read_temp_c()
    if temp is None or temp < max_temp:
        return
    print(f"  SoC at {temp:.1f}C (limit {max_temp:.1f}C) - pausing to cool down...")
    while temp is not None and temp >= max_temp:
        time.sleep(poll_seconds)
        temp = read_temp_c()
    if temp is not None:
        print(f"  Back to {temp:.1f}C, resuming.\n")
