"""Shared thermal/power helpers for Jetson scripts (benchmark_jetson.py,
case_report_jetson.py). Not shared with pi_thermal.py - Jetson has several
named thermal zones instead of one, and no vcgencmd (that's Broadcom/Pi
specific); tegrastats is the Jetson equivalent for power draw. Degrades to
no-ops off-Jetson (no /sys/class/thermal zones, no tegrastats binary).
"""

import glob
import subprocess
import time
from pathlib import Path


def read_thermal_zones():
    """Every named thermal zone the kernel exposes (cpu, gpu, soc, etc.),
    not just zone0 - Jetson has several and zone0 is not reliably the CPU.
    Returns a dict of {zone_name: temp_c}; empty off-device.
    """
    zones = {}
    for type_path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/type")):
        temp_path = type_path.replace("/type", "/temp")
        try:
            name = Path(type_path).read_text().strip()
            milli = int(Path(temp_path).read_text().strip())
            zones[name] = milli / 1000
        except (FileNotFoundError, ValueError):
            continue
    return zones


def read_tegrastats_line(timeout=3):
    """One line of `tegrastats` output (power rails, GPU/CPU load), or
    None if the binary isn't on PATH (e.g. running this off-Jetson).
    """
    try:
        proc = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=subprocess.PIPE, universal_newlines=True)
    except FileNotFoundError:
        return None
    try:
        line = proc.stdout.readline().strip()
    except Exception:
        line = None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
    return line or None


def wait_for_safe_temp(max_temp, poll_seconds=5):
    zones = read_thermal_zones()
    if not zones:
        return
    hottest_name, hottest_temp = max(zones.items(), key=lambda kv: kv[1])
    if hottest_temp < max_temp:
        return
    print(f"  {hottest_name} at {hottest_temp:.1f}C (limit {max_temp:.1f}C) - pausing to cool down...")
    while hottest_temp >= max_temp:
        time.sleep(poll_seconds)
        zones = read_thermal_zones()
        if not zones:
            break
        hottest_name, hottest_temp = max(zones.items(), key=lambda kv: kv[1])
    print(f"  Back to {hottest_temp:.1f}C, resuming.\n")
