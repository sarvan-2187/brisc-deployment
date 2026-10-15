"""Instrumented re-runs for the paper 2141 revision (see brain-tumor/refactor-kit/RE-WORK-PI.md).

One CSV row per inference plus setup.json and summary.json per run, under
--out/<run_id>/. Reuses the deployed classify -> gate -> segment path
(case_report.classify_slice / segment_slice) and AdaptiveController, so the
timed work is exactly what benchmark_adaptive.py and the app run.

Subcommands:
  run      one sustained-load run:  --mode static-H | static-L | adaptive
  matrix   several runs, each started only once the SoC has cooled back to the
           idle baseline (each run is a fresh process, so peak RSS is per run)
  handoff  laptop side: time send_to_pi.py end to end for cases of N slices

Examples (on the Pi, inside the venv):
  python eval_dataset_sample.py --sample-size 1000          # full test split, seed 42
  python rerun_2141.py run --mode static-L --num-runs 20    # smoke test
  python rerun_2141.py matrix --jobs H,L,A,L,A,H --idle-minutes 10 --ambient-c 27 --cooling "heatsink+fan"
  python rerun_2141.py run --mode static-H --num-runs 50 --preload tier   # memory probe, one tier resident
  python rerun_2141.py run --mode adaptive --num-runs 400 --temp-engage-c 85 --temp-release-c 80 \
      --freq-cap-at 100 --freq-cap-khz 1200000 --freq-restore-at 250       # controller exercise (E4)
Micro-benchmark (E1, n = 100 over 20 images):
  python benchmark.py --manifest eval_sample_manifest.csv --num-images 20 --warmup 5 --runs 100 --out-csv results/micro.csv
Laptop side:
  python rerun_2141.py handoff --pi-host brisc-pi.local --pi-user pi --remote-dir ~/brisc-deployment
"""

import argparse
import csv
import glob
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from adaptive_controller import AdaptiveController, HIGH_ACCURACY, LIGHTWEIGHT
from benchmark import CKPT_DIR, CLASS_NAMES, load_cls_model, load_seg_model, percentile
from case_report import classify_slice, segment_slice
from eval_paired_accuracy import iou_dice, load_gt_mask, load_manifest
from pi_thermal import read_temp_c

MODES = {"static-H": HIGH_ACCURACY, "static-L": LIGHTWEIGHT, "adaptive": None}
JOB_CODES = {"H": "static-H", "L": "static-L", "A": "adaptive"}
FIELDS = [
    "run_id", "mode", "rep", "idx", "filename", "true_label", "tier_used",
    "cls_ms", "seg_ms", "total_ms", *[f"p_{c}" for c in CLASS_NAMES], "pred", "correct", "gated",
    "iou", "dice", "soc_temp_c", "arm_clock_hz", "throttled_hex", "rss_kb", "vmhwm_kb",
    "mem_available_kb", "swap_used_kb", "controller_state", "trigger_reason", "rolling_latency_ms",
    "aux_ms", "event",
]
THROTTLE_EVERY = 25
CPUFREQ = "/sys/devices/system/cpu/cpu{}/cpufreq/{}"


def sh(cmd):
    """stdout of a shell command, or None when it isn't available (e.g. off-Pi)."""
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def read_kv(path):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        key, _, val = line.partition(":") if ":" in line else line.partition(" ")
        parts = val.split()
        if parts and parts[0].isdigit():
            out[key.strip()] = int(parts[0])
    return out


def memory():
    st, mi = read_kv("/proc/self/status"), read_kv("/proc/meminfo")
    swap = mi["SwapTotal"] - mi["SwapFree"] if "SwapTotal" in mi else None
    return {"rss_kb": st.get("VmRSS"), "vmhwm_kb": st.get("VmHWM"),
            "mem_available_kb": mi.get("MemAvailable"), "swap_used_kb": swap}


def swap_counters():
    vm = read_kv("/proc/vmstat")
    return {k: vm.get(k) for k in ("pswpin", "pswpout")}


def arm_clock_hz():
    try:
        return int(Path(CPUFREQ.format(0, "scaling_cur_freq")).read_text()) * 1000
    except (OSError, ValueError):
        return None


def throttled_hex():
    out = sh("vcgencmd get_throttled")
    return out.split("=")[1] if out and "=" in out else None


def set_max_freq(khz):
    """Cap (or restore) the CPU clock on every core. Needs sudo for tee on the cpufreq files."""
    for path in glob.glob(CPUFREQ.format("*", "scaling_max_freq")):
        subprocess.run(["sudo", "-n", "tee", path], input=str(khz), text=True,
                       stdout=subprocess.DEVNULL, check=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_info(args, rows, controller):
    import segmentation_models_pytorch as smp
    import timm
    import torchvision
    return {
        "argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(), "python": sys.version.split()[0],
        "packages": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "timm": timm.__version__, "segmentation_models_pytorch": smp.__version__},
        "torch_threads": torch.get_num_threads(),
        "governor": sh(f"cat {CPUFREQ.format(0, 'scaling_governor')}"),
        "cpuinfo_max_freq_khz": sh(f"cat {CPUFREQ.format(0, 'cpuinfo_max_freq')}"),
        "arm_freq": sh("vcgencmd get_config arm_freq"),
        "swap": {"swapon": sh("swapon --show"), "dphys": sh("grep -v '^#' /etc/dphys-swapfile"),
                 "zram": sh("zramctl")},
        "git_commit": sh("git rev-parse HEAD"),
        "checkpoints_sha256": {p.name: sha256(p) for p in sorted(CKPT_DIR.glob("*_best.pt"))},
        "gate_threshold": args.threshold,
        "controller": None if controller is None else {
            "latency_threshold_ms": controller.latency_threshold_ms, "temp_engage_c": controller.temp_threshold_c,
            "temp_release_c": controller.temp_release_c, "window_size": controller.window_size,
            "consecutive_checks": controller.consecutive_trigger_count, "checked": "before every inference"},
        "manifest": args.manifest, "manifest_sha256": sha256(args.manifest),
        "slice_order": [r["filename"] for r in rows],
        "start_temp_c": read_temp_c(), "start_throttled_hex": throttled_hex(), "start_swap": swap_counters(),
        "ambient_c": args.ambient_c, "cooling": args.cooling, "notes": args.notes,
        "freq_cap": {"at": args.freq_cap_at, "khz": args.freq_cap_khz, "restore_at": args.freq_restore_at},
        "preload": args.preload,
    }


def latency_summary(values):
    if not values:
        return None
    v = sorted(values)
    return {"n": len(v), "mean": statistics.mean(v), "median": statistics.median(v),
            "p95": percentile(v, 95), "min": v[0], "max": v[-1]}


def summarize(log, setup, end):
    ok = [r for r in log if r["pred"] is not None]
    temps = [r["soc_temp_c"] for r in ok if r["soc_temp_c"] is not None]
    tiers = [r["tier_used"] for r in ok]
    mem = [r for r in ok if r["mem_available_kb"] is not None]
    seg = [r for r in ok if r["gated"] and r["iou"] is not None]
    return {
        "n": len(ok), "accuracy": sum(r["correct"] for r in ok) / len(ok) if ok else None,
        "latency_ms": {"workload": latency_summary([r["total_ms"] for r in ok]),
                       "segmented_path": latency_summary([r["total_ms"] for r in ok if r["gated"]]),
                       "fast_path": latency_summary([r["total_ms"] for r in ok if not r["gated"]]),
                       "cls": latency_summary([r["cls_ms"] for r in ok]),
                       "seg_when_run": latency_summary([r["seg_ms"] for r in ok if r["gated"]])},
        "fast_path_count": sum(1 for r in ok if not r["gated"]),
        "mean_iou_segmented": statistics.mean(r["iou"] for r in seg) if seg else None,
        "temp_c": {"start": setup["start_temp_c"], "mean": statistics.mean(temps) if temps else None,
                   "peak": max(temps) if temps else None},
        "throttled_hex": {"start": setup["start_throttled_hex"], "end": end["throttled_hex"],
                          "nonzero_rows": sum(1 for r in ok if r["throttled_hex"] not in (None, "0x0"))},
        "min_arm_clock_hz": min((r["arm_clock_hz"] for r in ok if r["arm_clock_hz"]), default=None),
        "tier_switches": sum(1 for a, b in zip(tiers, tiers[1:]) if a != b),
        "pct_lightweight": 100 * tiers.count("lightweight") / len(tiers) if tiers else None,
        "peak_rss_kb_vmhwm": end["vmhwm_kb"],
        "min_mem_available_kb": min((r["mem_available_kb"] for r in mem), default=None),
        "peak_swap_used_kb": max((r["swap_used_kb"] for r in mem), default=None),
        "swap_counters": {"start": setup["start_swap"], "end": end["swap"]},
        "aux_ms_mean": statistics.mean(r["aux_ms"] for r in ok) if ok else None,
        "aborted": end["aborted"], "wall_s": end["wall_s"],
    }


def cmd_run(args):
    rows = load_manifest(args.manifest)[:args.num_runs]
    pair_fixed = MODES[args.mode]
    controller = None
    run_id = f"{args.mode}_rep{args.rep}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "adaptive":
        controller = AdaptiveController(
            latency_threshold_ms=args.latency_threshold_ms, temp_threshold_c=args.temp_engage_c,
            temp_release_c=args.temp_release_c, window_size=args.window, log_path=out / "decision_log.csv")

    needed = (HIGH_ACCURACY, LIGHTWEIGHT) if args.preload == "all" or pair_fixed is None else (pair_fixed,)
    print(f"[{run_id}] loading {sorted({m for p in needed for m in p.values()})}")
    cls_models = {p["classifier"]: load_cls_model(p["classifier"]) for p in needed}
    seg_models = {p["segmenter"]: load_seg_model(p["segmenter"]) for p in needed}

    setup = setup_info(args, rows, controller)
    (out / "setup.json").write_text(json.dumps(setup, indent=2))

    log, capped, aborted = [], False, None
    t_start = time.monotonic()
    f = open(out / "inferences.csv", "w", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    try:
        for idx, r in enumerate(rows):
            event = ""
            if args.freq_cap_at is not None and idx == args.freq_cap_at:
                set_max_freq(args.freq_cap_khz)
                capped, event = True, f"freq_cap_{args.freq_cap_khz}khz"
            if capped and args.freq_restore_at is not None and idx == args.freq_restore_at:
                set_max_freq(setup["cpuinfo_max_freq_khz"])
                capped, event = False, "freq_restore"

            pair = controller.select_pair() if controller else pair_fixed
            tier = "high_accuracy" if pair == HIGH_ACCURACY else "lightweight"

            t0 = time.perf_counter()
            image, probs, top_class, _conf, verdict = classify_slice(
                Path(r["image_path"]), cls_models[pair["classifier"]], args.threshold)
            cls_ms = (time.perf_counter() - t0) * 1000
            mask, seg_ms = None, None
            if verdict == "tumor":
                t1 = time.perf_counter()
                mask = segment_slice(image, seg_models[pair["segmenter"]])
                seg_ms = (time.perf_counter() - t1) * 1000
            total_ms = cls_ms + (seg_ms or 0.0)
            if controller:
                controller.record_inference(total_ms)

            # Everything below is outside the timed region; aux_ms reports its cost.
            ta = time.perf_counter()
            iou = dice = None
            if mask is not None and r["has_mask"]:
                iou, dice = iou_dice(mask, load_gt_mask(r["mask_path"]))
            temp = read_temp_c()
            row = {
                "run_id": run_id, "mode": args.mode, "rep": args.rep, "idx": idx, "filename": r["filename"],
                "true_label": r["true_class"], "tier_used": tier, "cls_ms": round(cls_ms, 2),
                "seg_ms": None if seg_ms is None else round(seg_ms, 2), "total_ms": round(total_ms, 2),
                **{f"p_{c}": round(float(p), 6) for c, p in zip(CLASS_NAMES, probs)},
                "pred": top_class, "correct": top_class == r["true_class"], "gated": verdict == "tumor",
                "iou": iou, "dice": dice, "soc_temp_c": temp, "arm_clock_hz": arm_clock_hz(),
                "throttled_hex": throttled_hex() if idx % THROTTLE_EVERY == 0 else None, **memory(),
                "controller_state": controller.current_tier if controller else "",
                "trigger_reason": controller.last_reason if controller else "",
                "rolling_latency_ms": round(controller.avg_latency_ms, 2) if controller else None,
                "event": event,
            }
            row["aux_ms"] = round((time.perf_counter() - ta) * 1000, 2)
            writer.writerow(row)
            f.flush()
            log.append(row)
            print(f"\r  {idx + 1}/{len(rows)} {tier:13s} {total_ms:7.1f} ms  "
                  f"{'' if temp is None else f'{temp:.1f}C'}  {event}", end="", flush=True)
            if temp is not None and temp > args.abort_c:
                aborted = f"SoC {temp:.1f}C > {args.abort_c}C at idx {idx}"
                print(f"\nABORT: {aborted}")
                break
    finally:
        f.close()
        if capped:
            set_max_freq(setup["cpuinfo_max_freq_khz"])
    print()

    end = {"throttled_hex": throttled_hex(), "swap": swap_counters(), "vmhwm_kb": memory()["vmhwm_kb"],
           "aborted": aborted, "wall_s": time.monotonic() - t_start}
    summary = summarize(log, setup, end)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lat = summary["latency_ms"]["workload"]
    print(f"[{run_id}] n={summary['n']} acc={summary['accuracy']:.4f} workload mean {lat['mean']:.1f} ms, "
          f"peak temp {summary['temp_c']['peak']}, switches {summary['tier_switches']} -> {out}")
    return out


def wait_for_baseline(baseline, tol, max_wait_s):
    t0 = time.monotonic()
    while True:
        temp = read_temp_c()
        if temp is None or abs(temp - baseline) <= tol:
            return temp
        if time.monotonic() - t0 > max_wait_s:
            print(f"  cooldown timed out at {temp:.1f}C (baseline {baseline:.1f}C); starting anyway")
            return temp
        print(f"\r  cooling: {temp:.1f}C -> {baseline:.1f}C +/- {tol}", end="", flush=True)
        time.sleep(10)


def cmd_matrix(args):
    jobs = [JOB_CODES[j.strip()] for j in args.jobs.split(",")]
    baseline = args.baseline_c
    if baseline is None:
        print(f"Idling {args.idle_minutes} min to measure the baseline temperature...")
        time.sleep(args.idle_minutes * 60)
        samples = []
        for _ in range(6):
            samples.append(read_temp_c())
            time.sleep(10)
        samples = [s for s in samples if s is not None]
        baseline = statistics.mean(samples) if samples else None
    print(f"Baseline: {baseline}C; jobs: {jobs}")
    reps = {}
    for mode in jobs:
        reps[mode] = reps.get(mode, 0) + 1
        if baseline is not None:
            wait_for_baseline(baseline, args.tolerance_c, args.max_wait_s)
            print()
        cmd = [sys.executable, __file__, "run", "--mode", mode, "--rep", str(reps[mode]),
               "--num-runs", str(args.num_runs), "--manifest", args.manifest, "--out", args.out,
               "--threshold", str(args.threshold), "--cooling", args.cooling or "",
               "--notes", f"matrix baseline {baseline}C; {args.notes or ''}"]
        if args.ambient_c is not None:
            cmd += ["--ambient-c", str(args.ambient_c)]
        subprocess.run(cmd, check=True)


def cmd_handoff(args):
    from send_to_pi import send_case_to_pi
    tumor = [Path(r["image_path"]) for r in load_manifest(args.manifest) if r["true_class"] != "no_tumor"]
    out_csv = Path(args.out) / "handoff_timing.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slices", "repeat", "prepare_ms", "send_ms", "remote_case_ms",
                                          "receive_ms", "end_to_end_ms", "link"])
        w.writeheader()
        for n in [int(x) for x in args.sizes.split(",")]:
            for rep in range(1, args.repeats + 1):
                timings = {}
                t0 = time.perf_counter()
                send_case_to_pi(tumor[:n], args.pi_host, args.pi_user, args.remote_dir, Path(args.out) / "reports",
                                f"handoff_{n}_{rep}", ssh_opts=("-o", "BatchMode=yes"), timings=timings)
                row = {"slices": n, "repeat": rep, **{k: round(v, 1) for k, v in timings.items()},
                       "end_to_end_ms": round((time.perf_counter() - t0) * 1000, 1), "link": args.link}
                w.writerow(row)
                f.flush()
                print(row)
    print(f"Wrote {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--manifest", default="eval_sample_manifest.csv")
        p.add_argument("--out", default="results/pi-rerun-2026-10")
        p.add_argument("--num-runs", type=int, default=1000)
        p.add_argument("--threshold", type=float, default=0.70, help="gate tau (default 0.70, as deployed)")
        p.add_argument("--ambient-c", type=float, default=None)
        p.add_argument("--cooling", default=None, help='e.g. "bare", "heatsink", "heatsink+fan", "case model X"')
        p.add_argument("--notes", default=None)

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--mode", choices=list(MODES), required=True)
    r.add_argument("--rep", type=int, default=1)
    r.add_argument("--preload", choices=["all", "tier"], default="all")
    r.add_argument("--latency-threshold-ms", type=float, default=1350.0)
    r.add_argument("--temp-engage-c", type=float, default=65.0)
    r.add_argument("--temp-release-c", type=float, default=60.0)
    r.add_argument("--window", type=int, default=5)
    r.add_argument("--freq-cap-at", type=int, default=None)
    r.add_argument("--freq-cap-khz", type=int, default=1200000)
    r.add_argument("--freq-restore-at", type=int, default=None)
    r.add_argument("--abort-c", type=float, default=84.0)

    m = sub.add_parser("matrix")
    common(m)
    m.add_argument("--jobs", default="H,L,A,L,A,H", help="H=static-H, L=static-L, A=adaptive, in run order")
    m.add_argument("--baseline-c", type=float, default=None, help="skip the idle measurement and use this")
    m.add_argument("--idle-minutes", type=float, default=10)
    m.add_argument("--tolerance-c", type=float, default=1.0)
    m.add_argument("--max-wait-s", type=float, default=2400)

    h = sub.add_parser("handoff")
    h.add_argument("--manifest", default="eval_sample_manifest.csv")
    h.add_argument("--out", default="results/pi-rerun-2026-10/handoff")
    h.add_argument("--pi-host", required=True)
    h.add_argument("--pi-user", required=True)
    h.add_argument("--remote-dir", required=True)
    h.add_argument("--sizes", default="1,10,30")
    h.add_argument("--repeats", type=int, default=3)
    h.add_argument("--link", default="", help='record the link, e.g. "Wi-Fi 5 GHz" or "Ethernet"')

    args = ap.parse_args()
    {"run": cmd_run, "matrix": cmd_matrix, "handoff": cmd_handoff}[args.cmd](args)


if __name__ == "__main__":
    main()
