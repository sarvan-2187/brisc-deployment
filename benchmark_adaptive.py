"""Publication-grade comparison of AdaptiveController against a static
high-accuracy baseline, for a paper's abstract and Results section.

Runs two conditions back-to-back on the same cycling demo-image test set for
a fair comparison:

  A - static:   EfficientNetB0 + DeepLabV3+ forced for every inference,
                no switching allowed.
  B - adaptive: AdaptiveController.select_pair() decides every inference,
                using its normal thresholds and hysteresis.

Reuses benchmark.py's model loading/transforms/CLASS_NAMES, pi_thermal.py's
thermal reads, case_report.py's classify_slice/segment_slice (the same
classify-then-threshold-gated-segment logic the deployed app uses), and
adaptive_controller.AdaptiveController - no logic is reimplemented here.

Per-inference results go to CSV (one row per run, errors included but
excluded from stats), a structured summary goes to
benchmark_results/adaptive_vs_static_summary.json, and latency/temperature
-over-time charts go to PNGs - all under --out-dir (default benchmark_results/).

Usage:
    python benchmark_adaptive.py --num-runs 10      # smoke test
    python benchmark_adaptive.py                    # full run, 5 min/condition
    python benchmark_adaptive.py --duration 120      # 2 min/condition
"""

import argparse
import csv
import itertools
import json
import statistics
import time
from pathlib import Path

from adaptive_controller import AdaptiveController, HIGH_ACCURACY, LIGHTWEIGHT
from benchmark import CLASS_NAMES, load_cls_model, load_seg_model
from case_report import classify_slice, segment_slice
from pi_thermal import read_temp_c, read_throttled_flags

CSV_FIELDS = [
    "timestamp", "elapsed_s", "image", "true_class", "pair", "predicted_class",
    "correct", "latency_cls_ms", "latency_seg_ms", "latency_total_ms",
    "temp_c", "throttled", "reason", "error",
]
LIGHTWEIGHT_KEY = f"{LIGHTWEIGHT['classifier']}+{LIGHTWEIGHT['segmenter']}"


def gather_demo_images(demo_dir):
    images = []
    for cls_name in CLASS_NAMES:
        images.extend((cls_name, p) for p in sorted((demo_dir / cls_name).glob("*.jpg")))
    return images


def run_inference(path, cls_models, seg_models, pair, threshold):
    t0 = time.perf_counter()
    image, _probs, top_class, top_conf, verdict_kind = classify_slice(
        path, cls_models[pair["classifier"]], threshold)
    cls_latency_ms = (time.perf_counter() - t0) * 1000

    seg_latency_ms = 0.0
    if verdict_kind == "tumor":
        t1 = time.perf_counter()
        segment_slice(image, seg_models[pair["segmenter"]])
        seg_latency_ms = (time.perf_counter() - t1) * 1000

    return top_class, cls_latency_ms, seg_latency_ms, cls_latency_ms + seg_latency_ms


def run_condition(images, cls_models, seg_models, threshold, duration_s, num_runs, controller):
    rows = []
    errors = 0
    image_cycle = itertools.cycle(images)
    start = time.monotonic()
    i = 0
    while True:
        if num_runs is not None:
            if i >= num_runs:
                break
        elif time.monotonic() - start >= duration_s:
            break
        i += 1
        true_class, path = next(image_cycle)

        pair = controller.select_pair() if controller is not None else HIGH_ACCURACY
        reason = controller.last_reason if controller is not None else ""

        row = dict.fromkeys(CSV_FIELDS, None)
        row.update({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": time.monotonic() - start,
            "image": path.name,
            "true_class": true_class,
            "pair": f"{pair['classifier']}+{pair['segmenter']}",
            "reason": reason,
            "error": "",
        })
        try:
            predicted_class, cls_ms, seg_ms, total_ms = run_inference(
                path, cls_models, seg_models, pair, threshold)
            temp = read_temp_c()
            throttled = bool(read_throttled_flags())
            row.update({
                "predicted_class": predicted_class,
                "correct": predicted_class == true_class,
                "latency_cls_ms": cls_ms,
                "latency_seg_ms": seg_ms,
                "latency_total_ms": total_ms,
                "temp_c": temp,
                "throttled": throttled,
            })
            if controller is not None:
                controller.record_inference(total_ms)
        except Exception as e:  # noqa: BLE001 - a benchmark run failure must be recorded, not swallowed
            errors += 1
            row["error"] = str(e)

        rows.append(row)

    return rows, errors


def format_cell(v):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return v


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        for r in rows:
            writer.writerow([format_cell(r[k]) for k in CSV_FIELDS])


def latency_stats(values):
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    v = sorted(values)
    return {
        "mean": statistics.mean(v),
        "median": statistics.median(v),
        "p95": v[min(len(v) - 1, int(len(v) * 0.95))],
        "min": v[0],
        "max": v[-1],
    }


def count_transitions(bool_seq):
    events, prev = 0, False
    for b in bool_seq:
        if b and not prev:
            events += 1
        prev = b
    return events


def weighted_f1(y_true, y_pred, classes):
    if not y_true:
        return 0.0
    weighted_sum = 0.0
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = sum(1 for t in y_true if t == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        weighted_sum += f1 * support
    return weighted_sum / len(y_true)


def summarize_condition(label, rows, errors, adaptive=False):
    valid = [r for r in rows if not r["error"]]
    latencies = [r["latency_total_ms"] for r in valid]
    temps = [r["temp_c"] for r in valid if r["temp_c"] is not None]
    throttled_seq = [r["throttled"] for r in valid]
    y_true = [r["true_class"] for r in valid]
    y_pred = [r["predicted_class"] for r in valid]

    stats = {
        "label": label,
        "runs": len(rows),
        "valid_runs": len(valid),
        "errors": errors,
        "latency_ms": latency_stats(latencies),
        "peak_temp_c": max(temps) if temps else None,
        "throttle_events": count_transitions(throttled_seq),
        "throttled_runs": sum(throttled_seq),
        "accuracy": sum(1 for r in valid if r["correct"]) / len(valid) if valid else 0.0,
        "f1_weighted": weighted_f1(y_true, y_pred, CLASS_NAMES),
    }
    if adaptive:
        pair_seq = [r["pair"] for r in valid]
        stats["pct_lightweight"] = (
            100 * sum(1 for p in pair_seq if p == LIGHTWEIGHT_KEY) / len(pair_seq) if pair_seq else 0.0
        )
        stats["switch_events"] = sum(1 for a, b in zip(pair_seq, pair_seq[1:]) if a != b)
    return stats


def fmt_temp(t):
    return f"{t:.1f}C" if t is not None else "unavailable"


def print_summary_table(stats_a, stats_b, delta_f1):
    print("\n=== SUMMARY: static vs adaptive ===\n")
    print(f"{'Metric':38s}{'Static':>14s}{'Adaptive':>14s}")
    print("Full-pipeline latency (ms):")
    for key, label in (("mean", "  mean"), ("median", "  median"), ("p95", "  p95"),
                        ("min", "  min"), ("max", "  max")):
        print(f"{label:38s}{stats_a['latency_ms'][key]:14.1f}{stats_b['latency_ms'][key]:14.1f}")
    print(f"{'Peak SoC temp':38s}{fmt_temp(stats_a['peak_temp_c']):>14s}{fmt_temp(stats_b['peak_temp_c']):>14s}")
    print(f"{'Throttling events':38s}{stats_a['throttle_events']:14d}{stats_b['throttle_events']:14d}")
    print(f"{'Accuracy':38s}{stats_a['accuracy']:14.4f}{stats_b['accuracy']:14.4f}")
    print(f"{'F1 (weighted)':38s}{stats_a['f1_weighted']:14.4f}{stats_b['f1_weighted']:14.4f}")
    a_runs = f"{stats_a['valid_runs']}/{stats_a['runs']}/{stats_a['errors']}"
    b_runs = f"{stats_b['valid_runs']}/{stats_b['runs']}/{stats_b['errors']}"
    print(f"{'Runs (valid/total/errors)':38s}{a_runs:>14s}{b_runs:>14s}")
    print(f"\nDelta-F1 (static - adaptive): {delta_f1:+.4f}")
    print(f"Adaptive: {stats_b['pct_lightweight']:.1f}% of runs on lightweight tier, "
          f"{stats_b['switch_events']} tier-switch event(s)")


def build_interpretation(stats_a, stats_b, delta_f1):
    lat_a, lat_b = stats_a["latency_ms"]["mean"], stats_b["latency_ms"]["mean"]
    lat_diff_pct = 100 * (lat_a - lat_b) / lat_a if lat_a else 0.0
    peak_a, peak_b = stats_a["peak_temp_c"], stats_b["peak_temp_c"]

    if stats_a["throttle_events"] == 0 and stats_b["throttle_events"] == 0:
        throttle_sentence = (
            f"No throttling events were detected in either condition during this run "
            f"(peak temp {fmt_temp(peak_a)} static vs {fmt_temp(peak_b)} adaptive), so this run does not "
            f"demonstrate a thermal-throttling avoidance benefit."
        )
    elif stats_a["throttle_events"] > stats_b["throttle_events"]:
        throttle_sentence = (
            f"The static condition hit {stats_a['throttle_events']} throttling event(s) (peak {fmt_temp(peak_a)}) "
            f"while the adaptive condition hit {stats_b['throttle_events']} (peak {fmt_temp(peak_b)})."
        )
    elif stats_b["throttle_events"] > stats_a["throttle_events"]:
        throttle_sentence = (
            f"The adaptive condition hit more throttling events ({stats_b['throttle_events']}, "
            f"peak {fmt_temp(peak_b)}) than the static condition ({stats_a['throttle_events']}, "
            f"peak {fmt_temp(peak_a)})."
        )
    else:
        throttle_sentence = (
            f"Both conditions hit the same number of throttling events ({stats_a['throttle_events']}; "
            f"peak {fmt_temp(peak_a)} static vs {fmt_temp(peak_b)} adaptive)."
        )

    accuracy_note = "accuracy cost" if delta_f1 > 0 else ("no accuracy cost" if delta_f1 == 0 else "accuracy gain")

    return (
        f"Over {stats_a['runs']} static and {stats_b['runs']} adaptive runs, mean full-pipeline latency was "
        f"{lat_a:.1f}ms (static) vs {lat_b:.1f}ms (adaptive), a {lat_diff_pct:+.1f}% change. {throttle_sentence} "
        f"The adaptive controller ran the lightweight tier on {stats_b['pct_lightweight']:.1f}% of its runs "
        f"across {stats_b['switch_events']} tier-switch event(s). Classification F1 (weighted) was "
        f"{stats_a['f1_weighted']:.4f} (static, always high-accuracy tier) vs {stats_b['f1_weighted']:.4f} "
        f"(adaptive, mixed tier usage), a delta-F1 of {delta_f1:+.4f} ({accuracy_note} from adaptive switching)."
    )


def save_charts(rows_a, rows_b, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(rows, key):
        xs = [r["elapsed_s"] for r in rows if not r["error"] and r[key] is not None]
        ys = [r[key] for r in rows if not r["error"] and r[key] is not None]
        return xs, ys

    for key, ylabel, title, filename in (
        ("latency_total_ms", "Full-pipeline latency (ms)", "Latency over time: static vs adaptive",
         "latency_over_time.png"),
        ("temp_c", "SoC temperature (C)", "Temperature over time: static vs adaptive",
         "temperature_over_time.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        any_data = False
        for rows, label in ((rows_a, "Static"), (rows_b, "Adaptive")):
            xs, ys = series(rows, key)
            if xs:
                any_data = True
                ax.plot(xs, ys, marker="o", markersize=2, linewidth=1, label=label)
        if not any_data:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if any_data:
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=300.0,
                         help="Seconds of continuous back-to-back inference per condition (default 300 = 5 min)")
    parser.add_argument("--num-runs", type=int, default=None,
                         help="Fixed number of inferences per condition; overrides --duration if given")
    parser.add_argument("--threshold", type=float, default=0.70,
                         help="Confidence threshold for running segmentation (default 0.70, matches app.py)")
    parser.add_argument("--demo-dir", default="demo_images", help="Folder of demo_images/<class>/*.jpg")
    parser.add_argument("--out-dir", default="benchmark_results", help="Output directory for CSVs/JSON/PNGs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = gather_demo_images(Path(args.demo_dir))
    if not images:
        raise SystemExit(f"No demo images found under {args.demo_dir}/<class>/*.jpg")

    print(f"Test set: {len(images)} images across {len(CLASS_NAMES)} classes from {args.demo_dir}")
    if args.num_runs is not None:
        print(f"Runs per condition: {args.num_runs} (--num-runs overrides --duration)")
    else:
        print(f"Duration per condition: {args.duration:.0f}s")

    print("Loading all 4 checkpoints...")
    cls_models = {name: load_cls_model(name) for name in ("efficientnet_b0", "mobilenet_v3")}
    seg_models = {name: load_seg_model(name) for name in ("deeplabv3plus", "unet")}

    print("\n=== Condition A: static (EfficientNetB0 + DeepLabV3+ forced) ===")
    rows_a, errors_a = run_condition(images, cls_models, seg_models, args.threshold,
                                      args.duration, args.num_runs, controller=None)
    write_csv(out_dir / "condition_a_static.csv", rows_a)

    print("=== Condition B: adaptive (AdaptiveController) ===")
    controller = AdaptiveController(log_path=out_dir / "decision_log_condition_b.csv")
    rows_b, errors_b = run_condition(images, cls_models, seg_models, args.threshold,
                                      args.duration, args.num_runs, controller=controller)
    write_csv(out_dir / "condition_b_adaptive.csv", rows_b)

    stats_a = summarize_condition("Static (high-accuracy forced)", rows_a, errors_a)
    stats_b = summarize_condition("Adaptive (AdaptiveController)", rows_b, errors_b, adaptive=True)
    delta_f1 = stats_a["f1_weighted"] - stats_b["f1_weighted"]
    interpretation = build_interpretation(stats_a, stats_b, delta_f1)

    print_summary_table(stats_a, stats_b, delta_f1)
    print("\n" + interpretation)

    summary = {
        "config": {
            "duration_s": args.duration if args.num_runs is None else None,
            "num_runs": args.num_runs,
            "threshold": args.threshold,
            "controller_defaults": {
                "latency_threshold_ms": controller.latency_threshold_ms,
                "temp_threshold_c": controller.temp_threshold_c,
                "consecutive_trigger_count": controller.consecutive_trigger_count,
            },
        },
        "condition_a": stats_a,
        "condition_b": stats_b,
        "delta_f1": delta_f1,
        "interpretation": interpretation,
    }
    with open(out_dir / "adaptive_vs_static_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    save_charts(rows_a, rows_b, out_dir)

    print(f"\nWrote {out_dir / 'condition_a_static.csv'}, {out_dir / 'condition_b_adaptive.csv'}, "
          f"{out_dir / 'adaptive_vs_static_summary.json'}, and PNG charts to {out_dir}/")


if __name__ == "__main__":
    main()
