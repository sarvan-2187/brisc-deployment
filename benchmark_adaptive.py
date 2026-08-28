"""Publication-grade comparison of AdaptiveController against static
baselines, for a paper's abstract and Results section.

By default runs two conditions back-to-back on the same cycling demo-image
test set:

  static:   EfficientNetB0 + DeepLabV3+ forced for every inference,
            no switching allowed.
  adaptive: AdaptiveController.select_pair() decides every inference,
            using its normal thresholds and hysteresis.

Pass --all-pairs to run all 4 fixed classifier+segmenter combinations as
separate static conditions (EfficientNetB0+DeepLabV3+, EfficientNetB0+UNet,
MobileNetV3+DeepLabV3+, MobileNetV3+UNet) alongside adaptive, for a complete
picture of where adaptive's mixed-tier usage sits relative to every fixed
pair, not just the two tiers it switches between.

In --duration mode, every condition after the first is forced to run the
exact same number of inferences the first one ended up doing (not just the
same wall-clock time), so all conditions process an identical sequence of
images in the same order - otherwise a faster combination fits more/
different image-cycle coverage into the same time window, which would make
the accuracy comparison apples-to-oranges. Each condition's summary also
reports a first-pass (single cycle, no repeats) accuracy/F1 alongside the
full-run figures, since any repeated image beyond the first cycle isn't an
independent sample.

Reuses benchmark.py's model loading/transforms/CLASS_NAMES/CLASSIFIER_CHOICES/
SEGMENTER_CHOICES, pi_thermal.py's thermal reads, case_report.py's
classify_slice/segment_slice (the same classify-then-threshold-gated-segment
logic the deployed app uses), and adaptive_controller.AdaptiveController - no
logic is reimplemented here.

Per-inference results go to CSV (one row per run, errors included but
excluded from stats), a structured summary goes to
benchmark_results/adaptive_vs_static_summary.json, a per-image breakdown
(one row per unique image, hit/miss counts for every condition side by side)
goes to benchmark_results/per_image_accuracy.csv, a wide per-metric
comparison goes to benchmark_results/summary_comparison.csv, and
latency/temperature-over-time charts (one line per condition) go to PNGs -
all under --out-dir (default benchmark_results/).

Test set defaults to demo_images_200/ (200 images, 50 per class, pulled from
the BRISC2025 test split, disjoint from demo_images/ and demo_images_50/).
Any run under ~10-15 minutes cycles through it only 1-2x, so a per-run
accuracy/F1 number reflects close to 200 independent samples instead of a
handful of images repeated many times over (which lets a single hard image
dominate the whole delta - see demo_images_50/'s known glioma/meningioma
mixup on brisc2025_test_00004_gl_ax_t1.jpg). Pass --demo-dir demo_images_50
or --demo-dir demo_images for the smaller/faster sets (e.g. quick sanity
checks where you don't need a statistically meaningful accuracy number).

Usage:
    python benchmark_adaptive.py --num-runs 10      # smoke test
    python benchmark_adaptive.py                    # full run, 5 min/condition
    python benchmark_adaptive.py --duration 120      # 2 min/condition
    python benchmark_adaptive.py --all-pairs         # all 4 fixed pairs + adaptive
    python benchmark_adaptive.py --demo-dir demo_images --num-runs 10
        # smallest/fastest 20-image set for a quick sanity check
    python benchmark_adaptive.py --num-runs 30 --latency-threshold-ms 800 --temp-threshold-c 50
        # demo run with lowered thresholds to force a visible tier switch
"""

import argparse
import csv
import itertools
import json
import statistics
import time
from pathlib import Path

from adaptive_controller import AdaptiveController, HIGH_ACCURACY, LIGHTWEIGHT
from benchmark import CLASS_NAMES, CLASSIFIER_CHOICES, SEGMENTER_CHOICES, load_cls_model, load_seg_model
from case_report import classify_slice, segment_slice
from pi_thermal import read_temp_c, read_throttled_flags

CSV_FIELDS = [
    "timestamp", "elapsed_s", "image", "true_class", "pair", "predicted_class",
    "correct", "latency_cls_ms", "latency_seg_ms", "latency_total_ms",
    "temp_c", "throttled", "reason", "error",
]
LIGHTWEIGHT_KEY = f"{LIGHTWEIGHT['classifier']}+{LIGHTWEIGHT['segmenter']}"
ALL_STATIC_PAIRS = {
    f"{c}+{s}": {"classifier": c, "segmenter": s} for c in CLASSIFIER_CHOICES for s in SEGMENTER_CHOICES
}


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


def run_condition(images, cls_models, seg_models, threshold, duration_s, num_runs, controller=None, forced_pair=None):
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

        pair = controller.select_pair() if controller is not None else (forced_pair or HIGH_ACCURACY)
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


def per_image_breakdown(rows):
    agg = {}
    for r in rows:
        if r["error"]:
            continue
        a = agg.setdefault(r["image"], {"true_class": r["true_class"], "tested": 0, "correct": 0})
        a["tested"] += 1
        a["correct"] += 1 if r["correct"] else 0
    return agg


def write_per_image_csv(path, condition_rows):
    """condition_rows: list of (label, rows) tuples, in the order conditions were run."""
    aggs = [(label, per_image_breakdown(rows)) for label, rows in condition_rows]
    all_images = sorted(set().union(*(set(agg) for _, agg in aggs))) if aggs else []
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["image", "true_class"]
        for label, _ in aggs:
            header += [f"{label}_tested", f"{label}_correct", f"{label}_accuracy"]
        writer.writerow(header)
        for img in all_images:
            true_class = None
            values = []
            for _, agg in aggs:
                a = agg.get(img, {"true_class": None, "tested": 0, "correct": 0})
                true_class = true_class or a["true_class"]
                acc = f"{a['correct'] / a['tested']:.4f}" if a["tested"] else ""
                values += [a["tested"], a["correct"], acc]
            writer.writerow([img, true_class] + values)


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


def summarize_condition(label, rows, errors, num_images, adaptive=False):
    valid = [r for r in rows if not r["error"]]
    latencies = [r["latency_total_ms"] for r in valid]
    temps = [r["temp_c"] for r in valid if r["temp_c"] is not None]
    throttled_seq = [r["throttled"] for r in valid]
    y_true = [r["true_class"] for r in valid]
    y_pred = [r["predicted_class"] for r in valid]

    # First full cycle through the image set, before any image repeats - a
    # clean single-pass accuracy/F1 alongside the full (possibly-repeating)
    # figures above, since a run longer than one cycle sees some images more
    # than once and those repeats aren't independent samples.
    first_pass = [r for r in rows[:num_images] if not r["error"]]
    first_pass_y_true = [r["true_class"] for r in first_pass]
    first_pass_y_pred = [r["predicted_class"] for r in first_pass]

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
        "first_pass_runs": len(first_pass),
        "first_pass_accuracy": sum(1 for r in first_pass if r["correct"]) / len(first_pass) if first_pass else 0.0,
        "first_pass_f1_weighted": weighted_f1(first_pass_y_true, first_pass_y_pred, CLASS_NAMES),
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


def print_summary_table(condition_stats):
    """condition_stats: list of (label, stats, is_adaptive) tuples, in run order."""
    labels = [label for label, _, _ in condition_stats]
    col_w = max(14, max(len(l) for l in labels) + 2)

    def row(name, cells):
        print(f"{name:38s}" + "".join(f"{c:>{col_w}}" for c in cells))

    print("\n=== SUMMARY ===\n")
    row("Metric", labels)
    print("Full-pipeline latency (ms):")
    for key, name in (("mean", "  mean"), ("median", "  median"), ("p95", "  p95"),
                       ("min", "  min"), ("max", "  max")):
        row(name, [f"{stats['latency_ms'][key]:.1f}" for _, stats, _ in condition_stats])
    row("Peak SoC temp", [fmt_temp(stats["peak_temp_c"]) for _, stats, _ in condition_stats])
    row("Throttling events", [str(stats["throttle_events"]) for _, stats, _ in condition_stats])
    row("Accuracy (all runs, may repeat)", [f"{stats['accuracy']:.4f}" for _, stats, _ in condition_stats])
    row("F1, all runs (may repeat)", [f"{stats['f1_weighted']:.4f}" for _, stats, _ in condition_stats])
    row("Accuracy (first pass, no repeats)",
        [f"{stats['first_pass_accuracy']:.4f}" for _, stats, _ in condition_stats])
    row("F1, first pass (no repeats)",
        [f"{stats['first_pass_f1_weighted']:.4f}" for _, stats, _ in condition_stats])
    row("Runs (valid/total/errors)",
        [f"{stats['valid_runs']}/{stats['runs']}/{stats['errors']}" for _, stats, _ in condition_stats])

    for label, stats, is_adaptive in condition_stats:
        if is_adaptive:
            print(f"\nAdaptive ({label}): {stats['pct_lightweight']:.1f}% of runs on lightweight tier, "
                  f"{stats['switch_events']} tier-switch event(s)")


def build_interpretation(condition_stats, delta_f1):
    """condition_stats: list of (label, stats, is_adaptive) tuples. The first entry is treated as the
    high-accuracy reference baseline; the last is expected to be the adaptive condition."""
    reference_label, reference_stats, _ = condition_stats[0]
    adaptive_label, adaptive_stats, _ = next(c for c in condition_stats if c[2])

    lat_a, lat_b = reference_stats["latency_ms"]["mean"], adaptive_stats["latency_ms"]["mean"]
    lat_diff_pct = 100 * (lat_a - lat_b) / lat_a if lat_a else 0.0
    peak_a, peak_b = reference_stats["peak_temp_c"], adaptive_stats["peak_temp_c"]

    if reference_stats["throttle_events"] == 0 and adaptive_stats["throttle_events"] == 0:
        throttle_sentence = (
            f"No throttling events were detected in either condition during this run "
            f"(peak temp {fmt_temp(peak_a)} {reference_label} vs {fmt_temp(peak_b)} adaptive), so this run "
            f"does not demonstrate a thermal-throttling avoidance benefit."
        )
    elif reference_stats["throttle_events"] > adaptive_stats["throttle_events"]:
        throttle_sentence = (
            f"The {reference_label} condition hit {reference_stats['throttle_events']} throttling event(s) "
            f"(peak {fmt_temp(peak_a)}) while adaptive hit {adaptive_stats['throttle_events']} "
            f"(peak {fmt_temp(peak_b)})."
        )
    elif adaptive_stats["throttle_events"] > reference_stats["throttle_events"]:
        throttle_sentence = (
            f"Adaptive hit more throttling events ({adaptive_stats['throttle_events']}, peak {fmt_temp(peak_b)}) "
            f"than the {reference_label} condition ({reference_stats['throttle_events']}, peak {fmt_temp(peak_a)})."
        )
    else:
        throttle_sentence = (
            f"Both conditions hit the same number of throttling events ({reference_stats['throttle_events']}; "
            f"peak {fmt_temp(peak_a)} {reference_label} vs {fmt_temp(peak_b)} adaptive)."
        )

    accuracy_note = "accuracy cost" if delta_f1 > 0 else ("no accuracy cost" if delta_f1 == 0 else "accuracy gain")

    text = (
        f"Over {reference_stats['runs']} {reference_label} and {adaptive_stats['runs']} adaptive runs, "
        f"mean full-pipeline latency was {lat_a:.1f}ms ({reference_label}) vs {lat_b:.1f}ms (adaptive), "
        f"a {lat_diff_pct:+.1f}% change. {throttle_sentence} "
        f"The adaptive controller ran the lightweight tier on {adaptive_stats['pct_lightweight']:.1f}% of its "
        f"runs across {adaptive_stats['switch_events']} tier-switch event(s). Classification F1 (weighted) was "
        f"{reference_stats['f1_weighted']:.4f} ({reference_label}) vs {adaptive_stats['f1_weighted']:.4f} "
        f"(adaptive, mixed tier usage), a delta-F1 of {delta_f1:+.4f} ({accuracy_note} from adaptive switching)."
    )

    others = [c for c in condition_stats if not c[2] and c[0] != reference_label]
    if others:
        text += " Other fixed pairs for reference: " + "; ".join(
            f"{label} - F1 {stats['f1_weighted']:.4f}, mean latency {stats['latency_ms']['mean']:.1f}ms"
            for label, stats, _ in others
        ) + "."
    return text


def write_summary_csv(path, condition_stats, delta_f1):
    """condition_stats: list of (label, stats, is_adaptive) tuples, in run order."""
    labels = [label for label, _, _ in condition_stats]
    rows = [["metric"] + labels]

    def row(name, values):
        rows.append([name] + list(values))

    for key, name in (("mean", "latency_mean_ms"), ("median", "latency_median_ms"), ("p95", "latency_p95_ms"),
                       ("min", "latency_min_ms"), ("max", "latency_max_ms")):
        row(name, [f"{stats['latency_ms'][key]:.2f}" for _, stats, _ in condition_stats])
    row("peak_temp_c", [format_cell(stats["peak_temp_c"]) for _, stats, _ in condition_stats])
    row("throttle_events", [stats["throttle_events"] for _, stats, _ in condition_stats])
    row("throttled_runs", [stats["throttled_runs"] for _, stats, _ in condition_stats])
    row("accuracy_all_runs", [f"{stats['accuracy']:.4f}" for _, stats, _ in condition_stats])
    row("f1_weighted_all_runs", [f"{stats['f1_weighted']:.4f}" for _, stats, _ in condition_stats])
    row("accuracy_first_pass_no_repeats", [f"{stats['first_pass_accuracy']:.4f}" for _, stats, _ in condition_stats])
    row("f1_weighted_first_pass_no_repeats",
        [f"{stats['first_pass_f1_weighted']:.4f}" for _, stats, _ in condition_stats])
    row("runs_valid_total_errors",
        [f"{stats['valid_runs']}/{stats['runs']}/{stats['errors']}" for _, stats, _ in condition_stats])
    row("pct_runs_lightweight_tier",
        [f"{stats['pct_lightweight']:.2f}" if is_a else "" for _, stats, is_a in condition_stats])
    row("tier_switch_events", [stats["switch_events"] if is_a else "" for _, stats, is_a in condition_stats])
    row("delta_f1_reference_minus_adaptive", [f"{delta_f1:.4f}"] + [""] * (len(labels) - 1))

    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def save_charts(condition_rows, out_dir):
    """condition_rows: list of (label, rows) tuples, in run order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(rows, key):
        pts = [(r["elapsed_s"], r[key]) for r in rows if not r["error"] and r[key] is not None]
        return list(zip(*pts)) if pts else ([], [])

    for key, ylabel, title, filename in (
        ("latency_total_ms", "Full-pipeline latency (ms)", "Latency over time", "latency_over_time.png"),
        ("temp_c", "SoC temperature (C)", "Temperature over time", "temperature_over_time.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        any_data = False
        for label, rows in condition_rows:
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
    parser.add_argument("--demo-dir", default="demo_images_200",
                         help="Folder of <class>/*.jpg subfolders to cycle through (default: demo_images_200, "
                              "200 images; pass demo_images_50 or demo_images for smaller/faster sets)")
    parser.add_argument("--out-dir", default="benchmark_results", help="Output directory for CSVs/JSON/PNGs")
    parser.add_argument("--all-pairs", action="store_true",
                         help="Run all 4 fixed classifier+segmenter combinations as separate static conditions "
                              "alongside adaptive (default: only the high-accuracy baseline vs adaptive)")
    parser.add_argument("--latency-threshold-ms", type=float, default=None,
                         help="Override AdaptiveController's latency trigger threshold (default: its own "
                              "default, 1350ms). Lower this for a demo run that forces a visible tier switch - "
                              "not meant for production comparisons.")
    parser.add_argument("--temp-threshold-c", type=float, default=None,
                         help="Override AdaptiveController's temp trigger threshold (default: its own "
                              "default, 65C). Lower this for a demo run that forces a visible tier switch - "
                              "not meant for production comparisons.")
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
    if args.all_pairs:
        print(f"Running all {len(ALL_STATIC_PAIRS)} fixed pairs plus adaptive")

    print("Loading all 4 checkpoints...")
    cls_models = {name: load_cls_model(name) for name in ("efficientnet_b0", "mobilenet_v3")}
    seg_models = {name: load_seg_model(name) for name in ("deeplabv3plus", "unet")}

    static_specs = list(ALL_STATIC_PAIRS.items()) if args.all_pairs else [("static", HIGH_ACCURACY)]

    condition_rows = []   # [(label, rows), ...]
    condition_stats = []  # [(label, stats, is_adaptive), ...]
    reference_num_runs = args.num_runs

    for idx, (label, pair) in enumerate(static_specs):
        print(f"\n=== Condition: {label} ===")
        rows, errors = run_condition(images, cls_models, seg_models, args.threshold,
                                      args.duration, reference_num_runs, controller=None, forced_pair=pair)
        if reference_num_runs is None:
            reference_num_runs = len(rows)
            print(f"  (every remaining condition will run exactly {reference_num_runs} inferences too, "
                  f"matching this one's count, so all conditions see the same image sequence)")
        csv_name = "condition_a_static.csv" if not args.all_pairs else f"condition_{label.replace('+', '_')}.csv"
        write_csv(out_dir / csv_name, rows)
        condition_rows.append((label, rows))
        condition_stats.append((label, summarize_condition(label, rows, errors, len(images)), False))

    print("\n=== Condition: adaptive ===")
    controller_kwargs = {"log_path": out_dir / "decision_log_adaptive.csv"}
    if args.latency_threshold_ms is not None:
        controller_kwargs["latency_threshold_ms"] = args.latency_threshold_ms
    if args.temp_threshold_c is not None:
        controller_kwargs["temp_threshold_c"] = args.temp_threshold_c
    if controller_kwargs.keys() - {"log_path"}:
        print(f"  (thresholds overridden for this run: "
              f"latency>{controller_kwargs.get('latency_threshold_ms', 'default')}ms, "
              f"temp>{controller_kwargs.get('temp_threshold_c', 'default')}C)")
    controller = AdaptiveController(**controller_kwargs)
    rows_adaptive, errors_adaptive = run_condition(images, cls_models, seg_models, args.threshold,
                                                    args.duration, reference_num_runs, controller=controller)
    csv_name = "condition_b_adaptive.csv" if not args.all_pairs else "condition_adaptive.csv"
    write_csv(out_dir / csv_name, rows_adaptive)
    condition_rows.append(("adaptive", rows_adaptive))
    condition_stats.append(
        ("adaptive", summarize_condition("adaptive", rows_adaptive, errors_adaptive, len(images), adaptive=True),
         True))

    reference_label, reference_stats, _ = condition_stats[0]
    _, adaptive_stats, _ = condition_stats[-1]
    delta_f1 = reference_stats["f1_weighted"] - adaptive_stats["f1_weighted"]
    interpretation = build_interpretation(condition_stats, delta_f1)

    print_summary_table(condition_stats)
    print("\n" + interpretation)

    summary = {
        "config": {
            "duration_s": args.duration if args.num_runs is None else None,
            "num_runs": args.num_runs,
            "threshold": args.threshold,
            "all_pairs": args.all_pairs,
            "controller_defaults": {
                "latency_threshold_ms": controller.latency_threshold_ms,
                "temp_threshold_c": controller.temp_threshold_c,
                "consecutive_trigger_count": controller.consecutive_trigger_count,
            },
        },
        "conditions": {label: stats for label, stats, _ in condition_stats},
        "reference_label": reference_label,
        "delta_f1_reference_minus_adaptive": delta_f1,
        "interpretation": interpretation,
    }
    with open(out_dir / "adaptive_vs_static_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_summary_csv(out_dir / "summary_comparison.csv", condition_stats, delta_f1)
    write_per_image_csv(out_dir / "per_image_accuracy.csv", condition_rows)

    save_charts(condition_rows, out_dir)

    print(f"\nWrote per-condition CSVs, {out_dir / 'summary_comparison.csv'}, "
          f"{out_dir / 'per_image_accuracy.csv'}, {out_dir / 'adaptive_vs_static_summary.json'}, "
          f"and PNG charts to {out_dir}/")


if __name__ == "__main__":
    main()
