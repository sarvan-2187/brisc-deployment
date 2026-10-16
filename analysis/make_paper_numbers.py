"""Every Phase 2 number and figure for the paper 2141 revision, from logs only.

    python make_paper_numbers.py                  # E2 stats, de-dup subset, tau sweep, ECE, Fig. 6
    python make_paper_numbers.py --pi-dir <dir>   # also: E3/E4 condition table from rerun_2141.py output

Inputs (read-only):
  brain-tumor-deploy/eval_results/per_image_results.csv   paired E2 evaluation (both tiers, 1,000 slices)
  results/near_duplicate_pairs.csv                         near-duplicate audit (H-02)
  brain-tumor-deploy/models, data/                         only to recompute the softmax for the tau sweep
Outputs: out/numbers.json, out/softmax.csv, out/tau_sweep.csv, out/pi_conditions.csv,
         final_paper/gr6.png (misclassified-case figure).
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.proportion import proportion_confint

from paired_stats import bootstrap_diff, mcnemar_exact, read_excluded

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEPLOY = ROOT / "brain-tumor-deploy"
OUT = HERE / "out"
FIG = ROOT / "final_paper" / "gr6.png"
TIERS = {"H": "high_accuracy", "L": "lightweight"}
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
TUMOR = CLASSES[:3]
# E1 single-image latencies (Table 2 of the paper, ms): classifier, segmenter per tier.
E1_MS = {"H": (274.6, 979.9), "L": (177.5, 909.5)}  # n = 100 micro-benchmark on the Pi, 16 Oct 2026 (out/micro_n100.csv)
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load_e2():
    with open(DEPLOY / "eval_results" / "per_image_results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1000:
        raise SystemExit(f"expected 1000 rows in per_image_results.csv, found {len(rows)}")
    for r in rows:
        r["has_mask"] = r["has_mask"] == "True"
        for t, col in TIERS.items():
            r[f"pred_{t}"] = r[f"{col}_predicted_class"]
            r[f"iou_{t}"] = float(r[f"{col}_iou"]) if r[f"{col}_iou"] else None
            r[f"dice_{t}"] = float(r[f"{col}_dice"]) if r[f"{col}_dice"] else None
    return rows


def paired(rows):
    ok = {t: np.array([r[f"pred_{t}"] == r["true_class"] for r in rows]) for t in TIERS}
    n = len(rows)
    b, c = int((ok["L"] & ~ok["H"]).sum()), int((ok["H"] & ~ok["L"]).sum())
    out = {"n": n, "mcnemar": {"b_H_wrong_L_right": b, "c_L_wrong_H_right": c, "p": mcnemar_exact(b, c)},
           "errors": {t: int((~ok[t]).sum()) for t in TIERS}, "shared_errors": int((~ok["H"] & ~ok["L"]).sum())}
    for t in TIERS:
        k = int(ok[t].sum())
        lo, hi = proportion_confint(k, n, method="wilson")
        out[f"accuracy_{t}"] = {"value": k / n, "wilson95": [lo, hi]}
    seg = [r for r in rows if r["has_mask"]]
    for groups, label in ((("all",), "all"), (TUMOR, "per_class")):
        for g in groups:
            sub = seg if g == "all" else [r for r in seg if r["true_class"] == g]
            for m in ("iou", "dice"):
                x = np.array([r[f"{m}_H"] for r in sub])
                y = np.array([r[f"{m}_L"] for r in sub])
                d, lo, hi = bootstrap_diff(x, y)
                out.setdefault(m, {})[g] = {
                    "n": len(sub), "H": x.mean(), "L": y.mean(), "delta_H_minus_L": d, "boot95": [lo, hi],
                    "wilcoxon_p": wilcoxon(x, y).pvalue, "H_better_on": int((x > y).sum()),
                    "L_better_on": int((x < y).sum())}
    return out


def softmax_table(rows):
    """Recompute 4-class softmax for both classifiers with the deployed code (CPU)."""
    path = OUT / "softmax.csv"
    if path.exists():
        with open(path, newline="") as f:
            return {r["filename"]: {k: float(v) for k, v in r.items() if k != "filename"} for r in csv.DictReader(f)}
    sys.path.insert(0, str(DEPLOY))
    cwd = os.getcwd()
    os.chdir(DEPLOY)  # benchmark.CKPT_DIR is relative
    try:
        from benchmark import load_cls_model
        from case_report import classify_slice
        models = {"H": load_cls_model("efficientnet_b0"), "L": load_cls_model("mobilenet_v3")}
        table = {}
        for i, r in enumerate(rows):
            img = DEPLOY / "data" / "classification_task" / "test" / r["true_class"] / r["filename"]
            table[r["filename"]] = {}
            for t, m in models.items():
                _, probs, _, _, _ = classify_slice(img, m, 0.70)
                for c, p in zip(CLASSES, probs):
                    table[r["filename"]][f"{t}_{c}"] = float(p)
            print(f"\r  softmax {i + 1}/{len(rows)}", end="", flush=True)
        print()
    finally:
        os.chdir(cwd)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename"] + [f"{t}_{c}" for t in TIERS for c in CLASSES])
        w.writeheader()
        for fn, d in table.items():
            w.writerow({"filename": fn, **d})
    return table


def top(sm, fn, t):
    p = [sm[fn][f"{t}_{c}"] for c in CLASSES]
    i = int(np.argmax(p))
    return CLASSES[i], p[i]


def tau_sweep(rows, sm):
    out = []
    for t in TIERS:
        cls_ms, seg_ms = E1_MS[t]
        for tau in TAUS:
            gated = {r["filename"]: (lambda c, p: c in TUMOR and p >= tau)(*top(sm, r["filename"], t)) for r in rows}
            tumor = [r for r in rows if r["has_mask"]]
            out.append({
                "tier": t, "tau": tau,
                "pct_segmented": 100 * sum(gated.values()) / len(rows),
                "tumor_gated_out": sum(1 for r in tumor if not gated[r["filename"]]),
                "no_tumor_segmented": sum(1 for r in rows if r["true_class"] == "no_tumor" and gated[r["filename"]]),
                "expected_latency_ms": cls_ms + seg_ms * sum(gated.values()) / len(rows),
                "gated_miou": statistics.mean(r[f"iou_{t}"] if gated[r["filename"]] else 0.0 for r in tumor),
            })
    return out


def ece(rows, sm, t, bins=15):
    conf = np.array([top(sm, r["filename"], t)[1] for r in rows])
    hit = np.array([top(sm, r["filename"], t)[0] == r["true_class"] for r in rows])
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return e


def misclassified_figure(rows, sm, dups):
    """Old-style tile grid (as qualitative_grid.py / Fig. 6 of the submission): each tile is the
    deployed app's annotate_result output for the tier that got the slice wrong, plus the true class."""
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(DEPLOY))
    cwd = os.getcwd()
    os.chdir(DEPLOY)  # model paths are relative
    try:
        from benchmark import load_cls_model, load_seg_model
        from case_report import classify_slice, segment_slice
        from report import FONTS_DIR, annotate_result
        cls_m = {"H": load_cls_model("efficientnet_b0"), "L": load_cls_model("mobilenet_v3")}
        seg_m = {"H": load_seg_model("deeplabv3plus"), "L": load_seg_model("unet")}
        cell, gap, cols = 396, 6, 5
        errs = [r for r in rows if r["pred_H"] != r["true_class"] or r["pred_L"] != r["true_class"]]
        tiles = []
        for r in errs:
            wrong = [t for t in TIERS if r[f"pred_{t}"] != r["true_class"]]
            t = wrong[0]
            img = DEPLOY / "data" / "classification_task" / "test" / r["true_class"] / r["filename"]
            image, _probs, top_class, top_conf, verdict = classify_slice(img, cls_m[t], 0.70)
            mask = segment_slice(image, seg_m[t]) if verdict == "tumor" else None
            same = len(wrong) == 2 and r["pred_H"] == r["pred_L"]
            tag = "D, F" if same else {"H": "D", "L": "F"}[t]
            tile = annotate_result(image.resize((cell, cell)), mask,
                                   f"{top_class.replace('_', ' ').title()} - {top_conf:.1%} ({tag})")
            # True class, top-left, in the same style as annotate_result's bottom label.
            draw = ImageDraw.Draw(tile, "RGBA")
            font_size = max(14, tile.width // 18)
            font = ImageFont.truetype(str(FONTS_DIR / "Geist-Bold.ttf"), font_size)
            pad = font_size // 2
            text = f"True: {r['true_class'].replace('_', ' ').title()}" + ("*" if r["filename"] in dups else "")
            _, _, tw, th = draw.textbbox((0, 0), text, font=font)
            draw.rectangle((0, 0, tw + 2 * pad, th + 2 * pad), fill=(0, 0, 0, 160))
            draw.text((pad, pad), text, font=font, fill=(255, 255, 255, 255))
            tiles.append(tile.resize((cell, cell)))
    finally:
        os.chdir(cwd)

    nrows = -(-len(tiles) // cols)
    grid = Image.new("RGB", (cols * cell + (cols - 1) * gap, nrows * cell + (nrows - 1) * gap), "white")
    for i, tile in enumerate(tiles):
        rr, cc = divmod(i, cols)
        grid.paste(tile, (cc * (cell + gap), rr * (cell + gap)))
    grid.save(FIG, dpi=(300, 300))
    return [{"filename": r["filename"], "true": r["true_class"], "pred_H": r["pred_H"], "pred_L": r["pred_L"],
             "conf_H": top(sm, r["filename"], "H")[1], "conf_L": top(sm, r["filename"], "L")[1],
             "dup": r["filename"] in dups} for r in errs]


def pi_conditions(pi_dir):
    """E3/E4: one row per run from rerun_2141.py summary.json files, plus mean +/- SD per mode."""
    runs = []
    for s in sorted(Path(pi_dir).glob("*/summary.json")):
        d = json.loads(s.read_text())
        mode = s.parent.name.split("_rep")[0]
        lat = d["latency_ms"]
        runs.append({
            "run": s.parent.name, "mode": mode, "n": d["n"], "accuracy": d["accuracy"],
            "workload_mean_ms": lat["workload"]["mean"], "workload_median_ms": lat["workload"]["median"],
            "workload_p95_ms": lat["workload"]["p95"],
            "segmented_mean_ms": (lat["segmented_path"] or {}).get("mean"),
            "fast_mean_ms": (lat["fast_path"] or {}).get("mean"), "fast_path_count": d["fast_path_count"],
            "temp_start_c": d["temp_c"]["start"], "temp_mean_c": d["temp_c"]["mean"], "temp_peak_c": d["temp_c"]["peak"],
            "throttled_end": d["throttled_hex"]["end"], "min_clock_hz": d["min_arm_clock_hz"],
            "switches": d["tier_switches"], "pct_lightweight": d["pct_lightweight"],
            "peak_rss_mb": d["peak_rss_kb_vmhwm"] / 1024 if d["peak_rss_kb_vmhwm"] else None,
            "min_mem_available_mb": d["min_mem_available_kb"] / 1024 if d["min_mem_available_kb"] else None,
            "peak_swap_mb": d["peak_swap_used_kb"] / 1024 if d["peak_swap_used_kb"] is not None else None,
        })
    if not runs:
        raise SystemExit(f"no */summary.json under {pi_dir}")
    with open(OUT / "pi_conditions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    agg = {}
    for mode in sorted({r["mode"] for r in runs}):
        sub = [r for r in runs if r["mode"] == mode]
        agg[mode] = {"runs": len(sub)}
        for k, v in sub[0].items():
            vals = [r[k] for r in sub if isinstance(r[k], (int, float)) and not isinstance(r[k], bool)]
            if vals and k not in ("n",):
                agg[mode][k] = {"mean": statistics.mean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pi-dir", default=None, help="rerun_2141.py --out directory (Phase 3)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    rows = load_e2()
    dups = read_excluded(HERE / "results" / "near_duplicate_pairs.csv")
    numbers = {"e2_full": paired(rows)}
    dedup = [r for r in rows if r["filename"] not in dups]
    numbers["e2_dedup"] = paired(dedup)
    numbers["e2_dedup"]["excluded"] = len(rows) - len(dedup)

    sm = softmax_table(rows)
    mism = {t: sum(top(sm, r["filename"], t)[0] != r[f"pred_{t}"] for r in rows) for t in TIERS}
    if any(mism.values()):
        raise SystemExit(f"laptop softmax disagrees with the E2 predictions: {mism}")
    numbers["softmax_matches_e2_predictions"] = True
    sweep = tau_sweep(rows, sm)
    with open(OUT / "tau_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)
    numbers["tau_sweep"] = sweep
    numbers["ece_15bin"] = {t: ece(rows, sm, t) for t in TIERS}
    numbers["misclassified"] = misclassified_figure(rows, sm, dups)
    if args.pi_dir:
        numbers["pi_conditions"] = pi_conditions(args.pi_dir)

    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
    print(json.dumps({k: numbers[k] for k in ("e2_full", "e2_dedup", "ece_15bin")}, indent=1, default=float))
    print(f"\ntau sweep -> {OUT / 'tau_sweep.csv'}; misclassified figure -> {FIG}")


if __name__ == "__main__":
    main()
