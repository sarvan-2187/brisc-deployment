"""Paired significance tests for Tier-H vs Tier-L (ISSUES.md H-05, H-14, H-02).

Two ways to run it:

  1. Provisional, from the error counts quoted in the manuscript (no logs needed):
         python paired_stats.py --b 7 --c 6 --n 1000 --errors-h 9 --errors-l 8

  2. From per-image logs (one CSV per tier, columns from PLAN-PROMPT.md B3:
     filename, true_label, pred, iou, dice; iou/dice empty for no-tumor):
         python paired_stats.py --log-h H.csv --log-l L.csv \
             [--exclude results/near_duplicate_pairs.csv] [--export-errors errors.csv]

     --exclude drops test files whose mean_abs_pixel_diff_256px < 5, giving the
     de-duplicated subset.
"""

import argparse
import csv

import numpy as np
from scipy.stats import binomtest, wilcoxon
from statsmodels.stats.proportion import proportion_confint


def mcnemar_exact(b, c):
    return 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{k / n:.4f} [{lo:.4f}, {hi:.4f}]"


def bootstrap_diff(x, y, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    d = x[idx].mean(1) - y[idx].mean(1)
    return x.mean() - y.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def read_log(path):
    with open(path, newline="") as fh:
        return {r["filename"]: r for r in csv.DictReader(fh)}


def read_excluded(path, mad=5.0):
    with open(path, newline="") as fh:
        return {r["test_file"] for r in csv.DictReader(fh) if float(r["mean_abs_pixel_diff_256px"]) < mad}


def from_logs(args):
    h, l = read_log(args.log_h), read_log(args.log_l)
    if set(h) != set(l):
        raise SystemExit(f"logs cover different images: {len(set(h) ^ set(l))} differ")
    names = sorted(h)
    if args.exclude:
        drop = read_excluded(args.exclude)
        names = [n for n in names if n not in drop]
        print(f"de-duplicated subset: {len(names)} images ({len(h) - len(names)} excluded)")

    ok_h = np.array([h[n]["pred"] == h[n]["true_label"] for n in names])
    ok_l = np.array([l[n]["pred"] == l[n]["true_label"] for n in names])
    b, c = int((ok_l & ~ok_h).sum()), int((ok_h & ~ok_l).sum())
    n = len(names)
    print(f"accuracy  Tier-H {wilson(int(ok_h.sum()), n)}   Tier-L {wilson(int(ok_l.sum()), n)}")
    print(f"McNemar exact: b (H wrong, L right) = {b}, c (L wrong, H right) = {c}, p = {mcnemar_exact(b, c):.4f}")
    print(f"errors shared by both tiers: {int((~ok_h & ~ok_l).sum())}")

    seg = [nm for nm in names if h[nm]["iou"] not in ("", None)]
    for metric in ("iou", "dice"):
        x = np.array([float(h[nm][metric]) for nm in seg])
        y = np.array([float(l[nm][metric]) for nm in seg])
        d, lo, hi = bootstrap_diff(x, y)
        p = wilcoxon(x, y).pvalue
        print(f"{metric}: H {x.mean():.4f}  L {y.mean():.4f}  Δ(H−L) {d:+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"Wilcoxon p = {p:.4f}  (n = {len(seg)})")

    if args.export_errors:
        with open(args.export_errors, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "true_label", "pred_H", "pred_L", "wrong_in"])
            for nm, eh, el in zip(names, ~ok_h, ~ok_l):
                if eh or el:
                    w.writerow([nm, h[nm]["true_label"], h[nm]["pred"], l[nm]["pred"],
                                "both" if eh and el else ("H" if eh else "L")])
        print(f"wrote misclassified list to {args.export_errors}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int)
    ap.add_argument("--c", type=int)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--errors-h", type=int)
    ap.add_argument("--errors-l", type=int)
    ap.add_argument("--log-h")
    ap.add_argument("--log-l")
    ap.add_argument("--exclude")
    ap.add_argument("--export-errors")
    args = ap.parse_args()

    if args.log_h and args.log_l:
        from_logs(args)
    elif args.b is not None and args.c is not None:
        print(f"McNemar exact (provisional): b = {args.b}, c = {args.c}, p = {mcnemar_exact(args.b, args.c):.4f}")
        if args.errors_h is not None:
            print(f"accuracy Tier-H {wilson(args.n - args.errors_h, args.n)}")
        if args.errors_l is not None:
            print(f"accuracy Tier-L {wilson(args.n - args.errors_l, args.n)}")
    else:
        ap.error("give --b and --c, or --log-h and --log-l")


if __name__ == "__main__":
    main()
