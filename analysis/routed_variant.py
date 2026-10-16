"""R1.2: per-class H-L IoU with Bonferroni CIs, and an offline class-routed variant
(predicted glioma -> DeepLabV3+, otherwise UNet) from the E2 per-slice IoUs.
Writes out/routed.json.  python routed_variant.py
"""
import json

import numpy as np
from scipy.stats import wilcoxon

from make_paper_numbers import E1_MS, OUT, TUMOR, load_e2

RNG = np.random.default_rng(0)
B = 10000


def boot(d, level=95.0):
    idx = RNG.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    a = (100 - level) / 2
    return [float(np.percentile(m, a)), float(np.percentile(m, 100 - a))]


def routed_iou(rows, router):
    return np.array([r["iou_H"] if r[f"pred_{router}"] == "glioma" else r["iou_L"] for r in rows])


def main():
    seg = [r for r in load_e2() if r["has_mask"]]
    H = np.array([r["iou_H"] for r in seg])
    L = np.array([r["iou_L"] for r in seg])
    out = {"n_seg": len(seg), "per_class": {}, "routed": {}}
    for c in TUMOR:
        m = np.array([r["true_class"] == c for r in seg])
        d = H[m] - L[m]
        out["per_class"][c] = {"n": int(m.sum()), "delta": float(d.mean()), "boot95": boot(d),
                               "boot_bonf": boot(d, 100 - 5 / 3), "wilcoxon_p": float(wilcoxon(H[m], L[m]).pvalue),
                               "wilcoxon_p_bonf": min(1.0, 3 * float(wilcoxon(H[m], L[m]).pvalue))}
    is_gl = np.array([r["true_class"] == "glioma" for r in seg])
    for router in ("L", "H"):
        R = routed_iou(seg, router)
        to_h = np.array([r[f"pred_{router}"] == "glioma" for r in seg])
        share = float(to_h.mean())
        cls_ms = E1_MS[router][0]
        out["routed"][router] = {
            "miou": float(R.mean()), "miou_H": float(H.mean()), "miou_L": float(L.mean()),
            "delta_vs_H": float((R - H).mean()), "boot95_vs_H": boot(R - H),
            "delta_vs_L": float((R - L).mean()), "boot95_vs_L": boot(R - L),
            "share_to_deeplab": share, "true_glioma_to_deeplab": int((to_h & is_gl).sum()),
            "non_glioma_to_deeplab": int((to_h & ~is_gl).sum()), "true_glioma": int(is_gl.sum()),
            # ponytail: segmented-path forward-pass estimate from E1 component means, not a measured routed run
            "est_seg_ms": cls_ms + share * E1_MS["H"][1] + (1 - share) * E1_MS["L"][1]}
    # split-half: pick the better of {H, L, routed} on half A, score it on half B
    wins = {"H": 0, "L": 0, "routed": 0}
    gain = []
    for _ in range(2000):
        a = RNG.random(len(seg)) < 0.5
        cand = {"H": H, "L": L, "routed": routed_iou(seg, "L")}
        best = max(cand, key=lambda k: cand[k][a].mean())
        wins[best] += 1
        gain.append(cand[best][~a].mean() - max(H[~a].mean(), L[~a].mean()))
    out["split_half"] = {"wins": wins, "mean_gain_vs_best_fixed_on_B": float(np.mean(gain))}
    (OUT / "routed.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    assert abs(out["routed"]["L"]["miou_H"] - H.mean()) < 1e-12


if __name__ == "__main__":
    main()
