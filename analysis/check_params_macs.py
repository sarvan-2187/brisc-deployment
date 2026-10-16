"""Parameter count, FP32 size and MACs for the four deployed models (ISSUES.md H-01, H-10).

Builds the architectures exactly as app.py does (4-class heads, smp ResNet34
encoders). With --ckpt-dir it also loads the real checkpoints and counts the
tensors in each state_dict, which is the number Table 2 should report.

    pip install torch torchvision timm segmentation-models-pytorch thop
    python check_params_macs.py [--ckpt-dir ../../training_one]
"""

import argparse
from pathlib import Path

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from thop import profile
from torchvision.models import mobilenet_v3_large


def build(name):
    if name == "efficientnet_b0":
        return timm.create_model("efficientnet_b0", pretrained=False, num_classes=4), 224
    if name == "mobilenet_v3":
        m = mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 4)
        return m, 224
    cls = smp.Unet if name == "unet" else smp.DeepLabV3Plus
    return cls(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1), 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, help="folder with <name>_best.pt (real files, not LFS pointers)")
    args = ap.parse_args()

    print(f"{'model':16} {'params':>9} {'fp32 MiB':>9} {'fp32 MB':>8} {'GMACs':>6}  checkpoint")
    for name in ["efficientnet_b0", "mobilenet_v3", "deeplabv3plus", "unet"]:
        model, size = build(name)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        n_bytes = 4 * (n_params + sum(b.numel() for b in model.buffers()))
        macs, _ = profile(model, inputs=(torch.randn(1, 3, size, size),), verbose=False)
        ckpt_note = ""
        if args.ckpt_dir:
            path = args.ckpt_dir / f"{name}_best.pt"
            sd = torch.load(path, map_location="cpu")
            model.load_state_dict(sd)
            ckpt_note = f"loaded OK, file {path.stat().st_size / 2**20:.2f} MiB"
        print(f"{name:16} {n_params / 1e6:8.2f}M {n_bytes / 2**20:9.2f} {n_bytes / 1e6:8.2f} {macs / 1e9:6.2f}  {ckpt_note}")


if __name__ == "__main__":
    main()
