"""Middlebury evaluation for PyTorch PSCVStereo.

Reports RAFT-Stereo EPE/D1.

Inference resizes each pair to the training crop (tiny 416x576, base 512x896),
runs the model, then bilinear-upsamples disparity back to native resolution
(with horizontal disparity scaled by orig_W / infer_W).

Example:
  python evaluate_pscvstereo.py --restore_ckpt checkpoints/pscvstereo_tiny.pth \\
      --model tiny --dataset middlebury_Q
"""
from __future__ import print_function, division

import argparse
import importlib
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

import core.stereo_datasets as datasets
from core.utils.utils import InputPadder

try:
    from torch.cuda.amp import autocast
except ImportError:
    class autocast:
        def __init__(self, enabled=True):
            self.enabled = enabled

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False


# ---------------------------------------------------------------------------
# Model / eval
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "base": {
        "module": ("core.pscvstereo_base", "PSCVStereoBase"),
        "default_max_disp": 208,
        "default_iters": [5, 5],
        "default_image_size": [512, 896],  # H W (training crop)
    },
    "tiny": {
        "module": ("core.pscvstereo_tiny", "PSCVStereoTiny"),
        "default_max_disp": 176,
        "default_iters": [3, 5],
        "default_image_size": [416, 576],  # H W (training crop)
    },
}

MIDDLEBURY_SPLITS = ("F", "H", "Q", "2014", "2021")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(args):
    cfg = MODEL_REGISTRY[args.model]
    mod_name, cls_name = cfg["module"]
    module = importlib.import_module(mod_name)
    return getattr(module, cls_name)(
        max_disp=args.max_disp, iters=args.iters, test_mode=True
    )


def strip_module_prefix(state_dict):
    """Strip leading 'module.' from DataParallel checkpoints."""
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return type(state_dict)(
        (k[len("module."):] if k.startswith("module.") else k, v)
        for k, v in state_dict.items()
    )


def load_raw_checkpoint(path):
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return ckpt


def prepare_middlebury_dataset(split):
    """Build Middlebury loader with no aug / no stochastic downsampling."""
    val_dataset = datasets.Middlebury(None, split=split)
    val_dataset.downsamp_prob = 0.0
    # Identity: keep full-resolution evaluation (RAFT-Stereo style).
    val_dataset._downsample_for_large_disparity = (
        lambda img1, img2, flow, valid: (img1, img2, flow, valid)
    )
    return val_dataset


@torch.no_grad()
def validate_middlebury(
    model,
    split="F",
    mixed_prec=False,
    device=0,
    image_size=None,
):
    """Validate on Middlebury; infer at ``image_size`` then rescale to GT res.

    Pipeline matches deployment-style eval:
      1) resize stereo pair to training resolution (H, W)
      2) pad to multiple of 32 and run the model
      3) unpad, bilinear-upsample disparity to original HxW
      4) scale disparity values by (orig_W / infer_W)

    Reports RAFT-Stereo EPE / D1 (>2px).
    """
    model.eval()
    val_dataset = prepare_middlebury_dataset(split)
    assert image_size is not None and len(image_size) == 2
    infer_h, infer_w = int(image_size[0]), int(image_size[1])
    logging.info(
        "Inference resolution H=%d W=%d (disparity scaled back to native size)",
        infer_h,
        infer_w,
    )

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = (
            val_dataset[val_id]
        )
        image1 = image1[None].cuda(device)
        image2 = image2[None].cuda(device)
        orig_h, orig_w = image1.shape[-2:]

        image1 = F.interpolate(
            image1, size=(infer_h, infer_w), mode="bilinear", align_corners=False
        )
        image2 = F.interpolate(
            image2, size=(infer_h, infer_w), mode="bilinear", align_corners=False
        )

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            predictions, _ = model(image1, image2)
        flow_pr = padder.unpad(predictions[-1].float())

        # Upsample flow map to native resolution and scale horizontal disparity.
        scale_w = float(orig_w) / float(infer_w)
        flow_pr = F.interpolate(
            flow_pr, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        flow_pr = (flow_pr * scale_w).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)

        out = epe_flattened > 2.0
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(
            "Middlebury Iter %d out of %d. EPE %.4f D1 %.4f (%s)",
            val_id + 1,
            len(val_dataset),
            image_epe,
            image_out,
            imageL_file,
        )
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = float(np.mean(epe_list))
    d1 = float(100 * np.mean(out_list))

    print("Validation Middlebury%s: EPE %f, D1 %f" % (split, epe, d1))
    results = {"middlebury%s-epe" % split: epe, "middlebury%s-d1" % split: d1}

    return results


def load_pytorch_model(args, state_dict, device):
    """Load float weights into the PyTorch model."""
    model = build_model(args)
    state = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.warning("Missing keys (first 10): %s", missing[:10])
    if unexpected:
        logging.warning("Unexpected keys (first 10): %s", unexpected[:10])
    return model.cuda(device)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PyTorch PSCVStereo on Middlebury (RAFT-Stereo EPE/D1)"
        )
    )
    parser.add_argument(
        "--restore_ckpt",
        required=True,
        help="path to float .pth checkpoint",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="PSCVStereo variant",
    )
    parser.add_argument(
        "--dataset",
        default="middlebury_H",
        choices=["middlebury_%s" % s for s in MIDDLEBURY_SPLITS],
        help="Middlebury split to evaluate",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs="+",
        default=None,
        help="inference resolution as H W (default: training crop; "
             "tiny 416 576, base 512 896)",
    )
    parser.add_argument(
        "--max_disp",
        type=int,
        default=None,
        help="max disparity (default depends on --model)",
    )
    parser.add_argument(
        "--iters",
        nargs="+",
        type=int,
        default=None,
        help="update iters per pyramid level",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="CUDA device id(s); only the first is used for eval",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="enable autocast during forward",
    )
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    if args.max_disp is None:
        args.max_disp = cfg["default_max_disp"]
    if args.iters is None:
        args.iters = list(cfg["default_iters"])
    if args.image_size is None:
        args.image_size = list(cfg["default_image_size"])
    if len(args.image_size) != 2:
        raise ValueError("--image_size expects H W, got %s" % args.image_size)
    return args


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    assert args.restore_ckpt.endswith(".pth"), args.restore_ckpt
    device = args.gpus[0]
    torch.cuda.set_device(device)

    logging.info("Loading checkpoint from %s", args.restore_ckpt)
    state_dict = load_raw_checkpoint(args.restore_ckpt)

    t0 = time.time()
    model = load_pytorch_model(args, state_dict, device)
    model.eval()
    logging.info(
        "Model ready in %.2fs; %.2fM learnable params",
        time.time() - t0,
        count_parameters(model) / 1e6,
    )

    split = args.dataset.replace("middlebury_", "")
    validate_middlebury(
        model,
        split=split,
        mixed_prec=args.mixed_precision,
        device=device,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
