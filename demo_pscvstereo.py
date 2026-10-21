"""Run inference with PSCVStereo (base or tiny) checkpoints.

Loads .pth checkpoints only, runs disparity inference on user-specified left/right
images, and writes a pseudo-color disparity PNG. Optionally saves raw disparity as .npy.

Example:
  python demo_pscvstereo.py --model tiny --restore_ckpt checkpoints/pscvstereo_tiny.pth \\
      -l path/to/im0.png -r path/to/im1.png --output_directory demo_output
"""
import argparse
import glob
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

sys.path.append("core")

from core.utils.utils import InputPadder

MODEL_REGISTRY = {
    "base": {
        "module": ("core.pscvstereo_base", "PSCVStereoBase"),
        "default_max_disp": 208,
        "default_iters": [5, 5],
    },
    "tiny": {
        "module": ("core.pscvstereo_tiny", "PSCVStereoTiny"),
        "default_max_disp": 176,
        "default_iters": [3, 5],
    },
}


def load_image(imfile, scale=1.0):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if scale != 1.0:
        if scale < 1.0:
            ksize = int(1.0 / scale * 2 + 1)
            img = cv2.GaussianBlur(img, (ksize, ksize), 1, 1)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    if len(img.shape) <= 2:
        img = img.reshape(img.shape[-2], img.shape[-1], 1)
        img = np.concatenate([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, 0:3]

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return type(state_dict)(
        (k[len("module."):] if k.startswith("module.") else k, v)
        for k, v in state_dict.items()
    )


def load_checkpoint(path):
    assert path.endswith(".pth"), f"Only .pth checkpoints are supported: {path}"
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return strip_module_prefix(ckpt)


def build_model(args):
    cfg = MODEL_REGISTRY[args.model]
    mod_name, cls_name = cfg["module"]
    module = importlib.import_module(mod_name)
    return getattr(module, cls_name)(
        max_disp=args.max_disp, iters=args.iters, test_mode=True
    )


def output_stem(imfile):
    path = Path(imfile)
    if path.parent.name not in ("", "."):
        return path.parent.name
    return path.stem


@torch.no_grad()
def demo(args):
    model = build_model(args)
    model.load_state_dict(load_checkpoint(args.restore_ckpt), strict=False)
    model.cpu()
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))
    if not left_images or not right_images:
        raise FileNotFoundError(
            f"No images matched left={args.left_imgs!r} right={args.right_imgs!r}"
        )
    if len(left_images) != len(right_images):
        raise ValueError(
            f"Left/right image count mismatch: {len(left_images)} vs {len(right_images)}"
        )

    print(f"Found {len(left_images)} image pair(s). Saving to {output_directory}/")

    for imfile_left, imfile_right in tqdm(list(zip(left_images, right_images))):
        image1 = load_image(imfile_left, scale=args.scale)
        image2 = load_image(imfile_right, scale=args.scale)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        flow_up, dumps = model(image1, image2, test_mode=True, dump=True)
        lrc_mask = dumps.lrc_mask
        disp = padder.unpad(flow_up[-1] * (lrc_mask >= 0.0).float())
        disp_np = disp.cpu().numpy().squeeze()

        file_stem = output_stem(imfile_left)
        plt.imsave(output_directory / f"disp_{file_stem}.png", -disp_np, cmap="jet")
        if args.save_npy:
            np.save(output_directory / f"disp_{file_stem}.npy", disp_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PSCVStereo (base or tiny) inference demo"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="PSCVStereo variant (base or tiny)",
    )
    parser.add_argument(
        "--restore_ckpt",
        required=True,
        help="path to .pth checkpoint",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="also save disparity as .npy",
    )
    parser.add_argument(
        "-l",
        "--left_imgs",
        required=True,
        help="path or glob for left image(s)",
    )
    parser.add_argument(
        "-r",
        "--right_imgs",
        required=True,
        help="path or glob for right image(s)",
    )
    parser.add_argument(
        "--output_directory",
        default="demo_output",
        help="directory to write disparity outputs",
    )
    parser.add_argument(
        "--max_disp",
        type=int,
        default=None,
        help="max disparity (default: 208 for base, 176 for tiny)",
    )
    parser.add_argument(
        "--iters",
        nargs="+",
        type=int,
        default=None,
        help="update iters per pyramid level (default: 5 5 for base, 3 5 for tiny)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="resize factor applied to input images before inference",
    )
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    if args.max_disp is None:
        args.max_disp = cfg["default_max_disp"]
    if args.iters is None:
        args.iters = cfg["default_iters"]

    demo(args)
