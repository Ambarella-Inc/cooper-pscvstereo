# Cooper-PSCVStereo

**Cooper-PSCVStereo** is a PyTorch stereo matching network for rectified image pairs. It estimates disparity iteratively from coarse to fine with a **Progressive Slim Cost Volume**: a compact, multi-scale correlation volume. It includes modified portions of [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [IGEV-Stereo](https://github.com/gangweiX/IGEV), and [CREStereo](https://github.com/megvii-research/CREStereo).

Two variants are provided: **`base`** and **`tiny`**. `tiny` is a more lightweight variant.

Scripts are included for inference, training, and Middlebury evaluation.

## Installation

Python 3.8 and CUDA 11.6 are recommended (`requirements.txt` pins PyTorch for `cu116`).

```bash
conda create -n pscvstereo python=3.8 -y
conda activate pscvstereo
pip install -r requirements.txt
```

If your CUDA version differs, adjust the `torch` / `torchvision` entries in `requirements.txt` accordingly.

## Demo

Run disparity inference on a stereo pair. Output is a pseudo-color PNG in `--output_directory` (add `--save_npy` for raw disparity).

Prepare your own left/right images:

```bash
python demo_pscvstereo.py \
  --model base \
  --restore_ckpt pscvstereo_base.pth \
  -l left.png \
  -r right.png \
  --output_directory demo_output
```

Use `--model tiny` with a tiny checkpoint for the lighter variant.

## Training

Prepare datasets under `$STEREO_DATA_ROOT` or `./datasets` — see [datasets/README.md](datasets/README.md).

Training is driven by `train_pscvstereo.py`. Example shell scripts are included for reference only:

- `train_base.sh`
- `train_tiny.sh`

You may adapt datasets, hyperparameters, and GPU settings to your own setup.

Checkpoints go to `checkpoints/`. Logs go to `runs/` (TensorBoard).

## Evaluation on Middlebury

Evaluate a `.pth` checkpoint on Middlebury. `evaluate_pscvstereo.py` reports **RAFT-Stereo EPE** (mean absolute disparity error, pixels) and **D1** (percent of valid pixels with error greater than 2 px).

You can also use the official [Middlebury stereo evaluation](https://vision.middlebury.edu/stereo/eval3/) metrics (MiddEval3: coverage, bad 0.5/1/2/4, avgerr, rms, error quantiles) by running the benchmark toolchain.

**Q** (quarter resolution) is recommended for both `base` and `tiny` — its image size is closer to the training crop.

Place data as in [datasets/README.md](datasets/README.md#evaluation-middeval3).

```bash
python evaluate_pscvstereo.py \
  --model base \
  --dataset middlebury_Q \
  --restore_ckpt pscvstereo_base.pth
```

Other splits: `middlebury_F`, `middlebury_H`.

Useful flags: `--mixed_precision`, `--gpus`.

## License

See [LICENSE](LICENSE). Ambarella-owned portions are Apache 2.0. Modified third-party portions come from RAFT-Stereo and IGEV-Stereo (MIT) and CREStereo (Apache 2.0).
