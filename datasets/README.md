# Dataset layout

Training and evaluation scripts read stereo datasets from `$STEREO_DATA_ROOT` or, by default, this `datasets/` directory.

```bash
export STEREO_DATA_ROOT=/path/to/your/datasets   # optional
```

Download datasets from their official sources and arrange files to match the layouts below. Each name in `--train_datasets` maps to a loader in `core/stereo_datasets.py`.

**Which scenes or subsets are loaded** (e.g. TartanAir environments, VKITTI weather/view folders, custom data folders) is defined in `core/stereo_datasets.py` via each class’s `_SUBSETS` (and the scene list in `VKITTI`). Change those constants in code if you want a different mix.

## Overview

```
$STEREO_DATA_ROOT/
├── vkitti/
├── Middlebury/
│   ├── 2014_no_middeval3/
│   ├── 2021/
│   └── MiddEval3/          # evaluation
├── ETH3D/
├── instereo2k/
├── tartanair/
└── custom_dataset/
```

## Virtual KITTI (`vkitti`)

Root: `vkitti/`. Loader: `VKITTI` in `core/stereo_datasets.py`.

```
vkitti/
└── <scene>/
    └── <subset>/
        └── frames/
            ├── rgb/
            │   ├── Camera_0/*.jpg
            │   └── Camera_1/*.jpg
            └── depth/
                ├── Camera_0/depth_*.png
                └── Camera_1/depth_*.png
```

## TartanAir (`tartan_air`)

Root: `tartanair/`. Loader: `TartanAir` in `core/stereo_datasets.py`. Depth is stored as `.npy`; disparity is derived at load time.

```
tartanair/
└── <subset>/
    └── <trajectory>/
        ├── image_left/*_left.png
        ├── image_right/*_right.png
        ├── depth_left/*_left_depth.npy
        └── depth_right/*_right_depth.npy
```

Filenames must sort consistently across the four folders (same count and ordering).

## Custom dataset (`custom_dataset`)

Root: `custom_dataset/`. Loader: `CustomDataset` in `core/stereo_datasets.py`.

```
custom_dataset/
└── <subset>/
    └── <scene>/
        ├── im0.png
        ├── im1.png
        └── disp.npy
```

## Middlebury 2014 (`middlebury_2014_no_middeval3`)

Root: `Middlebury/2014_no_middeval3/`. This is not a separate official download — start from the [Middlebury 2014](https://vision.middlebury.edu/stereo/data/scenes2014/) scenes and place them here, **excluding** any scene pairs that overlap with the MiddEval3 training set (so training and evaluation do not share the same scenes).

```
Middlebury/2014_no_middeval3/
└── <scene-name>/
    ├── im0.png
    ├── im1.png
    ├── disp0.pfm
    └── disp1.pfm
```

## Middlebury 2021 (`middlebury_2021`)

Root: `Middlebury/2021/`.

```
Middlebury/2021/
└── <scene-name>/
    ├── im0.png
    ├── im1.png
    ├── disp0.pfm
    └── disp1.pfm
```

## ETH3D (`eth3d`)

Root: `ETH3D/`.

```
ETH3D/
└── two_view_training/
    └── <scene-name>/
        ├── im0.png
        ├── im1.png
        └── disp0GT.pfm
```

## InStereo2K (`instereo2k`)

Root: `instereo2k/`.

```
instereo2k/
└── part*/
    └── <scene-id>/
        ├── left.jpg
        ├── right.jpg
        ├── left_disp_fs.png
        └── right_disp_fs.png
```

## Evaluation (MiddEval3)

`evaluate_pscvstereo.py` expects Middlebury data under `Middlebury/`. For splits **F**, **H**, and **Q**:

```
Middlebury/MiddEval3/
├── trainingF/
│   └── <scene-name>/
│       ├── im0.png
│       ├── im1.png
│       ├── disp0GT.pfm
│       └── mask0nocc.png
├── trainingH/
│   └── ...
└── trainingQ/
    └── ...
```

Download from the [Middlebury Stereo Evaluation](https://vision.middlebury.edu/stereo/) site.
