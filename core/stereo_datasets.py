# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import copy
import random
from pathlib import Path
import cv2
from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

_REPO_ROOT = Path(__file__).resolve().parents[1]


def get_data_root(override=None) -> Path:
    """Resolve dataset root: CLI override, then env, then <repo>/datasets."""
    if override is not None:
        return Path(override).expanduser().resolve()
    env_root = os.environ.get("STEREO_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (_REPO_ROOT / "datasets").resolve()


def _resolve_dataset_path(subdir, root=None) -> str:
    if root is not None:
        return str(Path(root).expanduser().resolve())
    return str(get_data_root() / subdir)


def _glob_sorted(root, pattern):
    return sorted(str(path) for path in Path(root).glob(pattern))


class StereoDataset(data.Dataset):
    ext_downsamp_prob = 0.0
    max_train_disp = 192.0
    # Right-view disparity paths for datasets that flip stereo pairs in
    # __getitem__.
    _RIGHT_VIEW_FILES = {'disp1.pfm', 'disp5.png', 'right_disp_fs.png'}
    _RIGHT_VIEW_DIRS = {'Camera_1', 'depth_right'}

    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.downsamp_prob = 0.0
        self.augmentor = None
        self.sparse = sparse
        self.gray = False
        self.img_pad = aug_params.pop(
            "img_pad", None) if aug_params is not None else None

        if aug_params is not None and "crop_size" in aug_params:
            self.gray = aug_params["gray"]
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        # Allow aug_params=None / no crop_size for RAFT-style validation
        # loaders.
        self.crop_size = (
            aug_params["crop_size"]
            if aug_params is not None and "crop_size" in aug_params
            else None
        )
        self.blur = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def _add_stereo_pair(self, img1, img2, disp):
        self.image_list.append([img1, img2])
        self.disparity_list.append(disp)

    def _downsample_once(self, img1, img2, flow, valid):
        start_x = np.random.randint(0, 2)
        start_y = np.random.randint(0, 2)
        img1 = cv2.GaussianBlur(img1, (3, 3), 1, 1)
        img2 = cv2.GaussianBlur(img2, (3, 3), 1, 1)
        img1 = img1[start_y::2, start_x::2]
        img2 = img2[start_y::2, start_x::2]
        flow = flow[start_y::2, start_x::2] * 0.5
        valid = valid[start_y::2, start_x::2]
        return img1, img2, flow, valid

    def _apply_downsample_factor(self, img1, img2, flow, valid, factor):
        # factor encodes expected downsample count in two stages:
        # [0, 1] -> 0 or 1x; (1, 2] -> 1x + optional 2nd; >2 -> always 2x
        if np.random.rand() < factor:
            img1, img2, flow, valid = self._downsample_once(
                img1, img2, flow, valid)
        if np.random.rand() < factor - 1.0:
            img1, img2, flow, valid = self._downsample_once(
                img1, img2, flow, valid)
        return img1, img2, flow, valid

    def _downsample_for_large_disparity(self, img1, img2, flow, valid):
        for _ in range(3):
            if flow[valid].min() >= -self.max_train_disp:
                break
            img1, img2, flow, valid = self._downsample_once(
                img1, img2, flow, valid)
        return img1, img2, flow, valid

    def _flip_horizontal(self, img1, img2, flow, valid):
        return img1[:, ::-1], img2[:, ::-1], flow[:, ::-1], valid[:, ::-1]

    def _needs_right_view_flip(self, disp_path):
        parts = disp_path.split('/')
        return parts[-1] in self._RIGHT_VIEW_FILES or parts[-2] in self._RIGHT_VIEW_DIRS

    def _to_hwc3_uint8(self, img1, img2):
        """Convert a pair of PIL images to HWC uint8 arrays with exactly 3 channels."""
        if self.gray:
            g1 = np.array(img1.convert('L')).astype(np.uint8)
            g2 = np.array(img2.convert('L')).astype(np.uint8)
            img1 = np.stack([g1, g1, g1], axis=-1)
            img2 = np.stack([g2, g2, g2], axis=-1)
        else:
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            if img1.ndim == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
                img2 = np.tile(img2[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]
        return img1, img2

    def _init_worker_seed(self):
        if self.init_seed:
            return
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            torch.manual_seed(worker_info.id)
            np.random.seed(worker_info.id)
            random.seed(worker_info.id)
        self.init_seed = True

    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        self._init_worker_seed()
        index = index % len(self.image_list)

        # --- Disparity + validity mask ---
        disp = self.disparity_reader(self.disparity_list[index])
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        # --- Images as HWC uint8, 3-channel ---
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img1, img2 = self._to_hwc3_uint8(img1, img2)

        # --- Disparity -> horizontal flow (flow_x = -disp, flow_y = 0) ---
        disp = np.array(disp).astype(np.float32)
        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        # --- Stochastic / large-disparity downsampling ---
        img1, img2, flow, valid = self._apply_downsample_factor(
            img1, img2, flow, valid, StereoDataset.ext_downsamp_prob)
        img1, img2, flow, valid = self._apply_downsample_factor(
            img1, img2, flow, valid, self.downsamp_prob)
        img1, img2, flow, valid = self._downsample_for_large_disparity(
            img1, img2, flow, valid)

        # --- Flip if this sample is a right-view disparity ---
        if self._needs_right_view_flip(self.disparity_list[index]):
            img1, img2, flow, valid = self._flip_horizontal(
                img1, img2, flow, valid)

        # --- Augmentation ---
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(
                    img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        # --- To tensors ---
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)
        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

        flow = flow[:1]

        return (self.image_list[index] + [self.disparity_list[index]],
                img1, img2, flow, valid.float())

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root=None, split='training'):
        root = _resolve_dataset_path('ETH3D', root)
        super().__init__(aug_params, sparse=True)

        image1_list = _glob_sorted(root, 'two_view_training/*/im0.png')
        image2_list = _glob_sorted(root, 'two_view_training/*/im1.png')
        if split == 'training':
            disp_list = _glob_sorted(root, 'two_view_training/*/disp0GT.pfm')
        else:
            disp_path = str(Path(root) / 'two_view_training_gt/playground_1l/disp0GT.pfm')
            disp_list = [disp_path] * len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self._add_stereo_pair(img1, img2, disp)


class TartanAir(StereoDataset):
    _SUBSETS = (
        'carwelding',
    )

    @staticmethod
    def _collect_subset(root, subset):
        subset_root = Path(root) / subset
        return (
            _glob_sorted(subset_root, '*/depth_left/*_left_depth.npy'),
            _glob_sorted(subset_root, '*/image_left/*_left.png'),
            _glob_sorted(subset_root, '*/image_right/*_right.png'),
            _glob_sorted(subset_root, '*/depth_right/*_right_depth.npy'),
        )

    def __init__(self, aug_params=None, root=None):
        root = _resolve_dataset_path('tartanair', root)
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        disp_list = []
        image1_list = []
        image2_list = []
        disp1_list = []

        for subset in self._SUBSETS:
            d0, i0, i1, d1 = self._collect_subset(root, subset)
            disp_list += d0
            image1_list += i0
            image2_list += i1
            disp1_list += d1

        for img1, img2, disp, disp2 in zip(
                image1_list, image2_list, disp_list, disp1_list):
            self._add_stereo_pair(img1, img2, disp)
            self._add_stereo_pair(img2, img1, disp2)


class CustomDataset(StereoDataset):
    _SUBSETS = (
        # Add your custom dataset here
        'custom_subset_name',
    )

    @staticmethod
    def _collect_subset(root, subset):
        subset_root = Path(root) / subset
        return (
            _glob_sorted(subset_root, '*/im0.png'),
            _glob_sorted(subset_root, '*/im1.png'),
            _glob_sorted(subset_root, '*/disp.npy'),
        )

    def __init__(self, aug_params=None, root=None):
        root = _resolve_dataset_path('custom_dataset', root)
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispCustomDataset)
        assert os.path.exists(root)
        self.downsamp_prob = 0.5

        image1_list = []
        image2_list = []
        disp_list = []

        for subset in self._SUBSETS:
            i0, i1, d0 = self._collect_subset(root, subset)
            image1_list += i0
            image2_list += i1
            disp_list += d0

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self._add_stereo_pair(img1, img2, disp)


class VKITTI(StereoDataset):
    _SUBSETS = (
        '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right',
        'morning', 'overcast', 'sunset',
    )

    @staticmethod
    def _collect_subset(root, scene, subset):
        base = Path(root) / scene / subset / 'frames'
        depth0 = base / 'depth' / 'Camera_0'
        depth1 = base / 'depth' / 'Camera_1'
        rgb0 = base / 'rgb' / 'Camera_0'
        rgb1 = base / 'rgb' / 'Camera_1'
        return (
            _glob_sorted(depth0, 'depth_*.png'),
            _glob_sorted(depth1, 'depth_*.png'),
            _glob_sorted(rgb0, '*.jpg'),
            _glob_sorted(rgb1, '*.jpg'),
        )

    def __init__(self, aug_params=None, root=None, image_set='train'):
        root = _resolve_dataset_path('vkitti', root)
        super().__init__(
            aug_params, sparse=True, reader=frame_utils.readDispVKITTI)
        assert os.path.exists(root)
        self.downsamp_prob = 0.5
        scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
        disp_list = []
        disp2_list = []
        image1_list = []
        image2_list = []

        self.intrinsic_list = []
        for scene in scenes:
            for subset in self._SUBSETS:
                d0, d1, i0, i1 = self._collect_subset(root, scene, subset)
                disp_list += d0
                disp2_list += d1
                image1_list += i0
                image2_list += i1

        for idx, (img1, img2, disp, disp2) in enumerate(zip(
                image1_list, image2_list, disp_list, disp2_list)):
            if idx % 2 != 0:
                self._add_stereo_pair(img1, img2, disp)
            else:
                self._add_stereo_pair(img2, img1, disp2)


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root=None, split='F'):
        root = _resolve_dataset_path('Middlebury', root)
        super().__init__(
            aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in [
            "F", "H", "Q", "2005", "2006", "2014",
            "2014_no_middeval3", "2021",
        ]
        if split in ("2005", "2006"):
            self.downsamp_prob = 1.5
            scenes = sorted((Path(root) / split).glob("*"))
            for scene in scenes:
                required = [
                    scene / "view1.png",
                    scene / "view5.png",
                    scene / "disp1.png",
                    scene / "disp5.png",
                ]
                # Middlebury withholds GT for three 2005 scenes.
                if not scene.is_dir() or not all(path.exists() for path in required):
                    continue
                self._add_stereo_pair(str(required[0]), str(required[1]), str(required[2]))
                self._add_stereo_pair(str(required[1]), str(required[0]), str(required[3]))

        elif split in ("2014", "2014_no_middeval3"):
            # datasets/Middlebury/2014/... or 2014_no_middeval3/...
            self.downsamp_prob = 2.5
            for scene in (Path(root) / split).glob("*"):
                im0, im1 = scene / "im0.png", scene / "im1.png"
                self._add_stereo_pair(
                    str(im0), str(im1), str(scene / "disp0.pfm"))
                self._add_stereo_pair(
                    str(im1), str(im0), str(scene / "disp1.pfm"))

        elif split == "2021":
            self.downsamp_prob = 1.5
            for scene in (Path(root) / "2021").glob("*"):
                im0, im1 = scene / "im0.png", scene / "im1.png"
                self._add_stereo_pair(
                    str(im0), str(im1), str(scene / "disp0.pfm"))
                self._add_stereo_pair(
                    str(im1), str(im0), str(scene / "disp1.pfm"))

        else:
            if split == "F":
                self.downsamp_prob = 2.5
            elif split == "H":
                self.downsamp_prob = 1.5
            elif split == "Q":
                self.downsamp_prob = 0.5

            training = Path(root) / 'MiddEval3' / f'training{split}'
            scenes = sorted(path.name for path in training.iterdir() if path.is_dir())
            for name in scenes:
                scene = training / name
                self._add_stereo_pair(
                    str(scene / 'im0.png'),
                    str(scene / 'im1.png'),
                    str(scene / 'disp0GT.pfm'),
                )
            assert len(self.image_list) == len(scenes) > 0, [scenes, split]


class InStereo2K(StereoDataset):
    def __init__(self, aug_params=None, root=None, split='F'):
        root = _resolve_dataset_path('instereo2k', root)
        super().__init__(
            aug_params, sparse=True, reader=frame_utils.readDispInStereo2K)
        self.downsamp_prob = 1.5
        assert os.path.exists(root)
        image1_list = _glob_sorted(root, 'part*/*/left.jpg')
        image2_list = _glob_sorted(root, 'part*/*/right.jpg')
        disp_left_list = _glob_sorted(root, 'part*/*/left_disp_fs.png')
        disp_right_list = _glob_sorted(root, 'part*/*/right_disp_fs.png')

        for img1, img2, disp_left, disp_right in zip(
                image1_list, image2_list, disp_left_list, disp_right_list):
            self._add_stereo_pair(img1, img2, disp_left)
            self._add_stereo_pair(img2, img1, disp_right)


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """
    data_root = get_data_root(getattr(args, 'data_root', None))
    logging.info('Using dataset root: %s', data_root)

    aug_params = {
        'add_noise': 0.0,
        'crop_size': args.image_size,
        'min_scale': args.spatial_scale[0],
        'max_scale': args.spatial_scale[1],
        'do_flip': False,
        'yjitter': not getattr(args, 'noyjitter', False),
        'gray': getattr(args, 'gray', False),
        'blur': False,
        'all_left_prob': args.all_left_prob,
        'v_shift_prob': args.v_shift_prob,
        'eraser_aug_prob': args.eraser_aug_prob,
    }
    if getattr(args, 'saturation_range', None) is not None:
        aug_params['saturation_range'] = args.saturation_range
    if getattr(args, 'img_gamma', None) is not None:
        aug_params['gamma'] = args.img_gamma
    if getattr(args, 'do_flip', None) is not None:
        aug_params['do_flip'] = args.do_flip

    train_dataset = None
    for dataset_name, dataset_repeat in zip(
            args.train_datasets, args.dataset_repeat):
        if dataset_name.startswith("middlebury_"):
            mid_aug_params = copy.deepcopy(aug_params)
            mid_aug_params["add_noise"] = 0.0
            new_dataset = Middlebury(
                mid_aug_params,
                root=str(data_root / 'Middlebury'),
                split=dataset_name.replace('middlebury_', ''),
            ) * dataset_repeat
            logging.info(f"Adding {len(new_dataset)} samples from Middlebury")
        elif dataset_name.startswith("vkitti"):
            kitti_aug_params = copy.deepcopy(aug_params)
            kitti_aug_params["add_noise"] = 1.0
            kitti_aug_params["blur"] = True
            new_dataset = VKITTI(
                kitti_aug_params, root=str(data_root / 'vkitti')) * dataset_repeat
            logging.info(f"Adding {len(new_dataset)} samples from VKITTI")
        elif dataset_name.startswith("instereo2k"):
            kitti_aug_params = copy.deepcopy(aug_params)
            kitti_aug_params["add_noise"] = 0.0
            kitti_aug_params["blur"] = True
            new_dataset = InStereo2K(
                kitti_aug_params, root=str(data_root / 'instereo2k')) * dataset_repeat
            logging.info(f"Adding {len(new_dataset)} samples from InStereo2K")
        elif dataset_name == 'eth3d':
            new_dataset = ETH3D(
                aug_params, root=str(data_root / 'ETH3D')) * dataset_repeat
            logging.info(f"Adding {len(new_dataset)} samples from ETH3D")
        elif dataset_name == 'tartan_air':
            new_dataset = TartanAir(
                aug_params, root=str(data_root / 'tartanair')) * dataset_repeat
            logging.info(
                f"Adding {len(new_dataset)} samples from TartanAir "
                f"({', '.join(TartanAir._SUBSETS)})")
        elif dataset_name == 'custom_dataset':
            new_dataset = CustomDataset(
                aug_params, root=str(data_root / 'custom_dataset')) * dataset_repeat
            logging.info(
                f"Adding {len(new_dataset)} samples from CustomDataset "
                f"({', '.join(CustomDataset._SUBSETS)})")
        else:
            raise ValueError(f"Unknown train dataset: {dataset_name}")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    StereoDataset.ext_downsamp_prob = args.ext_downsamp_prob
    print('ext_downsamp_prob: ', StereoDataset.ext_downsamp_prob)
    StereoDataset.max_train_disp = getattr(args, 'max_train_disp', 192.0)
    print('max_train_disp: ', StereoDataset.max_train_disp)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2,
        drop_last=True
    )

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader
