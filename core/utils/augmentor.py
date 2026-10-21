import numpy as np
import random
from PIL import Image
import math
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter, functional, Compose
import torch.nn.functional as F

def block_edge(disp, valid, thresh, block_radius=1):
    height, width = disp.shape[:2]
    valid_ = (disp == disp)

    if block_radius == 1 and height > 2 and width > 2:
        # Vectorized fast path for the common block_radius=1 case.
        # For block_radius=1 the loop only visits diagonal neighbours
        # (i,j) in {(-1,-1),(-1,1),(1,-1),(1,1)}, for which
        # max(|i|,|j|) == 1, so the threshold is always -thresh.
        # AND is idempotent, so the repeated `& valid[interior]` across
        # iterations collapses to a single AND with valid[interior].
        t = -thresh
        cy, cx = slice(1, height - 1), slice(1, width - 1)
        center_disp = disp[cy, cx]
        result = valid_[cy, cx] & valid[cy, cx]
        for di, dj in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            ny, nx = slice(1 + di, height - 1 + di), slice(1 + dj, width - 1 + dj)
            result &= ~valid[ny, nx] | (center_disp - disp[ny, nx] > t)
        valid_[cy, cx] = result
    else:
        # General (block_radius != 1) fallback: original loop semantics.
        # For each interior pixel, invalidate it when a diagonal neighbour
        # (i,j) (both nonzero) has disparity higher than the center by more
        # than thresh * max(|i|,|j|) and that neighbour is itself valid.
        b = block_radius
        interior = (slice(b, height - b), slice(b, width - b))
        center_disp = disp[interior]
        for i in range(-b, b + 1):
            for j in range(-b, b + 1):
                if i == 0 or j == 0:
                    continue
                neighbor = (slice(b + i, height - b + i), slice(b + j, width - b + j))
                t = -thresh * max(abs(i), abs(j))
                cond = ~valid[neighbor] | (center_disp - disp[neighbor] > t)
                valid_[interior] = valid_[interior] & valid[interior] & cond
    return valid_


class AdjustGamma(object):
    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, yjitter=False, saturation_range=[0.6,1.4], gamma=[0.8,1.2,0.8,1.0],add_noise = 1.0,gray=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, img1.shape[-1]), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0]+1) / float(ht),
            (self.crop_size[1]+1) / float(wd))

        scale = 2 ** np.random.uniform(min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)
        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]
        if len(img1.shape) == 2:
            img1 = np.expand_dims(img1, axis = -1)
            img2 = np.expand_dims(img2, axis = -1)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp
            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)
            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y1:y1+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        return img1, img2, flow


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=0.8, max_scale=1.2, do_flip=False, yjitter=False, saturation_range=[0.8,1.2], gamma=[0.8,1.2,1,1], add_noise = 1.0,gray=False,blur=False,all_left_prob = 0.05, v_shift_prob = 0.5,eraser_aug_prob=0.5):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.noise = add_noise
        self.noise_prob = add_noise
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.1
        self.v_flip_prob = 0.1
        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.2, contrast=0.2, saturation=saturation_range, hue=0.3/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = eraser_aug_prob # don't do random erase
        self.all_left_prob = all_left_prob # to improve robustness in repetive scene, set this to 0.05 when fine-tuning,don't set this when pretrain
        self.v_shift_prob = v_shift_prob # to improve robustness in repetive scene, set this to 0.5 when fine-tuning,don't set this when pretrain
        self.inverse_color_prob = 0.0 # to improve robustness in different color style scene
        self.rng = np.random.RandomState(1234)

    def color_transform(self, img1, img2):
        shape = img1.shape
        if shape[-1] == 1:
            img1 = img1.squeeze()
            img2 = img2.squeeze()
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)
        if shape[-1] == 1:
            img1 = img1.reshape(shape)
            img2 = img2.reshape(shape)

        if np.random.rand() < self.inverse_color_prob and img1.ndim == 3 and img1.shape[-1] == 3:
            perm = ((0, 2, 1), (2, 1, 0), (1, 0, 2))[random.randint(0, 2)]
            img1 = img1[..., perm]
            img2 = img2[..., perm]

        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        # Eraser patch is 50-100px; skip when downsample left a frame
        # too small for randint(0, ht-dy) / randint(0, wd-dx).
        if np.random.rand() < self.eraser_aug_prob and ht > 50 and wd > 50:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            erase_num = 1
            for _ in range(erase_num):
                dx = np.random.randint(50, min(100, wd))
                dy = np.random.randint(50, min(100, ht))
                x0 = np.random.randint(0, wd-dx)
                y0 = np.random.randint(0, ht-dy)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        # Coordinates of valid pixels in row-major order (matches the original
        # meshgrid + stack + reshape ordering), without building a full
        # (ht, wd, 2) coordinate grid.
        valid_mask = valid >= 1
        valid_mask_flat = valid_mask.ravel()
        rows, cols = np.nonzero(valid_mask)

        # Scale original (x, y) = (col, row) and round to integer pixel positions.
        xx = np.round(cols.astype(np.float32) * fx).astype(np.int32)
        yy = np.round(rows.astype(np.float32) * fy).astype(np.int32)

        # Keep only points strictly inside the new image bounds.
        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]

        # Gather flow and validity at the surviving points, then scale.
        flow1 = flow.reshape(-1, 2)[valid_mask_flat][v].astype(np.float32) * [fx, fy]
        valid1 = valid.ravel().astype(np.float32)[valid_mask_flat][v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = valid1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        # scale = np.random.uniform(self.min_scale, self.max_scale)
        scale = np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.v_shift_prob:
            left_v_shift = self.rng.uniform(-1.0/scale_y, 1.0/scale_y)
            right_v_shift = self.rng.uniform(-1.0/scale_y, 1.0/scale_y)
            ag = math.atan((left_v_shift-right_v_shift)/img2.shape[0])
            px = (left_v_shift-right_v_shift)/2
            py=0
            image_center = (img2.shape[0]/2,img2.shape[1]/2)

            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            rot_mat[1,2] = rot_mat[1,2] + px
            img2 = cv2.warpAffine(
                img2, rot_mat, img2.shape[1::-1], flags=cv2.INTER_LINEAR
            )

        if np.random.rand() < self.spatial_aug_prob:
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(
                flow, valid, fx=scale_x, fy=scale_y
            )
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        if img1.shape[0] < (self.crop_size[0]+1) or img1.shape[1] < (self.crop_size[1]+1):
            if img1.shape[0] < (self.crop_size[0]+1):
                pad_h = (self.crop_size[0]+1-img1.shape[0])
            else:
                pad_h = 0
            if img1.shape[1] < (self.crop_size[1]+1):
                pad_w = (self.crop_size[1]+1-img1.shape[1])
            else:
                pad_w = 0
            img1 = np.pad(img1,((0,pad_h+7),(0,pad_w+7),(0,0)),'constant')
            img2 = np.pad(img2,((0,pad_h+7),(0,pad_w+7),(0,0)),'constant')
            flow = np.pad(flow,((0,pad_h+7),(0,pad_w+7),(0,0)),'constant')
            valid = np.pad(valid,((0,pad_h+7),(0,pad_w+7)),'constant')

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        if self.do_flip:
            if self.do_flip in ('hf', 'h', 'hv') and np.random.rand() < self.h_flip_prob:
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]
            if self.do_flip in ('v', 'hv') and np.random.rand() < self.v_flip_prob:
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                valid = valid[::-1, :]

        flow_ = -flow[:,:,0]
        flow_[~valid] = 0.0
        block_edge_valid = block_edge(flow_,valid, 1.0,1)

        return img1, img2, flow, valid & block_edge_valid

    def __call__(self, img1, img2, flow, valid):
        if np.random.rand() < self.all_left_prob:
            ht, wd = img1.shape[:2]
            disp = np.random.randint(low=0, high=3)
            if np.random.rand() < 0.5:
                img1=img2
            img2[:,:wd-disp] = img1[:,disp:]
            img2[:,wd-disp:] = 0
            flow[:] = -disp
            valid[:] = True
            if disp > 0:
                valid[:,0:disp] = False

        img1, img2, flow, valid = self.spatial_transform(
            img1, img2, flow, valid
        )

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid
