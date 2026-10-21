import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCorrBlock(nn.Module):
    def __init__(self, corr_split=2, reverse=False):
        super(AttentionCorrBlock, self).__init__()
        self.corr_list = []
        self.corr_split = corr_split
        self.reverse = reverse

    def get_full_corr(self):
        return [self.corr_list[i] for i in range(self.corr_split)]

    def set_external_corr(self, ext_corr_list):
        self.corr_list = ext_corr_list

    def calc_corr(self, fmap1, fmap2, corr_range=(-2, 32)):
        fmap1_list = fmap1.split(dim=1, split_size=fmap1.shape[1] // self.corr_split)
        fmap2_list = fmap2.split(dim=1, split_size=fmap2.shape[1] // self.corr_split)

        self.corr_list.clear()
        for p in range(self.corr_split):
            corr = AttentionCorrBlock.corr(
                fmap1_list[p],
                fmap2_list[p],
                corr_range,
                reverse=self.reverse)
            batch, h1, w1, w2 = corr.shape
            self.corr_list.append(corr.reshape(batch * h1 * w1, 1, 1, w2))

        return self.corr_list

    def __call__(self, disp, step, radius=4):
        r = radius
        batch, _, h1, w1 = disp.shape
        dx = torch.linspace(-r, r, 2 * r + 1, device=disp.device)
        dx = dx.view(1, 1, 2 * r + 1, 1)

        x0 = dx * step + disp.view(batch * h1 * w1, 1, 1, 1)

        hh, ww = self.corr_list[0].shape[-2:]
        x0_ = 2 * (x0 + 0.5) / ww - 1
        y0 = torch.zeros_like(x0_)
        coords = torch.cat([x0_, y0], dim=-1)

        corr_stacked = torch.cat([self.corr_list[p]
                                 for p in range(self.corr_split)], dim=1)
        corr1 = torch.nn.functional.grid_sample(
            corr_stacked, coords, mode='bilinear', align_corners=False)
        corr1 = corr1.view(batch, h1, w1, self.corr_split, 2 * r + 1)
        corr1 = corr1.permute(0, 3, 4, 1, 2).reshape(
            batch, self.corr_split * (2 * r + 1), h1, w1)

        return corr1.contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2, corr_range=(-2, 32), reverse=False):
        B, D, H, W1 = fmap1.shape
        fmap2_pad = F.pad(
            fmap2, (corr_range[1], -corr_range[0], 0, 0), mode='constant')

        shifts = [fmap2_pad[:, :, :, corr_range[1] - i:W1 + \
            (corr_range[1] - i)] for i in range(corr_range[0], corr_range[1])]
        fmap2_shifted = torch.stack(shifts, dim=0)
        corr = torch.einsum('bdhw,sbdhw->bhws', fmap1, fmap2_shifted)

        if reverse:
            corr = corr.flip(-1)
        return corr * (1 / D)
