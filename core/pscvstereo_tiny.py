from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import FlowHead, ConvMiniGRU
from core.extractor import BasicEncoderTiny, FeatureTiny, Clamp8
from core.corr import AttentionCorrBlock


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        if self.relu:
            self.relu_out = Clamp8()
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=True, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=True, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu_out(x)
        return x

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))
        self.conv2 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))
        self.conv3 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))
        self.conv3_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv2_up = BasicConv(in_channels * 2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_out = BasicConv(in_channels, in_channels, is_3d=True, bn=False, relu=False, kernel_size=3,
                      padding=1, stride=1, dilation=1)
        self.agg_0 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1))
        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.classifier = nn.Conv3d(4, 1, 3, 1, 1, bias=False)
        self.corr_out = nn.Conv3d(4, 1, 3, 1, 1, bias=False)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = conv1 * features[0]
        conv2 = self.conv2(conv1)
        conv2 = conv2 * features[1]
        conv3 = self.conv3(conv2)
        conv3 = conv3 * features[2]
        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), 1)
        conv2 = self.agg_0(conv2)
        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), 1)
        conv1 = self.agg_1(conv1)
        conv = self.conv_out(conv1)

        return self.classifier(conv), self.corr_out(conv)[:, 0]


class TwoStepUpsamp(nn.Module):
    def __init__(self, upsample_ratio = 8):
        super().__init__()
        self.hidden_dims = 64

        self.mask_ctx = nn.Sequential(
            nn.Conv2d(self.hidden_dims, self.hidden_dims, kernel_size=3, padding=1,bias=True),
            nn.ReLU(inplace=True))
        self.mask_expand = nn.Sequential(
            nn.Conv2d(self.hidden_dims, self.hidden_dims*2, kernel_size=3, padding=1,bias=True),
            nn.ReLU(inplace=True))

        self.mask_out = nn.ConvTranspose2d(self.hidden_dims+64, 9, kernel_size=upsample_ratio, stride=upsample_ratio, padding=0)

    def forward(self, net):
        up_mask_context = self.mask_ctx(net)
        up_mask_context = self.mask_expand(up_mask_context)
        return self.mask_out(up_mask_context)


def corr_downsample(corr):
    N, D, H, W = corr.shape
    corr_down = F.avg_pool2d(corr,(2,1),(2,1))
    corr_down_00 = corr_down[:,0::2,:,1::2]
    corr_down_01 = corr_down[:,1::2,:,1::2]
    corr_down_10 = torch.cat([torch.zeros(N,1,H//2,W//2,device=corr.device),corr_down[:,1:D-1:2,:,0::2]],1)
    corr_down_11 = corr_down[:,0::2,:,0::2]
    return (corr_down_00 + corr_down_01 + corr_down_10 + corr_down_11) / 4


def disparity_regression(x, maxdisp):
    x = F.softmax(x.squeeze(1), dim=1)
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


@dataclass
class PSCVStereoDumps:
    lrc_mask: torch.Tensor
    corrs_dump: list
    disps_dump: list
    nets_dump: list
    disp_r: torch.Tensor
    inp_0_dump: list
    up_mask_context_d4_dump: list
    fmap1: torch.Tensor
    fmap2: torch.Tensor
    feat_x_list: list
    corr_d8_0_0: torch.Tensor


class PSCVStereoTiny(nn.Module):
    def __init__(self, max_disp=176, iters=(3, 5), test_mode=False):
        super().__init__()
        self.test_mode = test_mode

        self.pyrd_levels = 2
        self.corr_split = 1
        self.max_disp = max_disp
        self.iters = list(iters)

        self.step0 = PSCVStereoStep0()
        self.step1 = PSCVStereoStep1()
        self.step2 = PSCVStereoStep2()
        self.coarse_upsamp = TwoStepUpsamp(8)

        self.corr_agg = hourglass(4)

        self.corr_block_in = AttentionCorrBlock(4)
        self.corr_block = nn.ModuleList([AttentionCorrBlock(self.corr_split) for i in range(self.pyrd_levels)])

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, image1, image2, test_mode=False, dump=False):
        """ Estimate optical flow between pair of frames """
        predictions = []
        if dump:
            disps_dump = []
            corrs_dump = []
            nets_dump = []
            inp_0_dump = []
            up_mask_context_d4_dump = []

        fmap1, feat_x_list, inp_list, net_list, upsamp_context_d4 = self.step0(image1)
        fmap2, feat_x_list_r, _, _, _ = self.step0(image2, get_ctx=False)

        last_disp = None
        corr_pad = 2
        radius_list = [4, 4, 4]
        max_disp_d8 = self.max_disp // 8 + corr_pad

        self.corr_block[0].calc_corr(
            fmap1[0], fmap2[0],
            corr_range=(-corr_pad, self.max_disp // 8),
        )
        corr_d8 = self.corr_block[0].get_full_corr()

        self.corr_block_in.calc_corr(
            fmap1[0][:, 0:128, :, :], fmap2[0][:, 0:128, :, :],
            corr_range=(-corr_pad, self.max_disp // 8),
        )
        corr_d8_in = self.corr_block_in.get_full_corr()

        feat_x_list = [f.unsqueeze(2) for f in feat_x_list]

        N0, _, H0, W0 = net_list[0].shape
        corr_d8_0_0 = torch.cat(
            [corr_d8_in[i].reshape(N0, H0, W0, max_disp_d8).permute(0, 3, 1, 2).unsqueeze(1)
             for i in range(4)],
            dim=1,
        )

        disp_cost_init, corr_d8_0 = self.corr_agg(corr_d8_0_0, feat_x_list)
        disp_init = disparity_regression(disp_cost_init, max_disp_d8)

        corr_d16_0 = corr_downsample(corr_d8_0)
        corr_d16_0 = corr_d16_0.permute(0, 2, 3, 1).reshape(-1, 1, 1, self.max_disp // 16 + corr_pad // 2)
        corr_d8_0 = corr_d8_0.permute(0, 2, 3, 1).reshape(-1, 1, 1, max_disp_d8)

        self.corr_block[0].set_external_corr(
            [(corr_d8_0 + corr_d8[0]) / 2])
        self.corr_block[1].calc_corr(
            fmap1[1], fmap2[1],
            corr_range=(-corr_pad // 2, self.max_disp // 16)
        )
        corr_d16 = self.corr_block[1].get_full_corr()
        self.corr_block[1].set_external_corr(
            [(corr_d16_0 + corr_d16[0]) / 2])

        iters = self.iters
        if iters[0] > 0:
            disp0 = F.avg_pool2d(disp_init - corr_pad, 2, 2) / 2.0
        else:
            disp0 = disp_init - corr_pad

        for level in range(0,self.pyrd_levels):
            if dump:
                disps_dump_ = []
                corrs_dump_ = []
                nets_dump_ = []
                inp_0_dump_ = []
                up_mask_context_d4_dump_ = []

            idx_this = self.pyrd_levels-1-level
            disp1 = disp0

            for itr in range(iters[level]):
                step=1.0
                disp1 = disp1.detach()
                corr = self.corr_block[idx_this](disp1 + 2.0**level, step, radius=radius_list[level])

                if upsamp_context_d4.shape[-1] == net_list[idx_this].shape[-1] * 4:
                    upsamp_ctx = F.interpolate(
                        upsamp_context_d4,
                        size=(upsamp_context_d4.shape[2] // 2, upsamp_context_d4.shape[3] // 2),
                        mode="bilinear",
                        align_corners=False,
                        )
                else:
                    upsamp_ctx = upsamp_context_d4
                if dump:
                    disps_dump_.append(disp1.contiguous())
                    corrs_dump_.append(corr.contiguous())
                    nets_dump_.append(net_list[idx_this].contiguous())
                    inp_0_dump_.append(inp_list[idx_this])
                    up_mask_context_d4_dump_.append(upsamp_ctx)

                inpz, inpq = inp_list[idx_this].split(dim=1, split_size=inp_list[idx_this].shape[1]//2)

                is_last_step = (level == self.pyrd_levels-1 and itr == iters[level]-1)
                disp1, net_list[idx_this], upmask_1 = self.step1(disp1, corr, net_list[idx_this], inpz, inpq, upsamp_ctx, is_last_step=is_last_step)
                last_disp = disp1

                if (not self.test_mode) or is_last_step:
                    N,_,H,W = disp1.shape
                    scale = 2**(idx_this) * 8
                    if is_last_step:
                        up_mask = upmask_1.view(N,9,H,8,W,8).permute(0,1,3,5,2,4).reshape(N,9*8*8,H,W)
                    else:
                        up_mask_1 = self.coarse_upsamp(net_list[idx_this])
                        up_mask = up_mask_1.view(N,9,H,8,W,8).permute(0,1,3,5,2,4).reshape(N,9*8*8,H,W)
                    predictions.append(-self.step2(disp1, up_mask, scale, up_mask_scale=8))

            if dump:
                corrs_dump.append(corrs_dump_)
                nets_dump.append(nets_dump_)
                inp_0_dump.append(inp_0_dump_)
                disps_dump.append(disps_dump_)
                up_mask_context_d4_dump.append(up_mask_context_d4_dump_)

            if iters[level] > 0:
                disp0 = 2.0 * F.interpolate(
                        disp1,
                        size=(disp1.shape[2]*2, disp1.shape[3]*2),
                        mode="bilinear",
                        align_corners=False,
                        )

        predictions.insert(-2,-self.step2(disp_init-corr_pad,up_mask.detach(),8,8))

        dumps = None
        if dump:
            cost_max = F.softmax(disp_cost_init.squeeze(1),dim=1)
            cost_max,cost_max_idx = cost_max.max(dim=1,keepdim=True)
            cost_mask = (cost_max > 0.3).float()

            feat_x_list_r = [f.unsqueeze(2) for f in feat_x_list_r]

            disp_init_corr_pad = disp_init - corr_pad
            disp_r = compute_reverse_disp(
                fmap2[0][:, 0:128, :, :], fmap1[0][:, 0:128, :, :],
                feat_x_list_r, self.corr_agg, self.max_disp // 8)
            lrc_mask = get_lrc_mask(disp_init_corr_pad, disp_r, 0.5)
            lrc_mask_1 = get_lrc_mask(last_disp, disp_r, 0.5)
            diff_mask = ((last_disp - disp_init_corr_pad).abs() < 0.5).float()
            lrc_mask = lrc_mask * diff_mask * lrc_mask_1 * cost_mask
            lrc_mask = self.step2(lrc_mask, up_mask.detach(), 8, 8) / 8.0

            dumps = PSCVStereoDumps(
                lrc_mask=lrc_mask,
                corrs_dump=corrs_dump,
                disps_dump=disps_dump,
                nets_dump=nets_dump,
                disp_r=disp_r,
                inp_0_dump=inp_0_dump,
                up_mask_context_d4_dump=up_mask_context_d4_dump,
                fmap1=fmap1[0],
                fmap2=fmap2[0],
                feat_x_list=feat_x_list,
                corr_d8_0_0=corr_d8_0_0,
            )
        return predictions, dumps


class PSCVStereoStep0(nn.Module):
    def __init__(self):
        super().__init__()

        self.pyrd_levels = 2
        self.input_dims = 3
        self.encoder_dims = 64
        self.feature_dims = 128
        self.context_dims = 64
        self.hidden_dims = 64

        self.act = Clamp8
        self.cnet1 = BasicEncoderTiny(self.input_dims, self.encoder_dims, act=self.act)
        self.feat_conv = nn.Conv2d(self.encoder_dims, self.feature_dims, 1, 1, 0)
        self.feat_clamp = nn.Hardtanh(-7.9375, 7.9375)
        self.inp_conv = nn.Conv2d(self.encoder_dims, self.context_dims * 2, 1, 1, 0)

        self.upsamp_context_d4 = nn.Sequential(
            nn.Conv2d(32,32,3,2,1),
            self.act(inplace=True),
            nn.Conv2d(32,32,3,1,1),
            self.act(inplace=True)
        )

        self.net_conv = nn.Conv2d(self.encoder_dims,self.hidden_dims,1,1)

        self.net_tanh = nn.ModuleList([nn.Tanh() for i in range(self.pyrd_levels)])
        self.inp_relu = nn.ModuleList([self.act(inplace=True) for i in range(self.pyrd_levels)])

        self.net_avgpool2d = nn.ModuleList([nn.AvgPool2d(2,2) for i in range(self.pyrd_levels-1)])
        self.inp_avgpool2d = nn.ModuleList([nn.AvgPool2d(2,2) for i in range(self.pyrd_levels-1)])
        self.fmap_avgpool2d = nn.ModuleList([nn.AvgPool2d(2,2) for i in range(self.pyrd_levels-1)])

        self.feat_encoder = FeatureTiny(act=Clamp8, act_out=Clamp8, input_dims=self.input_dims)

        self.feat_att1 = nn.Sequential(
            BasicConv(64, 64 // 2, kernel_size=1, stride=1, padding=0, relu=False),
            self.act(inplace=True),
            nn.Conv2d(64 // 2, 4, 1),
            nn.Sigmoid())
        self.feat_att2 = nn.Sequential(
            BasicConv(48, 48 // 2, kernel_size=1, stride=1, padding=0, relu=False),
            self.act(inplace=True),
            nn.Conv2d(48 // 2, 8, 1),
            nn.Sigmoid())
        self.feat_att3 = nn.Sequential(
            BasicConv(80, 80 // 2, kernel_size=1, stride=1, padding=0, relu=False),
            self.act(inplace=True),
            nn.Conv2d(80 // 2, 16, 1),
            nn.Sigmoid())

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, image1, get_ctx=True):
        """ Estimate optical flow between pair of frames """
        fmap_list = []
        feat_x_list = []
        net_list = []
        inp_list = []
        up_d4 = None

        image = image1 * (2.0 / 256.0) + (-1.0)
        feat_code_d8, feat_x8, feat_x16, feat_x32 = self.feat_encoder(image)

        fmap1 = self.feat_clamp(self.feat_conv(feat_code_d8))
        fmap_list.append(fmap1)
        fmap_list.append(self.fmap_avgpool2d[0](fmap1))

        feat_x_d8 = self.feat_att1(feat_x8)
        feat_x_d16 = self.feat_att2(feat_x16)
        feat_x_d32 = self.feat_att3(feat_x32)
        feat_x_list.append(feat_x_d8)
        feat_x_list.append(feat_x_d16)
        feat_x_list.append(feat_x_d32)

        if get_ctx:
            code1, code_d4, code_d2 = self.cnet1(image)

            inp_0 = self.inp_relu[0](self.inp_conv(code1))
            inp_1 = self.inp_avgpool2d[0](inp_0)
            inp_list.append(inp_0)
            inp_list.append(inp_1)

            net_0 = self.net_tanh[0](self.net_conv(code1))
            net_1 = self.net_avgpool2d[0](net_0)
            net_list.append(net_0)
            net_list.append(net_1)

            up_d4 = self.upsamp_context_d4(code_d2)

        return fmap_list, feat_x_list, inp_list, net_list, up_d4


class PSCVStereoStep1(nn.Module):
    def __init__(self):
        super().__init__()

        self.pyrd_levels = 2
        self.encoder_dims = 64
        self.context_dims = 64
        self.hidden_dims = 64
        self.corr_split = 1
        self.corr_radius = 4
        self.act = Clamp8

        self.gru = ConvMiniGRU(self.hidden_dims, self.context_dims + self.hidden_dims - 1)

        self.up_mask_scale = 8

        self.motion_convc1 = nn.Conv2d((self.corr_radius * 2 + 1) * self.corr_split, self.hidden_dims, 1, padding=0)
        self.motion_convc2 = nn.Conv2d(self.hidden_dims, self.hidden_dims, 3, padding=1)
        self.motion_convf1 = nn.Conv2d(1, self.hidden_dims, 7, padding=3)
        self.motion_convf2 = nn.Conv2d(self.hidden_dims, self.hidden_dims//2, 3, padding=1)
        self.motion_conv = nn.Conv2d(self.hidden_dims//2+self.hidden_dims, self.hidden_dims-1, 3, padding=1)
        self.flow_head=FlowHead(self.hidden_dims,self.hidden_dims * 2,1,act=nn.ReLU)

        self.disp_convz = nn.Conv2d(1,self.hidden_dims,3,padding=1,bias=False)
        self.disp_convq = nn.Conv2d(1,self.hidden_dims,3,padding=1,bias=False)

        self.relu_corr1 = self.act(inplace=True)
        self.relu_corr2 = self.act(inplace=True)
        self.relu_flow1 = self.act(inplace=True)
        self.relu_flow2 = self.act(inplace=True)
        self.relu_motion = self.act(inplace=True)

        self.mask_ctx = nn.Sequential(
            nn.Conv2d(self.hidden_dims, self.hidden_dims, kernel_size=3, padding=1,bias=True),
            self.act(inplace=True))
        self.mask_refine = nn.Sequential(
            nn.Conv2d(self.hidden_dims, self.hidden_dims, kernel_size=3, padding=1,bias=True),
            self.act(inplace=True))
        self.mask_up2x = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims, 32, kernel_size=4, stride=2,padding=1,bias=False),
            self.act(inplace=True))
        self.mask_fuse = nn.Sequential(
            nn.Conv2d(32+32, 64, kernel_size=3, stride=1,padding=1),
            self.act(inplace=True))
        self.mask_out = nn.ConvTranspose2d(64, 9, kernel_size=8, stride=4,padding=2)

    def forward(self, disp, corr, net, inpz, inpq, upsamp_context_d4, is_last_step=True):
        cor = self.relu_corr1(self.motion_convc1(corr))
        cor = self.relu_corr2(self.motion_convc2(cor))
        flo = self.relu_flow1(self.motion_convf1(disp))
        flo = self.relu_flow2(self.motion_convf2(flo))
        cor_flo = torch.cat([cor, flo], 1)
        out = self.relu_motion(self.motion_conv(cor_flo))

        disp_offset = disp + (-16.0)
        disp_z = self.disp_convz(disp_offset)
        disp_q = self.disp_convq(disp_offset)
        net = self.gru(net, disp_z, disp_q, inpz, inpq, out)

        delta_disp = self.flow_head(net)
        disp = disp + delta_disp

        up_mask = None
        if is_last_step:
            up_mask_context = self.mask_ctx(net)
            up_mask = self.mask_refine(up_mask_context)
            up_mask = self.mask_up2x(up_mask)
            up_mask = self.mask_fuse(torch.cat([up_mask, upsamp_context_d4], 1))
            up_mask = self.mask_out(up_mask)

        return disp, net, up_mask

class PSCVStereoStep2(nn.Module):
    def __init__(self):
        super().__init__()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask, scale=4):
        """Upsample `disp` by `scale` using a 3x3 softmax-weighted neighborhood.

        mask: (N, 9*scale*scale, H, W) logits. softmax is taken over the 9
            spatial neighbors for each of the scale*scale output sub-positions.
        disp: (N, D, H, W) disparity to be upsampled.
        """
        N, D, H, W = disp.shape
        mask = mask.view(N * D, 9, scale * scale, H * W)
        mask = torch.softmax(mask, dim=1)

        # 3x3 spatial neighborhood of the padded disparity. The 9 slices are
        # ordered row-major over (dy, dx) in {-1, 0, 1} x {-1, 0, 1} to match
        # the 9 channels of `mask`.
        disp_pad = F.pad(disp, (1, 1, 1, 1))
        neighbors = [disp_pad[:, :, dy:dy + H, dx:dx + W]
                     for dy in range(3) for dx in range(3)]
        up_flow = torch.cat(neighbors, dim=1)        # (N, 9*D, H, W)
        up_flow = up_flow.reshape(N * D, 9, 1, H * W)

        up_flow = torch.sum(mask * up_flow, dim=1)   # (N*D, scale*scale, H*W)
        up_flow = up_flow.reshape(N, D, scale, scale, H, W)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # (N, D, H, scale, W, scale)
        up_flow = up_flow.reshape(N, D, scale * H, scale * W)

        return up_flow

    def forward(self, disp, mask, scale, up_mask_scale=8):
        disp2 = self.upsample_disp(disp, mask, up_mask_scale)

        if scale!=up_mask_scale:
            disp_up = scale * F.interpolate(
                        disp2,
                        size=(scale * disp2.shape[2] // up_mask_scale, scale * disp2.shape[3] // up_mask_scale),
                        mode="bilinear",
                        align_corners=False,
                        )
        else:
            disp_up = scale * disp2
        return disp_up


def compute_reverse_disp(fmap_l, fmap_r, featx_list_r, hourglass_mod, max_disp):
    """Compute right disparity using reverse corr + shared hourglass."""
    B, _, H, W = fmap_l.shape
    corr_block_in = AttentionCorrBlock(4, reverse=True)
    corr_block_in.calc_corr(fmap_l, fmap_r, corr_range=(-max_disp + 1, 2 + 1))
    corr_d8_in = corr_block_in.get_full_corr()

    corr_d8_0_0 = torch.cat(
        [corr_d8_in[i].reshape(B, H, W, max_disp + 2).permute(0, 3, 1, 2).unsqueeze(1)
         for i in range(4)],
        dim=1,
    )
    disp_cost_init, _ = hourglass_mod(corr_d8_0_0, featx_list_r)
    disp_r = disparity_regression(disp_cost_init, max_disp + 2) - 2
    return disp_r


def get_lrc_mask(disp_l, disp_r, thresh=0.5):
    b, _, h, w = disp_l.shape

    b_space = torch.arange(b, device=disp_l.device).reshape(b, 1, 1, 1)
    x_space = torch.arange(w, device=disp_l.device).reshape(1, 1, 1, w)
    y_space = torch.arange(h, device=disp_l.device).reshape(1, 1, h, 1).long()
    l2r = x_space - disp_l

    disp_r_reshape = disp_r.reshape(-1)

    l1r_int = torch.floor(l2r).long().clamp(0, w - 1)
    gather_idx = l1r_int + y_space * w + b_space * w * h
    gathered_idx = disp_r_reshape[gather_idx].reshape(b, 1, h, w)

    l1r_int = torch.ceil(l2r).long().clamp(0, w - 1)
    gather_idx1 = l1r_int + y_space * w + b_space * w * h
    gathered_idx1 = disp_r_reshape[gather_idx1].reshape(b, 1, h, w)

    lrc_mask = ((disp_l - gathered_idx).abs() < thresh).float()
    lrc_mask1 = ((disp_l - gathered_idx1).abs() < thresh).float()
    return 1.0 - (1.0 - lrc_mask) * (1.0 - lrc_mask1)
