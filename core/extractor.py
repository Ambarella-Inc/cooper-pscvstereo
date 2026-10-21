import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm


class Clamp8(nn.Module):
    def __init__(self, inplace=True):
        super(Clamp8,self).__init__()
        self.clamp = nn.Hardtanh(0.0, 7.9375, inplace=inplace)

    def forward(self, x):
        return self.clamp(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='batch', mid_down=1, stride=(1,1), act=nn.ReLU):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes//mid_down, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes//mid_down, planes, kernel_size=3, padding=1)
        self.relu1 = act(inplace=True)
        self.relu2 = act(inplace=True)
        self.relu3 = act(inplace=True)
        self.relu4 = act(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride[0] == 1 and stride[1] == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=in_planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//mid_down)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride[0] == 1 and stride[1] == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
                self.norm4 = nn.BatchNorm2d(in_planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//mid_down)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride[0] == 1 and stride[1] == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)
                self.norm4 = nn.InstanceNorm2d(in_planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride[0] == 1 and stride[1] == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()
                self.norm4 = nn.Sequential()

        if stride[0] == 1 and stride[1] == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride))

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu2(y)

        if self.downsample is not None:
            x = self.norm3(self.downsample(x))

        return self.relu4(x + y)


class BasicEncoderBase(nn.Module):
    def __init__(self, input_dim=3, output_dims=128, act=nn.ReLU):
        super(BasicEncoderBase, self).__init__()
        self.norm_fn = 'batch'
        self.norm1 = nn.BatchNorm2d(output_dims//3)

        self.act = act
        self.conv1 = nn.Conv2d(input_dim, output_dims//3, kernel_size=7, stride=2, padding=3)

        self.relu1 = self.act(inplace=True)

        self.layer1 = self._make_layer(output_dims//3, output_dims//3, stride=(1, 1))
        self.layer2 = self._make_layer(output_dims//3, output_dims//3 * 2, stride=(2, 2))
        self.layer3 = self._make_layer(output_dims//3 * 2, output_dims, stride=(2, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_dim, dim, stride=(1,1)):
        layer1 = ResidualBlock(in_dim, dim, self.norm_fn, stride=stride, act=self.act)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=(1,1), act=self.act)
        layers = (layer1, layer2)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x_d2 = self.layer1(x)
        x_d4 = self.layer2(x_d2)
        x_d8 = self.layer3(x_d4)
        return x_d8, x_d4, x_d2


class BasicEncoderTiny(nn.Module):
    def __init__(self, input_dim=3, output_dims=128, act=nn.ReLU):
        super(BasicEncoderTiny, self).__init__()
        self.norm_fn = 'batch'
        self.norm1 = nn.BatchNorm2d(output_dims//2)

        self.act = act
        self.conv1 = nn.Conv2d(input_dim, output_dims//2, kernel_size=7, stride=2, padding=3)
        self.relu1 = self.act(inplace=True)
        self.layer1_1 = self._make_layer(output_dims//2, output_dims//2, stride=(1,1))
        self.layer2_1 = self._make_layer(output_dims//2, output_dims//4 * 3, stride=(2,2))
        self.layer3_1 = self._make_layer(output_dims//4 * 3,output_dims, stride=(2,2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self,in_dim, dim, stride=(1,1)):
        layer1 = ResidualBlock(in_dim, dim, self.norm_fn, stride=stride,act=self.act)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=(1,1),act=self.act)
        layers = (layer1, layer2)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x_d2 = self.layer1_1(x)
        x_d4 = self.layer2_1(x_d2)
        x_d8 = self.layer3_1(x_d4)

        return x_d8,x_d4,x_d2


def fuse_conv_bn(conv, bn):
    with torch.no_grad():
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)

        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        b = (b - bn.running_mean) * scale + bn.bias

        fused_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=False,
            padding_mode=conv.padding_mode,
        )
        fused_conv.weight = nn.Parameter(
            conv.weight * scale.reshape(conv.out_channels, 1, 1, 1)
        )

        bn.running_mean.zero_()
        bn.running_var.fill_(1)
        bn.weight.data.fill_(1)
        bn.bias.data.copy_(b)
    return fused_conv, bn


class FeatureBase(nn.Module):
    def __init__(self, act = nn.ReLU6, act_out = nn.ReLU, input_dims=3):
        super(FeatureBase, self).__init__()
        pretrained = False
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        self.bias = False
        layers = [1, 2, 3, 5, 6]
        chans = [16, 24, 32, 96, 160]
        if input_dims==3:
            self.conv_stem = model.conv_stem
        else:
            self.conv_stem =  nn.Conv2d(input_dims, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=self.bias)
        self.bn1 = nn.BatchNorm2d(32)

        self.act = act
        self.act_out = act_out
        self.act1 = self.act(inplace=True)
        self.conv_stem, self.bn1 = fuse_conv_bn(self.conv_stem, self.bn1)

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.block0[0][0].conv_dw, self.block0[0][0].bn1 = fuse_conv_bn(self.block0[0][0].conv_dw, self.block0[0][0].bn1)
        self.block0[0][0].conv_pw, self.block0[0][0].bn2 = fuse_conv_bn(self.block0[0][0].conv_pw, self.block0[0][0].bn2)

        self.block1[0][0].conv_pw, self.block1[0][0].bn1 = fuse_conv_bn(self.block1[0][0].conv_pw, self.block1[0][0].bn1)
        self.block1[0][0].conv_dw, self.block1[0][0].bn2 = fuse_conv_bn(self.block1[0][0].conv_dw, self.block1[0][0].bn2)
        self.block1[0][0].conv_pwl, self.block1[0][0].bn3 = fuse_conv_bn(self.block1[0][0].conv_pwl, self.block1[0][0].bn3)

        self.block1[0][1].conv_pw, self.block1[0][1].bn1 = fuse_conv_bn(self.block1[0][1].conv_pw, self.block1[0][1].bn1)
        self.block1[0][1].conv_dw, self.block1[0][1].bn2 = fuse_conv_bn(self.block1[0][1].conv_dw, self.block1[0][1].bn2)
        self.block1[0][1].conv_pwl, self.block1[0][1].bn3 = fuse_conv_bn(self.block1[0][1].conv_pwl, self.block1[0][1].bn3)

        self.block2[0][0].conv_pw, self.block2[0][0].bn1 = fuse_conv_bn(self.block2[0][0].conv_pw, self.block2[0][0].bn1)
        self.block2[0][0].conv_dw, self.block2[0][0].bn2 = fuse_conv_bn(self.block2[0][0].conv_dw, self.block2[0][0].bn2)
        self.block2[0][0].conv_pwl, self.block2[0][0].bn3 = fuse_conv_bn(self.block2[0][0].conv_pwl, self.block2[0][0].bn3)

        self.block2[0][1].conv_pw, self.block2[0][1].bn1 = fuse_conv_bn(self.block2[0][1].conv_pw, self.block2[0][1].bn1)
        self.block2[0][1].conv_dw, self.block2[0][1].bn2 = fuse_conv_bn(self.block2[0][1].conv_dw, self.block2[0][1].bn2)
        self.block2[0][1].conv_pwl, self.block2[0][1].bn3 = fuse_conv_bn(self.block2[0][1].conv_pwl, self.block2[0][1].bn3)

        self.block2[0][2].conv_pw, self.block2[0][2].bn1 = fuse_conv_bn(self.block2[0][2].conv_pw, self.block2[0][2].bn1)
        self.block2[0][2].conv_dw, self.block2[0][2].bn2 = fuse_conv_bn(self.block2[0][2].conv_dw, self.block2[0][2].bn2)
        self.block2[0][2].conv_pwl, self.block2[0][2].bn3 = fuse_conv_bn(self.block2[0][2].conv_pwl, self.block2[0][2].bn3)

        self.block3[0][0].conv_pw, self.block3[0][0].bn1 = fuse_conv_bn(self.block3[0][0].conv_pw, self.block3[0][0].bn1)
        self.block3[0][0].conv_dw, self.block3[0][0].bn2 = fuse_conv_bn(self.block3[0][0].conv_dw, self.block3[0][0].bn2)
        self.block3[0][0].conv_pwl, self.block3[0][0].bn3 = fuse_conv_bn(self.block3[0][0].conv_pwl, self.block3[0][0].bn3)

        self.block3[0][1].conv_pw, self.block3[0][1].bn1 = fuse_conv_bn(self.block3[0][1].conv_pw, self.block3[0][1].bn1)
        self.block3[0][1].conv_dw, self.block3[0][1].bn2 = fuse_conv_bn(self.block3[0][1].conv_dw, self.block3[0][1].bn2)
        self.block3[0][1].conv_pwl, self.block3[0][1].bn3 = fuse_conv_bn(self.block3[0][1].conv_pwl, self.block3[0][1].bn3)

        self.block3[0][2].conv_pw, self.block3[0][2].bn1 = fuse_conv_bn(self.block3[0][2].conv_pw, self.block3[0][2].bn1)
        self.block3[0][2].conv_dw, self.block3[0][2].bn2 = fuse_conv_bn(self.block3[0][2].conv_dw, self.block3[0][2].bn2)
        self.block3[0][2].conv_pwl, self.block3[0][2].bn3 = fuse_conv_bn(self.block3[0][2].conv_pwl, self.block3[0][2].bn3)

        self.block3[0][3].conv_pw, self.block3[0][3].bn1 = fuse_conv_bn(self.block3[0][3].conv_pw, self.block3[0][3].bn1)
        self.block3[0][3].conv_dw, self.block3[0][3].bn2 = fuse_conv_bn(self.block3[0][3].conv_dw, self.block3[0][3].bn2)
        self.block3[0][3].conv_pwl, self.block3[0][3].bn3 = fuse_conv_bn(self.block3[0][3].conv_pwl, self.block3[0][3].bn3)


        self.block3[1][0].conv_pw, self.block3[1][0].bn1 = fuse_conv_bn(self.block3[1][0].conv_pw, self.block3[1][0].bn1)
        self.block3[1][0].conv_dw, self.block3[1][0].bn2 = fuse_conv_bn(self.block3[1][0].conv_dw, self.block3[1][0].bn2)
        self.block3[1][0].conv_pwl, self.block3[1][0].bn3 = fuse_conv_bn(self.block3[1][0].conv_pwl, self.block3[1][0].bn3)

        self.block3[1][1].conv_pw, self.block3[1][1].bn1 = fuse_conv_bn(self.block3[1][1].conv_pw, self.block3[1][1].bn1)
        self.block3[1][1].conv_dw, self.block3[1][1].bn2 = fuse_conv_bn(self.block3[1][1].conv_dw, self.block3[1][1].bn2)
        self.block3[1][1].conv_pwl, self.block3[1][1].bn3 = fuse_conv_bn(self.block3[1][1].conv_pwl, self.block3[1][1].bn3)

        self.block3[1][2].conv_pw, self.block3[1][2].bn1 = fuse_conv_bn(self.block3[1][2].conv_pw, self.block3[1][2].bn1)
        self.block3[1][2].conv_dw, self.block3[1][2].bn2 = fuse_conv_bn(self.block3[1][2].conv_dw, self.block3[1][2].bn2)
        self.block3[1][2].conv_pwl, self.block3[1][2].bn3 = fuse_conv_bn(self.block3[1][2].conv_pwl, self.block3[1][2].bn3)

        self.block4[0][0].conv_pw, self.block4[0][0].bn1 = fuse_conv_bn(self.block4[0][0].conv_pw, self.block4[0][0].bn1)
        self.block4[0][0].conv_dw, self.block4[0][0].bn2 = fuse_conv_bn(self.block4[0][0].conv_dw, self.block4[0][0].bn2)
        self.block4[0][0].conv_pwl, self.block4[0][0].bn3 = fuse_conv_bn(self.block4[0][0].conv_pwl, self.block4[0][0].bn3)

        self.block4[0][1].conv_pw, self.block4[0][1].bn1 = fuse_conv_bn(self.block4[0][1].conv_pw, self.block4[0][1].bn1)
        self.block4[0][1].conv_dw, self.block4[0][1].bn2 = fuse_conv_bn(self.block4[0][1].conv_dw, self.block4[0][1].bn2)
        self.block4[0][1].conv_pwl, self.block4[0][1].bn3 = fuse_conv_bn(self.block4[0][1].conv_pwl, self.block4[0][1].bn3)

        self.block4[0][2].conv_pw, self.block4[0][2].bn1 = fuse_conv_bn(self.block4[0][2].conv_pw, self.block4[0][2].bn1)
        self.block4[0][2].conv_dw, self.block4[0][2].bn2 = fuse_conv_bn(self.block4[0][2].conv_dw, self.block4[0][2].bn2)
        self.block4[0][2].conv_pwl, self.block4[0][2].bn3 = fuse_conv_bn(self.block4[0][2].conv_pwl, self.block4[0][2].bn3)

        self.conv32_16_dw = nn.Sequential(
            nn.ConvTranspose2d(chans[4], chans[4], kernel_size=4, stride=2,padding=1,groups = chans[4],bias=False),
            nn.BatchNorm2d(chans[4]),
            self.act(inplace=True),
            nn.Conv2d(chans[4], 96, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(96,affine=True)
        )

        self.conv16_16_dw = nn.Sequential(
            nn.Conv2d(chans[3]+96, chans[3]+96, kernel_size=3, stride=1,padding=1,groups = chans[3]+96,bias=False),
            nn.BatchNorm2d(chans[3]+96),
            self.act(inplace=True),
            nn.Conv2d(chans[3]+96, 96, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(96,affine=True),
        )

        self.conv16_8_dw = nn.Sequential(
            nn.ConvTranspose2d(96, 96, kernel_size=4, stride=2,padding=1,bias=False,groups = 96),
            nn.BatchNorm2d(96),
            self.act(inplace=True),
            nn.Conv2d(96, 32, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(32,affine=True)
        )

        self.conv4_8_dw_ = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=1, stride=1,padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2,padding=1,bias=False,groups = 32),
            nn.BatchNorm2d(32),
            self.act(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(32,affine=True),
        )

        self.conv8_8_dw = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1,padding=1,bias=False,groups = 96),
            nn.BatchNorm2d(96),
            self.act(inplace=True),
            nn.Conv2d(96, 96, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(96),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(96,affine=True),
            self.act_out(inplace=True)
        )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        x1 = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x1)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x32_16 = self.conv32_16_dw(x32)
        x16_16 = self.conv16_16_dw(torch.cat([x32_16, x16], 1))
        x16_8 = self.conv16_8_dw(x16_16)
        x4_8 = self.conv4_8_dw_(x4)
        x8_8 = self.conv8_8_dw(torch.cat([x4_8, x8, x16_8], 1))
        return self.conv_out(x8_8), x8_8, x16_16, x32


class FeatureTiny(nn.Module):
    def __init__(self, act = nn.ReLU6, act_out = nn.ReLU, input_dims=3):
        super(FeatureTiny, self).__init__()
        pretrained = False
        model = timm.create_model('mobilenetv2_050', pretrained=pretrained, features_only=True)
        self.bias = False
        layers = [1, 2, 3, 5, 6]
        chans = [8, 16, 16, 48, 80]
        if input_dims==3:
            self.conv_stem = model.conv_stem
        else:
            self.conv_stem =  nn.Conv2d(input_dims, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=self.bias)
        self.bn1 = nn.BatchNorm2d(16)

        self.act = act
        self.act_out = act_out
        self.act1 = self.act(inplace=True)
        self.conv_stem, self.bn1 = fuse_conv_bn(self.conv_stem, self.bn1)

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])

        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])

        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.block0[0][0].conv_dw, self.block0[0][0].bn1 = fuse_conv_bn(self.block0[0][0].conv_dw, self.block0[0][0].bn1)
        self.block0[0][0].conv_pw, self.block0[0][0].bn2 = fuse_conv_bn(self.block0[0][0].conv_pw, self.block0[0][0].bn2)

        self.block1[0][0].conv_pw, self.block1[0][0].bn1 = fuse_conv_bn(self.block1[0][0].conv_pw, self.block1[0][0].bn1)
        self.block1[0][0].conv_dw, self.block1[0][0].bn2 = fuse_conv_bn(self.block1[0][0].conv_dw, self.block1[0][0].bn2)
        self.block1[0][0].conv_pwl, self.block1[0][0].bn3 = fuse_conv_bn(self.block1[0][0].conv_pwl, self.block1[0][0].bn3)

        self.block1[0][1].conv_pw, self.block1[0][1].bn1 = fuse_conv_bn(self.block1[0][1].conv_pw, self.block1[0][1].bn1)
        self.block1[0][1].conv_dw, self.block1[0][1].bn2 = fuse_conv_bn(self.block1[0][1].conv_dw, self.block1[0][1].bn2)
        self.block1[0][1].conv_pwl, self.block1[0][1].bn3 = fuse_conv_bn(self.block1[0][1].conv_pwl, self.block1[0][1].bn3)

        self.block2[0][0].conv_pw, self.block2[0][0].bn1 = fuse_conv_bn(self.block2[0][0].conv_pw, self.block2[0][0].bn1)
        self.block2[0][0].conv_dw, self.block2[0][0].bn2 = fuse_conv_bn(self.block2[0][0].conv_dw, self.block2[0][0].bn2)
        self.block2[0][0].conv_pwl, self.block2[0][0].bn3 = fuse_conv_bn(self.block2[0][0].conv_pwl, self.block2[0][0].bn3)

        self.block2[0][1].conv_pw, self.block2[0][1].bn1 = fuse_conv_bn(self.block2[0][1].conv_pw, self.block2[0][1].bn1)
        self.block2[0][1].conv_dw, self.block2[0][1].bn2 = fuse_conv_bn(self.block2[0][1].conv_dw, self.block2[0][1].bn2)
        self.block2[0][1].conv_pwl, self.block2[0][1].bn3 = fuse_conv_bn(self.block2[0][1].conv_pwl, self.block2[0][1].bn3)

        self.block2[0][2].conv_pw, self.block2[0][2].bn1 = fuse_conv_bn(self.block2[0][2].conv_pw, self.block2[0][2].bn1)
        self.block2[0][2].conv_dw, self.block2[0][2].bn2 = fuse_conv_bn(self.block2[0][2].conv_dw, self.block2[0][2].bn2)
        self.block2[0][2].conv_pwl, self.block2[0][2].bn3 = fuse_conv_bn(self.block2[0][2].conv_pwl, self.block2[0][2].bn3)

        self.block3[0][0].conv_pw, self.block3[0][0].bn1 = fuse_conv_bn(self.block3[0][0].conv_pw, self.block3[0][0].bn1)
        self.block3[0][0].conv_dw, self.block3[0][0].bn2 = fuse_conv_bn(self.block3[0][0].conv_dw, self.block3[0][0].bn2)
        self.block3[0][0].conv_pwl, self.block3[0][0].bn3 = fuse_conv_bn(self.block3[0][0].conv_pwl, self.block3[0][0].bn3)

        self.block3[0][1].conv_pw, self.block3[0][1].bn1 = fuse_conv_bn(self.block3[0][1].conv_pw, self.block3[0][1].bn1)
        self.block3[0][1].conv_dw, self.block3[0][1].bn2 = fuse_conv_bn(self.block3[0][1].conv_dw, self.block3[0][1].bn2)
        self.block3[0][1].conv_pwl, self.block3[0][1].bn3 = fuse_conv_bn(self.block3[0][1].conv_pwl, self.block3[0][1].bn3)

        self.block3[0][2].conv_pw, self.block3[0][2].bn1 = fuse_conv_bn(self.block3[0][2].conv_pw, self.block3[0][2].bn1)
        self.block3[0][2].conv_dw, self.block3[0][2].bn2 = fuse_conv_bn(self.block3[0][2].conv_dw, self.block3[0][2].bn2)
        self.block3[0][2].conv_pwl, self.block3[0][2].bn3 = fuse_conv_bn(self.block3[0][2].conv_pwl, self.block3[0][2].bn3)

        self.block3[0][3].conv_pw, self.block3[0][3].bn1 = fuse_conv_bn(self.block3[0][3].conv_pw, self.block3[0][3].bn1)
        self.block3[0][3].conv_dw, self.block3[0][3].bn2 = fuse_conv_bn(self.block3[0][3].conv_dw, self.block3[0][3].bn2)
        self.block3[0][3].conv_pwl, self.block3[0][3].bn3 = fuse_conv_bn(self.block3[0][3].conv_pwl, self.block3[0][3].bn3)

        self.block3[1][0].conv_pw, self.block3[1][0].bn1 = fuse_conv_bn(self.block3[1][0].conv_pw, self.block3[1][0].bn1)
        self.block3[1][0].conv_dw, self.block3[1][0].bn2 = fuse_conv_bn(self.block3[1][0].conv_dw, self.block3[1][0].bn2)
        self.block3[1][0].conv_pwl, self.block3[1][0].bn3 = fuse_conv_bn(self.block3[1][0].conv_pwl, self.block3[1][0].bn3)

        self.block3[1][1].conv_pw, self.block3[1][1].bn1 = fuse_conv_bn(self.block3[1][1].conv_pw, self.block3[1][1].bn1)
        self.block3[1][1].conv_dw, self.block3[1][1].bn2 = fuse_conv_bn(self.block3[1][1].conv_dw, self.block3[1][1].bn2)
        self.block3[1][1].conv_pwl, self.block3[1][1].bn3 = fuse_conv_bn(self.block3[1][1].conv_pwl, self.block3[1][1].bn3)

        self.block3[1][2].conv_pw, self.block3[1][2].bn1 = fuse_conv_bn(self.block3[1][2].conv_pw, self.block3[1][2].bn1)
        self.block3[1][2].conv_dw, self.block3[1][2].bn2 = fuse_conv_bn(self.block3[1][2].conv_dw, self.block3[1][2].bn2)
        self.block3[1][2].conv_pwl, self.block3[1][2].bn3 = fuse_conv_bn(self.block3[1][2].conv_pwl, self.block3[1][2].bn3)

        self.block4[0][0].conv_pw, self.block4[0][0].bn1 = fuse_conv_bn(self.block4[0][0].conv_pw, self.block4[0][0].bn1)
        self.block4[0][0].conv_dw, self.block4[0][0].bn2 = fuse_conv_bn(self.block4[0][0].conv_dw, self.block4[0][0].bn2)
        self.block4[0][0].conv_pwl, self.block4[0][0].bn3 = fuse_conv_bn(self.block4[0][0].conv_pwl, self.block4[0][0].bn3)

        self.block4[0][1].conv_pw, self.block4[0][1].bn1 = fuse_conv_bn(self.block4[0][1].conv_pw, self.block4[0][1].bn1)
        self.block4[0][1].conv_dw, self.block4[0][1].bn2 = fuse_conv_bn(self.block4[0][1].conv_dw, self.block4[0][1].bn2)
        self.block4[0][1].conv_pwl, self.block4[0][1].bn3 = fuse_conv_bn(self.block4[0][1].conv_pwl, self.block4[0][1].bn3)

        self.block4[0][2].conv_pw, self.block4[0][2].bn1 = fuse_conv_bn(self.block4[0][2].conv_pw, self.block4[0][2].bn1)
        self.block4[0][2].conv_dw, self.block4[0][2].bn2 = fuse_conv_bn(self.block4[0][2].conv_dw, self.block4[0][2].bn2)
        self.block4[0][2].conv_pwl, self.block4[0][2].bn3 = fuse_conv_bn(self.block4[0][2].conv_pwl, self.block4[0][2].bn3)

        self.conv32_16_dw = nn.Sequential(
            nn.ConvTranspose2d(chans[4], chans[4], kernel_size=4, stride=2,padding=1,groups = chans[4],bias=False),
            nn.BatchNorm2d(chans[4]),
            self.act(inplace=True),
            nn.Conv2d(chans[4], 48, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(48,affine=True)
        )

        self.conv16_16_dw = nn.Sequential(
            nn.Conv2d(chans[3]+48, chans[3]+48, kernel_size=3, stride=1,padding=1,groups = chans[3]+48,bias=False),
            nn.BatchNorm2d(chans[3]+48),
            self.act(inplace=True),
            nn.Conv2d(chans[3]+48, 48, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(48,affine=True),
        )

        self.conv16_8_dw = nn.Sequential(
            nn.ConvTranspose2d(48, 48, kernel_size=4, stride=2,padding=1,bias=False,groups = 48),
            nn.BatchNorm2d(48),
            self.act(inplace=True),
            nn.Conv2d(48, 24, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(24,affine=True)
        )

        self.conv4_8_dw_ = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=1, stride=1,padding=0, bias=False),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=3, stride=2,padding=1,bias=False,groups = 24),
            nn.BatchNorm2d(24),
            self.act(inplace=True),
            nn.Conv2d(24, 24, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(24,affine=True),
        )

        self.conv8_8_dw = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1,bias=False,groups = 64),
            nn.BatchNorm2d(64),
            self.act(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(64),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64,affine=True),
            self.act_out(inplace=True)
        )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        x1 = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x1)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x32_16 = self.conv32_16_dw(x32)
        x16_16 = self.conv16_16_dw(torch.cat([x32_16, x16], 1))
        x16_8 = self.conv16_8_dw(x16_16)

        x4_8 = self.conv4_8_dw_(x4)
        x8_8 = self.conv8_8_dw(torch.cat([x4_8, x8, x16_8], 1))

        return self.conv_out(x8_8), x8_8, x16_16, x32
