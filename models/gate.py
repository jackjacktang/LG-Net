import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    pad = int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)
    return pad

class LGNet(nn.Module):
    def __init__(self, in_ch, out_ch, batch_norm=True, cnum=32, use_gate=True):
        super(LGNet, self).__init__()
        print('LG-Net, cnum = ', cnum)
        self.enc1 = LesionGateConv(in_ch, cnum, 5, 1, padding=get_pad(256, 5, 1), batch_norm=batch_norm)
        # downsample 128
        self.enc2_1 = LesionGateConv(cnum, 2 * cnum, 4, 2, padding=get_pad(256, 4, 2), batch_norm=batch_norm)
        self.enc2_2 = LesionGateConv(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1), batch_norm=batch_norm)
        # downsample to 64
        self.enc3_1 = LesionGateConv(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2), batch_norm=batch_norm)
        self.enc3_2 = LesionGateConv(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), batch_norm=batch_norm)
        self.enc3_3 = LesionGateConv(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), batch_norm=batch_norm)
        self.enc4_1 = LesionGateConv(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), batch_norm=batch_norm)
        self.enc4_2 = LesionGateConv(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), batch_norm=batch_norm)
        self.enc4_3 = LesionGateConv(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), batch_norm=batch_norm)
        self.enc4_4 = LesionGateConv(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), batch_norm=batch_norm)
        self.enc4_5 = LesionGateConv(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), batch_norm=batch_norm)
        self.enc4_6 = LesionGateConv(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), batch_norm=batch_norm)

        # upsample
        self.dec1_1 = LesionGateDeConv(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1), batch_norm=batch_norm)
        self.dec1_2 = LesionGateConv(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1), batch_norm=batch_norm)

        self.dec2_1 = LesionGateDeConv(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1), batch_norm=batch_norm)
        self.dec2_2 = LesionGateConv(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1), batch_norm=batch_norm)
        self.final = LesionGateConv(cnum // 2, out_ch, 3, 1, padding=get_pad(128, 3, 1), activation=None, batch_norm=batch_norm)

    def forward(self, x, encoder_only=False, save_feat=False, lgc_layers=['enc4_6']):
        feat = []
        x = self.enc1(x)
        x = self.enc2_1(x)
        if 'enc2_1' in lgc_layers:
            feat.append(x)
        x = self.enc2_2(x)
        x = self.enc3_1(x)
        if 'enc3_1' in lgc_layers:
            feat.append(x)
        x = self.enc3_2(x)
        x = self.enc3_3(x)
        x = self.enc4_1(x)
        if 'enc4_2' in lgc_layers:
            feat.append(x)
        x = self.enc4_2(x)
        if 'enc4_3' in lgc_layers:
            feat.append(x)
        x = self.enc4_3(x)
        if 'enc4_4' in lgc_layers:
            feat.append(x)
        x = self.enc4_4(x)
        x = self.enc4_5(x)
        x = self.enc4_6(x)
        if 'enc4_6' in lgc_layers:
            feat.append(x)

        if encoder_only:
            return feat

        x = self.dec1_1(x)
        x = self.dec1_2(x)
        x = self.dec2_1(x)
        x = self.dec2_2(x)
        x = self.final(x)
        out = torch.clamp(x, 0., 1.)

        if save_feat:
            return out, feat
        else:
            return out


class LesionGateConv(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True), use_gate=True):
        super(LesionGateConv, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.use_gate = use_gate

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)

        if self.activation is not None:
            if self.use_gate:
                x = self.activation(x) * self.gated(mask)
            else:
                x = self.activation(x)
        else:
            if self.use_gate:
                x = x * self.gated(mask)

        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class LesionGateDeConv(torch.nn.Module):

    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True,activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(LesionGateDeConv, self).__init__()
        self.conv2d = LesionGateConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2)
        x = self.conv2d(x)
        return x
