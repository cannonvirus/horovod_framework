""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
from torchsummary import summary

from .unet_parts import *


class UNet(nn.Module):
    def __init__(
        self, n_channels, n_classes, bilinear=True, half_model=False, scale=96
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if half_model:
            self.inc = DoubleConv(n_channels, 32)
            self.down1 = Down(32, 64)
            self.down2 = Down(64, 128)
            self.down3 = Down(128, 256)
            factor = 2 if bilinear else 1
            self.down4 = Down(256, 512 // factor)

            if scale == 96:
                self.fc1 = torch.nn.Linear(256 * 6 * 6, 256, bias=True)  # scale : 96
            elif scale == 48:
                self.fc1 = torch.nn.Linear(256 * 3 * 3, 256, bias=True)  # scale : 48
            self.relu = nn.ReLU()
            self.fc2 = torch.nn.Linear(256, 1, bias=True)
            self.dropout = torch.nn.Dropout(0.2)
            self.sigmoid = torch.nn.Sigmoid()

            self.up1 = Up(512, 256 // factor, bilinear)
            self.up2 = Up(256, 128 // factor, bilinear)
            self.up3 = Up(128, 64 // factor, bilinear)
            self.up4 = Up(64, 32, bilinear)
            self.outc = OutConv(32, n_classes)

        else:
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)

            if scale == 96:
                self.fc1 = torch.nn.Linear(512 * 6 * 6, 512, bias=True)  # scale : 96
            elif scale == 48:
                self.fc1 = torch.nn.Linear(512 * 3 * 3, 512, bias=True)  # scale : 48
            self.relu = nn.ReLU()
            self.fc2 = torch.nn.Linear(512, 1, bias=True)
            self.dropout = torch.nn.Dropout(0.2)
            self.sigmoid = torch.nn.Sigmoid()

            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)
            self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        binary_x1 = self.fc1(x5.view(x5.size(0), -1))
        binary_x1_dr = self.dropout(binary_x1)
        binary_x1_rl = self.relu(binary_x1_dr)
        binary_x1_rl_dr = self.dropout(binary_x1_rl)
        binary_x2 = self.sigmoid(self.fc2(binary_x1_rl_dr))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, binary_x2


class UNet_ENC(nn.Module):
    def __init__(
        self, n_channels, n_classes, bilinear=True, half_model=False, scale=96
    ):
        super(UNet_ENC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if half_model:
            self.inc = DoubleConv(n_channels, 32)
            self.down1 = Down(32, 64)
            self.down2 = Down(64, 128)
            self.down3 = Down(128, 256)
            factor = 2 if bilinear else 1
            self.down4 = Down(256, 512 // factor)

            if scale == 96:
                self.fc1 = torch.nn.Linear(256 * 6 * 6, 256, bias=True)  # scale : 96
            elif scale == 48:
                self.fc1 = torch.nn.Linear(256 * 3 * 3, 256, bias=True)  # scale : 48
            self.relu = nn.ReLU()
            self.fc2 = torch.nn.Linear(256, 1, bias=True)
            self.dropout = torch.nn.Dropout(0.2)
            self.sigmoid = torch.nn.Sigmoid()

        else:
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)

            self.fc1 = torch.nn.Linear(512 * 15 * 15, 512, bias=True)  # scale : 240
            # self.fc1 = torch.nn.Linear(512*40*40, 512, bias=True) # scale : 640
            self.relu = nn.ReLU()
            self.fc2 = torch.nn.Linear(512, 1, bias=True)
            self.dropout = torch.nn.Dropout(0.2)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        binary_x1 = self.fc1(x5.view(x5.size(0), -1))
        binary_x1_dr = self.dropout(binary_x1)
        binary_x1_rl = self.relu(binary_x1_dr)
        binary_x1_rl_dr = self.dropout(binary_x1_rl)
        binary_x2 = self.sigmoid(self.fc2(binary_x1_rl_dr))

        return binary_x2


class UNet_ENC_Double(nn.Module):
    def __init__(
        self, n_channels, n_classes, bilinear=True, half_model=False, scale=96
    ):
        super(UNet_ENC_Double, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # origin layer
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.fc1 = torch.nn.Linear(512 * 15 * 15, 25, bias=True)  # scale : 240

        # crop layer
        self.inc_crop = DoubleConv(n_channels, 64)
        self.down1_crop = Down(64, 128)
        self.down2_crop = Down(128, 256)
        self.down3_crop = Down(256, 512)
        factor_crop = 2 if bilinear else 1
        self.down4_crop = Down(512, 1024 // factor_crop)
        self.fc1_crop = torch.nn.Linear(512 * 15 * 15, 4, bias=True)  # scale : 240

        self.dropout_step1 = torch.nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.dropout_step2 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(25 + 4, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        #! FIXME
        x_origin, x_crop = x[:, 0:3, :, :], x[:, 3:, :, :]

        x1_origin = self.inc(x_origin)
        x2_origin = self.down1(x1_origin)
        x3_origin = self.down2(x2_origin)
        x4_origin = self.down3(x3_origin)
        x5_origin = self.down4(x4_origin)
        binary_x1_origin = self.fc1(x5_origin.view(x5_origin.size(0), -1))

        x1_crop = self.inc_crop(x_crop)
        x2_crop = self.down1_crop(x1_crop)
        x3_crop = self.down2_crop(x2_crop)
        x4_crop = self.down3_crop(x3_crop)
        x5_crop = self.down4_crop(x4_crop)
        binary_x1_crop = self.fc1_crop(x5_crop.view(x5_crop.size(0), -1))

        #! FIXME cat, stack 사용
        binary_x1 = torch.cat([binary_x1_origin, binary_x1_crop], dim=1)

        # binary_x1_dr = self.dropout(binary_x1)
        binary_x1_rl = self.relu(binary_x1)
        binary_x1_rl_dr = self.dropout_step2(binary_x1_rl)
        result_ = self.sigmoid(self.fc2(binary_x1_rl_dr))

        return result_


class UNet_ENC_Double_Up(nn.Module):
    def __init__(
        self, n_channels, n_classes, bilinear=True, half_model=False, scale=96
    ):
        super(UNet_ENC_Double_Up, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # origin layer
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.fc1 = torch.nn.Linear(512 * 15 * 15, 512, bias=True)  # scale : 240

        # crop layer
        self.inc_crop = DoubleConv(n_channels, 64)
        self.down1_crop = Down(64, 128)
        self.down2_crop = Down(128, 256)
        self.down3_crop = Down(256, 512)
        factor_crop = 2 if bilinear else 1
        self.down4_crop = Down(512, 1024 // factor_crop)
        self.fc1_crop = torch.nn.Linear(512 * 15 * 15, 32, bias=True)  # scale : 240

        self.dropout_step1 = torch.nn.Dropout(0.2)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.dropout_step2 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(512 + 32, 16, bias=True)
        self.fc3 = torch.nn.Linear(16, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        #! FIXME
        x_origin, x_crop = x[:, 0:3, :, :], x[:, 3:, :, :]

        x1_origin = self.inc(x_origin)
        x2_origin = self.down1(x1_origin)
        x3_origin = self.down2(x2_origin)
        x4_origin = self.down3(x3_origin)
        x5_origin = self.down4(x4_origin)
        binary_x1_origin = self.fc1(x5_origin.view(x5_origin.size(0), -1))  # 512

        x1_crop = self.inc_crop(x_crop)
        x2_crop = self.down1_crop(x1_crop)
        x3_crop = self.down2_crop(x2_crop)
        x4_crop = self.down3_crop(x3_crop)
        x5_crop = self.down4_crop(x4_crop)
        binary_x1_crop = self.fc1_crop(x5_crop.view(x5_crop.size(0), -1))  # 32

        #! FIXME cat, stack 사용
        binary_x1 = torch.cat([binary_x1_origin, binary_x1_crop], dim=1)  # 512 + 32
        binary_x1_rl = self.relu_1(binary_x1)
        # binary_x1_dr = self.dropout(binary_x1)
        binary_x2 = self.fc2(binary_x1_rl)  # 16
        binary_x2_rl = self.relu_2(binary_x2)

        # binary_x1_rl_dr = self.dropout_step2(binary_x1_rl)
        result_ = self.sigmoid(self.fc3(binary_x2_rl))

        return result_


class UNet_ENTRY_ENS(nn.Module):
    def __init__(
        self, n_channels, n_classes, bilinear=True, half_model=False, scale=96
    ):
        super(UNet_ENTRY_ENS, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # origin layer
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.fc1 = torch.nn.Linear(512 * 15 * 15, 512, bias=True)  # scale : 240

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.dropout_step1 = torch.nn.Dropout(0.2)
        self.dropout_step2 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(512, 1, bias=True)
        self.fc3 = torch.nn.Linear(9, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.out = torch.nn.Linear(9, 1)

    def forward(self, x):

        x_origin = torch.cat([x[:, 0:3, :, :], x[:, 3:, :, :]], dim=1)
        x1_origin = self.inc(x_origin)
        x2_origin = self.down1(x1_origin)
        x3_origin = self.down2(x2_origin)
        x4_origin = self.down3(x3_origin)
        x5_origin = self.down4(x4_origin)
        binary_x1_origin = self.fc1(x5_origin.view(x5_origin.size(0), -1))  # 512
        binary_x1_dr = self.dropout_step1(binary_x1_origin)
        binary_x1_rl = self.relu_1(binary_x1_dr)
        # binary_x2 = self.fc2(binary_x1_rl)

        result_ = self.sigmoid(self.fc2(binary_x1_rl))

        return result_


class UNet_AIGC_ver2(nn.Module):
    def __init__(
        self, n_channels, n_classes, bilinear=True, half_model=False, scale=96
    ):
        super(UNet_AIGC_ver2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # origin layer
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # crop layer
        self.inc_c = DoubleConv(n_channels, 64)
        self.down1_c = Down(64, 128)
        self.down2_c = Down(128, 256)
        self.down3_c = Down(256, 512)
        factor_c = 2 if bilinear else 1
        self.down4_c = Down(512, 1024 // factor_c)

        self.fc1 = torch.nn.Linear(512 * 16 * 16, 512, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 1, bias=True)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):

        x1 = self.inc(x)
        y1 = self.inc_c(y)
        y1_re = y1.repeat(1, 1, 2, 2)
        xy1 = x1 + y1_re

        x2 = self.down1(xy1)
        y2 = self.down1_c(y1)
        y2_re = y2.repeat(1, 1, 2, 2)
        xy2 = x2 + y2_re

        x3 = self.down2(xy2)
        y3 = self.down2_c(y2)
        y3_re = y3.repeat(1, 1, 2, 2)
        xy3 = x3 + y3_re

        x4 = self.down3(xy3)
        y4 = self.down3_c(y3)
        y4_re = y4.repeat(1, 1, 2, 2)
        xy4 = x4 + y4_re

        x5 = self.down4(xy4)
        y5 = self.down4_c(y4)
        y5_re = y5.repeat(1, 1, 2, 2)
        xy5 = x5 + y5_re

        binary_x1 = self.fc1(xy5.view(xy5.size(0), -1))
        binary_x1_dr = self.dropout1(binary_x1)
        binary_x1_rl = self.relu(binary_x1_dr)
        binary_x1_rl_dr = self.dropout2(binary_x1_rl)
        binary_x2 = self.sigmoid(self.fc2(binary_x1_rl_dr))

        return binary_x2


if __name__ == "__main__":
    net = UNet_AIGC_ver2(
        n_channels=3, n_classes=3, bilinear=True, half_model=False, scale=240
    ).to("cuda")
    summary(net, [(3, 240, 240), (3, 120, 120)])
    # print(net)
