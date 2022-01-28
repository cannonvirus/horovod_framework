import argparse
import logging
import os
import sys
from os import listdir
import yaml

sys.path.append("/works/Anormal_Unet/")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model_unet import (
    UNet,
    UNet_ENC,
    UNet_ENC_Double,
    UNet_ENC_Double_Up,
    UNet_ENTRY_ENS,
    UNet_AIGC_ver2,
)
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import cv2


def predict_img(net, background, crop_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img_stack = BasicDataset.two_img_preprocess(background, crop_img, scale_factor)

    img_stack = img_stack.unsqueeze(0)
    img_stack = img_stack.to(device=device, dtype=torch.float32)

    ab_status = net(img_stack)

    print("Prob : {:.2f}".format(ab_status.cpu().detach().numpy()[0][0]))

    return ab_status


def mask_to_image(mask, bgr2rgb):
    image = (
        (mask.cpu().detach().numpy() * 255)
        .astype("uint8")
        .squeeze(0)
        .transpose(1, 2, 0)
    )
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_yaml(yaml_path):
    with open(yaml_path) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict


if __name__ == "__main__":
    config = get_yaml("./predict/test_one.yaml")
    net = UNet_ENC(n_channels=6, n_classes=3, half_model=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device=device)
    net.load_state_dict(torch.load(config["model_path"], map_location=device))

    mask = predict_img(
        net=net,
        background=config["background_path"],
        crop_img=config["crop_path"],
        scale_factor=config["scale"],
        out_threshold=config["threshold"],
        device=device,
    )
