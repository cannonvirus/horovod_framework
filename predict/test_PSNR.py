import argparse
import logging
import os
from os import listdir
import yaml
import sys
import re

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

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
import torch.nn as nn
import math
from model_resnet import *


def predict_img(net, background, crop_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    background_img = cv2.imread(background)
    crop_img = cv2.imread(crop_img)

    img_cc = BasicDataset.preprocess(img=background_img, scale=256)
    crp_cc = BasicDataset.preprocess(img=crop_img, scale=128)

    # img_stack = BasicDataset.two_img_preprocess(background, crop_img, scale_factor)

    img_cc = img_cc.unsqueeze(0)
    img_cc = img_cc.to(device=device, dtype=torch.float32)

    crp_cc = crp_cc.unsqueeze(0)
    crp_cc = crp_cc.to(device=device, dtype=torch.float32)

    prod_anormal = net(img_cc, crp_cc)

    return prod_anormal


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


def Mean_PSNR(config, model, device):

    criterion = nn.MSELoss().to(device)

    test_folders = sorted(listdir(config["input_path"]))
    sum_of_psnr = 0
    max_psnr = 0
    min_psnr = 99
    max_folder = None
    min_folder = None

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    SUM_TP_PSNR = 0
    SUM_TN_PSNR = 0
    SUM_FP_PSNR = 0
    SUM_FN_PSNR = 0

    for folder in test_folders:
        file_path_ = os.path.join(config["input_path"], folder)
        # model_input = [os.path.join(file_path_,i) for i in sorted(listdir(file_path_))][:-1]
        prod_anormal = predict_img(
            net=model,
            background=os.path.join(file_path_, "origin.jpg"),
            crop_img=os.path.join(file_path_, "crop.jpg"),
            scale_factor=config["scale"],
            out_threshold=config["threshold"],
            device=device,
        )

        # * anormal_prob
        anb_prob = prod_anormal.cpu().detach().numpy()[0][0]

        pred_answer = 1 if anb_prob > config["threshold"] else 0
        answer_hint = int(re.split("_", re.split("/", file_path_)[-1])[0])
        if answer_hint != 0:
            answer_hint = 1
        else:
            answer_hint = 0

        if answer_hint == 1 and pred_answer == answer_hint:
            # * 정상인데 정상으로 잡는 경우
            TP += 1
            # SUM_TP_PSNR += psnr
            # print("TP : {} | {}".format(file_path_, anb_prob))

        elif answer_hint == 0 and pred_answer == answer_hint:
            # * 비정상인데 비정상으로 잡는 경우
            TN += 1
            # SUM_TN_PSNR += psnr
            # print("TN : {} | {}".format(file_path_, anb_prob))

        elif answer_hint == 1 and pred_answer != answer_hint:
            # * 정상인데 비정상으로 잡는 경우
            FN += 1
            # SUM_FN_PSNR += psnr

            # print("FN : {} | {}".format(file_path_, anb_prob))

        else:
            # * 비정상인데 정상으로 잡은 경우
            FP += 1
            # SUM_FP_PSNR += psnr
            # print("FP : {} | {}".format(file_path_, anb_prob))

    avg_psnr = sum_of_psnr / len(test_folders)
    print("----------------------------------------------------")
    # print("Max PSNR : {:.2f} | folder : {}".format(max_psnr, max_folder))
    # print("Min PSNR : {:.2f} | folder : {}".format(min_psnr, min_folder))
    print("total set : {}".format(len(test_folders)))
    print("TP : {} | PSNR : {:.2f}".format(TP, SUM_TP_PSNR / (TP + 1)))
    print("TN : {} | PSNR : {:.2f}".format(TN, SUM_TN_PSNR / (TN + 1)))
    print("FP : {} | PSNR : {:.2f}".format(FP, SUM_FP_PSNR / (FP + 1)))
    print("FN : {} | PSNR : {:.2f}".format(FN, SUM_FN_PSNR / (FN + 1)))

    precision_st = TP / (TP + FP + 1)
    recall_st = TP / (TP + FN + 1)

    print("Precision : {:.2f}".format(precision_st))
    print("Recall : {:.2f}".format(recall_st))
    print(
        "F1-score : {:.2f}".format(
            2 * (precision_st * recall_st) / (precision_st + recall_st)
        )
    )
    print("Accuracy : {:.2f}%".format((TP + TN) / len(test_folders) * 100))

    # model_input = [os.path.join(file_path_,i) for i in sorted(listdir(file_path_))][:-1]
    # mask, prod_anormal_ = predict_img(net=model, full_img=model_input, scale_factor=96, out_threshold=config['threshold'], device=device)
    # min_img = mask_to_image(mask, config['bgr2rgb'])
    # cv2.imwrite("min_img.jpg", min_img)

    return avg_psnr


if __name__ == "__main__":
    config = get_yaml("/works/Anormal_Unet/predict/test_PSNR.yaml")
    # net = UNet_ENC_Double(n_channels=3, n_classes=3, half_model=False)
    # net = UNet_ENC(n_channels=6, n_classes=3, half_model=False)
    # net = UNet_AIGC_ver2(n_channels=3, n_classes=3, bilinear=True)
    net = ResNet(Bottleneck, [3, 4, 6, 3])

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:7")
    net.to(device=device)
    net.load_state_dict(torch.load(config["model_path"], map_location=device))

    M_psnr = Mean_PSNR(config, model=net, device=device)
    print("average PSNR : {}".format(M_psnr))
    print("----------------------------------------------------")
