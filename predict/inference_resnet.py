import os
import sys
import yaml
import shutil
import re

sys.path.append('/works/Anormal_Unet/')

import numpy as np
import math
import torch
from glob import glob
from tqdm import tqdm
import pandas as pd

from utils.dataset import BasicDataset, CustomDataset_ens, leaf_csv_reader
import cv2
from model_resnet import *
from model_efficientnet import *


def get_yaml(yaml_path):
    with open(yaml_path) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict

def extract_folder(path, ext="dir", full_path=False):
    '''
    path : search path folder
    mode : "dir", ".jpg", ".png" ...
    full_path : os.path.join(*) or not
    '''

    if ext == "dir":
        result = [i for i in sorted(os.listdir(path)) if re.search("[.]", i) is None]
    elif ext == "all":
        result = sorted(os.listdir(path))
    else:
        result = [i for i in sorted(os.listdir(path)) if re.search(ext, i) is not None]

    if full_path:
        result = [os.path.join(path, i) for i in result]

    return result

def load_model(struct, model_path, device_=None):

    #! resnet50
    if struct == "resnet50":
        model = ResNet(Bottleneck, [3, 4, 6, 3])
        # model = ResNet2(Bottleneck, [3, 4, 6, 3])
    else:
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=111)

    if device_ is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_)
    model.to(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("Load Model Complete")
    return model, device

def inference_resnet(net, device, label_encoder, label_decoder, tmp_folder, frame_interval=15, scale=[256,128], threshold=0.8, width_cut=240):
    #! init
    net.eval()
    frame_count = 0
    time_prob_list = []

    if not os.path.isdir(tmp_folder):
        os.mkdir(tmp_folder)

    glob_test = sorted(glob("/works/lg_test/data/test/*"))
    test_dataset = CustomDataset_ens(glob_test,  label_encoder, mode="test")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, num_workers=8, shuffle=False
    )

    tqdm_dataset = tqdm(enumerate(test_dataloader))
    results = []
    for batch, batch_item in tqdm_dataset:
        img = batch_item["image"].to(device)
        with torch.no_grad():
            output = net(img)
        # output = (
        #     torch.tensor(torch.argmax(output, dim=1), dtype=torch.int32).cpu().numpy()
        # )
        output = torch.argmax(output, dim=1).clone().detach().cpu().numpy()
        
        results.extend(output)

    preds = np.array([label_decoder[int(val)] for val in results])

    submission = pd.read_csv("/works/lg_test/data/sample_submission.csv")
    submission["label"] = preds
    submission.to_csv("/works/Anormal_Unet/predict/lg_eff4_400_inference_code.csv", index=False)    


if __name__ == "__main__":

    config = get_yaml("./predict/inference_resnet.yaml")
    _, label_encoder, label_decoder = leaf_csv_reader(enc_dec_mode=0)

    net, device = load_model(struct = "effi", model_path = config['model_path'], device_="cuda")
    inference_resnet(net=net, device=device, label_encoder = label_encoder, label_decoder = label_decoder, tmp_folder=config['tmp_path'])


