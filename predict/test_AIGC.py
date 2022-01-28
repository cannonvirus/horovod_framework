import os
import sys
import yaml
import shutil
import re

sys.path.append('/works/Anormal_Unet/')

import numpy as np
import math
import torch

from utils.dataset import BasicDataset
import cv2
from model_resnet import *


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
        # model = ResNet(Bottleneck, [3, 4, 6, 3])
        model = ResNet2(Bottleneck, [3, 4, 6, 3])
    else:
        return None

    if device_ is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_)
    model.to(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("Load Model Complete")
    return model, device

def when_snapshot(net, device, video_cap, snapshot, tmp_folder, frame_interval=15, scale=[256,128], threshold=0.8, width_cut=240):
    #! init
    net.eval()
    frame_count = 0
    time_prob_list = []

    if not os.path.isdir(tmp_folder):
        os.mkdir(tmp_folder)

    while(1):
        ret, frame = video_cap.read() 

        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            print("frame : {}".format(frame_count))

            clean_img = frame[:, width_cut:frame.shape[1]-width_cut, :]

            clean_img = cv2.resize(clean_img, dsize=(scale[0], scale[0]), interpolation=cv2.INTER_AREA)
            snapshot = cv2.resize(snapshot, dsize=(scale[1], scale[1]), interpolation=cv2.INTER_AREA)

            img_cc = BasicDataset.preprocess(img=clean_img, scale=scale[0])
            crp_cc = BasicDataset.preprocess(img=snapshot, scale=scale[1])

            img_cc = img_cc.unsqueeze(0)
            img_cc = img_cc.to(device=device, dtype=torch.float32)

            crp_cc = crp_cc.unsqueeze(0)
            crp_cc = crp_cc.to(device=device, dtype=torch.float32)
            
            ab_status = net(img_cc, crp_cc)
            prob = ab_status.cpu().detach().numpy()[0][0]
            
            if prob >= threshold:
                time_prob_list.append([frame_count, prob])
                print("frame : {} | Prob : {:.2f}".format(frame_count, prob))

        cv2.imwrite( os.path.join(tmp_folder, "{}.jpg".format(str(frame_count).zfill(5))), frame )
        frame_count += 1
    
    print("Make Timetable Complete")
    return time_prob_list

def ready_video_and_image(video_path, snapshot_path):
    cap = cv2.VideoCapture(video_path)
    snapshot = cv2.imread(snapshot_path)
    return cap, snapshot

def point_time(time_table, time_threshold=100):
    #! example
    if time_table is None:
        time_table = [[0,0.9],[15,0.8],[30,0.9],[60,0.92],[75,0.94],
                        [300,0.92],[315,0.84],[330,0.98],[360,0.92],[475,0.94],
                        [700,0.91],[715,0.82],[730,0.93],[745,0.94],[900,0.95]]

    time_table = np.array(time_table)

    first_point = time_table[0][0]
    result_box = []
    point_box = []
    for idx, (t, p) in enumerate(time_table):
        if idx == 0:
            point_box.append([t,p])
            continue

        if idx == len(time_table)-1:
            point_box = np.array(point_box)
            time_ = int(np.median(point_box.T[0]))
            score_ = np.average(point_box.T[1])
            final_score = round(math.log(len(point_box.T[0])) * score_, 2)
            result_box.append([final_score, time_])
            break

        if t - first_point <= time_threshold:
            point_box.append([t,p])
            first_point = t
        else:
            point_box = np.array(point_box)
            time_ = int(np.median(point_box.T[0]))
            score_ = np.average(point_box.T[1])
            final_score = round(math.log(len(point_box.T[0])) * score_, 2)
            result_box.append([final_score, time_])
            point_box = []
            first_point = t

    print("Point Time Complete")
    print( sorted(result_box, reverse=True) )

    return sorted(result_box, reverse=True)[0][1]

def choice_time(time_table, tmp_folder, time_between=200, time_ = None ):

    #! time choice
    if len(time_table) > 1:
        time_ = point_time(time_table)
    elif len(time_table) == 1:
        time_ = time_table[0][0]
    else:
        shutil.rmtree(tmp_folder)
        os.mkdir(tmp_folder)
        return None

    #! 사진 제거
    list_tmp = extract_folder(tmp_folder, ext=".jpg", full_path=True)
    for path in list_tmp:
        frame = int(re.split("[.]",os.path.basename(path))[0])
        if frame >= time_ - time_between and frame <= time_ + time_between:
            pass
        else:
            if os.path.exists(path):
                os.remove(path)

    return time_


if __name__ == "__main__":

    config = get_yaml("./predict/test_AIGC.yaml")

    net, device = load_model(struct = "resnet50", model_path = config['model_path'], device_="cuda")
    video_cap, snapshot = ready_video_and_image(config['video_path'], config['snap_shot_path'])
    time_table = when_snapshot(net=net, device=device, video_cap=video_cap, snapshot=snapshot, tmp_folder=config['tmp_path'])
    this_is_time = choice_time(time_table, tmp_folder=config['tmp_path'])

    print(this_is_time)

