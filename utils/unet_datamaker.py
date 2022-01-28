import sys
import os

script_path = os.path.dirname(__file__)
os.chdir(script_path)

import re
import cv2
import numpy as np
import math
import yaml
import shutil
import random
import os_module as om
import albumentations as A

def main(config):
    
    normal = om.extract_folder(config['normal_path'], ext="dir", full_path=True)
    if not config['only_normal']:
        anormal = om.extract_folder(config['anormal_path'], ext="dir", full_path=True)
    
    if not os.path.isdir(config['out_path']):
        os.mkdir(config['out_path'])
        
    if config['normal_sampling'] > 0:
        normal = random.sample(normal, config['normal_sampling'])
    elif config['normal_sampling'] == -1:
        #* 모든 data 사용
        pass
    else:
        print("Error : normal_sampling INT type")
        return 0
    
    if not config['only_normal']:
        if config['anormal_sampling'] > 0:
            normal = random.sample(normal, config['anormal_sampling'])
        elif config['anormal_sampling'] == -1:
            #* 모든 data 사용
            pass
        else:
            print("Error : anormal_sampling INT type")
            return 0
        
    #? NOTE : Data 저장 시 유의할점
    #? abnormal : 0_XXX/img.jpg ...
    #? normal : 1_XXX/img.jpg ...
    
    for idx, src in enumerate(normal):
        file_name = "{}_{}".format(1, str(idx).zfill(5))
        dst = os.path.join(config['out_path'], file_name)
        shutil.copytree( src, dst )
    
    if not config['only_normal']: # abnormal layer 학습 시 사용
        for idx, src in enumerate(anormal):
            file_name = "{}_{}".format(0, str(idx).zfill(5))
            dst = os.path.join(config['out_path'], file_name)
            shutil.copytree( src, dst )
     
    

if __name__ == '__main__':
    with open('unet_datamaker.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    main(config)

