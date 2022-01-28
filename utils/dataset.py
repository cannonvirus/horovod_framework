from os.path import splitext
from os import listdir
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import re
import cv2
import json

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, scale=512, sub_scale=64, time_series=4, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.scale = scale
        self.scale_sub = sub_scale
        self.time_series = time_series
        self.mask_suffix = mask_suffix
        
        self.ids = []
        for kd in listdir(imgs_dir):
            if len(listdir(os.path.join(imgs_dir, kd))) > self.time_series:
                self.ids.append(os.path.join(self.imgs_dir, kd))    
                
            # if re.search(".png", kd) is not None or re.search(".jpg", kd) is not None:
            #     self.ids.append(os.path.join(self.imgs_dir, kd))
                
        self.ids = sorted(self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, scale, bgr2rgb = True, float32 = True):
        # FIXME : 가변변수로 받을 수 있도록
        h,w,c = img.shape
        # if w > scale or h > scale:
        #     img = cv2.resize(img, dsize=(scale, scale), interpolation=cv2.INTER_AREA)

        # if len(img.shape) == 2:
        #     img_nd = np.expand_dims(img, axis=2)
            
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img = torch.from_numpy(img.transpose(2, 0, 1))
        
        if float32:
            img = img.float()

        # HWC to CHW
        if img.max() > 1:
            img = img / 255

        return img
    
    @classmethod
    def time_series_preprocess(cls, img_list, scale, bgr2rgb = True, float32 = True):
        # FIXME : 가변변수로 받을 수 있도록
        img_stack = None
        
        for img_path in img_list:
            img = cv2.imread(img_path)
            h,w,c = img.shape
    
            if w != scale or h != scale:
                img = cv2.resize(img, dsize=(scale, scale), interpolation=cv2.INTER_AREA)

            if len(img.shape) == 2:
                img_nd = np.expand_dims(img, axis=2)
                
            if img.shape[2] == 3 and bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            if img_stack is None:
                img_stack = img.transpose(2, 0, 1)
            else:
                img = img.transpose(2, 0, 1)
                img_stack = np.vstack([img_stack, img])
            
        img = torch.from_numpy(img_stack)
        
        if float32:
            img = img.float()

        # HWC to CHW
        if img.max() > 1:
            img_final = img / 255

        return img_final

    @classmethod
    def two_img_preprocess(cls, background, crop_img, scale, bgr2rgb = True, float32 = True):
        # FIXME : 가변변수로 받을 수 있도록
        img_stack = None
        
        for img_path in [background, crop_img]:
            img = cv2.imread(img_path)
            h,w,c = img.shape
    
            if w != scale or h != scale:
                img = cv2.resize(img, dsize=(scale, scale), interpolation=cv2.INTER_AREA)

            if len(img.shape) == 2:
                img_nd = np.expand_dims(img, axis=2)
                
            if img.shape[2] == 3 and bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            if img_stack is None:
                img_stack = img.transpose(2, 0, 1)
            else:
                img = img.transpose(2, 0, 1)
                img_stack = np.vstack([img_stack, img])
            
        img = torch.from_numpy(img_stack)
        
        if float32:
            img = img.float()

        # HWC to CHW
        if img.max() > 1:
            img_final = img / 255

        return img_final

    @classmethod
    def two_img_preprocess_ENS(cls, background, crop_img, scale, bgr2rgb = True, float32 = True):
        # FIXME : 가변변수로 받을 수 있도록
        img_stack = None

        Edge_point_hw = [(0,0),(0,240),(0,480),(120,120),(120,240),(120,360),(240,0),(240,240),(240,480)]

        
        for img_path in [background, crop_img]:
            img = cv2.imread(img_path)
            h,w,c = img.shape
    
            if w != scale or h != scale: # ORIGIN
                # img = cv2.resize(img, dsize=(scale, scale), interpolation=cv2.INTER_AREA)
                for (yp, xp) in Edge_point_hw:

                    img_point = img[yp:yp+scale, xp:xp+scale, :]
                        
                    if img.shape[2] == 3 and bgr2rgb:
                        img_point = cv2.cvtColor(img_point, cv2.COLOR_BGR2RGB)
                        
                    if img_stack is None:
                        img_stack = img_point.transpose(2, 0, 1)
                    else:
                        img_point = img_point.transpose(2, 0, 1)
                        img_stack = np.vstack([img_stack, img_point])

            else:
                if len(img.shape) == 2:
                    img_nd = np.expand_dims(img, axis=2)
                    
                if img.shape[2] == 3 and bgr2rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                if img_stack is None:
                    img_stack = img.transpose(2, 0, 1)
                else:
                    img = img.transpose(2, 0, 1)
                    img_stack = np.vstack([img_stack, img])

            
        img = torch.from_numpy(img_stack)
        
        if float32:
            img = img.float()

        if img.max() > 1:
            img_final = img / 255

        return img_final

    @classmethod
    def AIGC_process(cls, background, crop_img, scale, bgr2rgb = True, float32 = True):
        # FIXME : 가변변수로 받을 수 있도록
        img_stack = None

        #? background 
        h,w,c = background.shape

        if w != scale or h != scale:
            background = cv2.resize(background, dsize=(scale, scale), interpolation=cv2.INTER_AREA)

        if background.shape[2] == 3 and bgr2rgb:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        img_stack = background.transpose(2, 0, 1)

        #? crop_img
        h,w,c = crop_img.shape

        if w != scale or h != scale:
            crop_img = cv2.resize(crop_img, dsize=(scale, scale), interpolation=cv2.INTER_AREA)

        if crop_img.shape[2] == 3 and bgr2rgb:
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        crop_img = crop_img.transpose(2, 0, 1)
        img_stack = np.vstack([img_stack, crop_img])
            
        img = torch.from_numpy(img_stack)
        
        if float32:
            img = img.float()

        # HWC to CHW
        if img.max() > 1:
            img_final = img / 255

        return img_final

    def __getitem__(self, i):
        
        pig_series_path = self.ids[i]
        
        label = 0
        
        if int(re.split("_",re.split("/", pig_series_path)[-1])[0]) == 0:
            label = 0
        else:
            label = 1
        
        
        background_path = os.path.join(pig_series_path, "origin.jpg")
        crop_img_path = os.path.join(pig_series_path, "crop.jpg")

        # img_cc = self.two_img_preprocess(background_path, crop_img_path, self.scale)
        # img_cc = self.two_img_preprocess_ENS(background_path, crop_img_path, self.scale)

        background_img = cv2.imread(background_path)
        crop_img = cv2.imread(crop_img_path)

        img_cc = self.preprocess(img=background_img, scale=self.scale)
        crp_cc = self.preprocess(img=crop_img, scale=self.scale_sub)

        return {
            'image': img_cc,
            'crop' : crp_cc,
            'label' : label
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, scale=1):
        super().__init__(imgs_dir, scale, mask_suffix='_mask')


class CustomDataset_ens(Dataset):
    def __init__(
        self, files, label_encoder, labels=None, mode="train"
    ):
        self.mode = mode
        self.files = files
        self.max_len = 24 * 6
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split("/")[-1]

        # image
        image_path = f"{file}/{file_name}.jpg"
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255
        img = np.transpose(img, (2, 0, 1))

        if self.mode == "train":
            json_path = f"{file}/{file_name}.json"
            with open(json_path, "r") as f:
                json_file = json.load(f)

            crop = json_file["annotations"]["crop"]
            disease = json_file["annotations"]["disease"]
            risk = json_file["annotations"]["risk"]
            label = f"{crop}_{disease}_{risk}"

            return {
                "image": torch.tensor(img, dtype=torch.float32),
                "label": torch.tensor(
                    self.label_encoder[label], dtype=torch.long
                ),
            }
            # return {
            #     "img": torch.tensor(img, dtype=torch.float32),
            #     "label_crop": torch.tensor(
            #         self.label_encoder[0][f"{crop}"], dtype=torch.long
            #     ),
            #     "label_disease": torch.tensor(
            #         self.label_encoder[1][f"{disease}"], dtype=torch.long
            #     ),
            #     "label_risk": torch.tensor(
            #         self.label_encoder[2][f"{risk}"], dtype=torch.long
            #     ),
            # }
        else:
            return {
                "image": torch.tensor(img, dtype=torch.float32),
            }

def leaf_csv_reader(enc_dec_mode=0):
    csv_features = [
        "내부 온도 1 평균",
        "내부 온도 1 최고",
        "내부 온도 1 최저",
        "내부 습도 1 평균",
        "내부 습도 1 최고",
        "내부 습도 1 최저",
        "내부 이슬점 평균",
        "내부 이슬점 최고",
        "내부 이슬점 최저",
    ]
    csv_files = sorted(glob("data/train/*/*.csv"))

    # temp_csv = pd.read_csv(csv_files[0])[csv_features]
    # max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    # # feature 별 최대값, 최솟값 계산
    # for csv in tqdm(csv_files[1:]):
    #     temp_csv = pd.read_csv(csv)[csv_features]
    #     temp_csv = temp_csv.replace("-", np.nan).dropna()
    #     if len(temp_csv) == 0:
    #         continue
    #     temp_csv = temp_csv.astype(float)
    #     temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
    #     max_arr = np.max([max_arr, temp_max], axis=0)
    #     min_arr = np.min([min_arr, temp_min], axis=0)

    # feature 별 최대값, 최솟값 dictionary 생성
    min_arr = np.array([3.4, 3.4, 3.3, 23.7, 25.9, 0, 0.1, 0.2, 0.0])
    max_arr = np.array([46.8, 47.1, 46.6, 100, 100, 100, 34.5, 34.7, 34.4])

    csv_feature_dict = {
        csv_features[i]: [min_arr[i], max_arr[i]] for i in range(len(csv_features))
    }
    # 변수 설명 csv 파일 참조
    crop = {"1": "딸기", "2": "토마토", "3": "파프리카", "4": "오이", "5": "고추", "6": "시설포도"}
    disease = {
        "1": {
            "a1": "딸기잿빛곰팡이병",
            "a2": "딸기흰가루병",
            "b1": "냉해피해",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "2": {
            "a5": "토마토흰가루병",
            "a6": "토마토잿빛곰팡이병",
            "b2": "열과",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "3": {
            "a9": "파프리카흰가루병",
            "a10": "파프리카잘록병",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "4": {
            "a3": "오이노균병",
            "a4": "오이흰가루병",
            "b1": "냉해피해",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "5": {
            "a7": "고추탄저병",
            "a8": "고추흰가루병",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "6": {"a11": "시설포도탄저병", "a12": "시설포도노균병", "b4": "일소피해", "b5": "축과병"},
    }
    risk = {"1": "초기", "2": "중기", "3": "말기"}

    label_description = {}  # classification 111 number ex) '딸기_다량원소결핍 (P)_말기'

    label_description_crop = {}
    label_description_disease = {}
    label_description_risk = {}
    for key, value in disease.items():
        label_description[f"{key}_00_0"] = f"{crop[key]}_정상"
        for disease_code in value:
            for risk_code in risk:
                label = f"{key}_{disease_code}_{risk_code}"
                label_crop = f"{key}"
                label_disease = f"{disease_code}"
                label_risk = f"{risk_code}"

                label_description[
                    label
                ] = f"{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}"
                label_description_crop[label_crop] = f"{crop[key]}"
                label_description_disease[
                    label_disease
                ] = f"{disease[key][disease_code]}"
                label_description_risk[label_risk] = f"{risk[risk_code]}"

    label_description_disease["00"] = "정상"
    label_description_risk["0"] = "정상"

    # ex) '1_00_0' : 0
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_encoder_crop = {key: idx for idx, key in enumerate(label_description_crop)}
    label_encoder_disease = {
        key: idx for idx, key in enumerate(label_description_disease)
    }
    label_encoder_risk = {key: idx for idx, key in enumerate(label_description_risk)}

    # ex) '0' : '1_00_0'
    label_decoder = {val: key for key, val in label_encoder.items()}
    label_decoder_crop = {val: key for key, val in label_encoder_crop.items()}
    label_decoder_disease = {val: key for key, val in label_encoder_disease.items()}
    label_decoder_risk = {val: key for key, val in label_encoder_risk.items()}

    # print(label_decoder)
    if enc_dec_mode == 0:
        return csv_feature_dict, label_encoder, label_decoder
    else:
        return (
            csv_feature_dict,
            [label_encoder_crop, label_encoder_disease, label_encoder_risk],
            [label_decoder_crop, label_decoder_disease, label_decoder_risk],
        )