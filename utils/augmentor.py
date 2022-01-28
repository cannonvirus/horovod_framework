import albumentations as A
import cv2
import numpy as np
import random
import os_module as om
import yaml
import os
import shutil


def logic_select(max_algorithm = 3, option_algorithm = False):

    #* Default
    #* 1 :  A.HorizontalFlip(p=1.0)
    #* 2 :  A.VerticalFlip(p=1.0)
    #* 3 :  A.Rotate(limit=[x,x], border_mode=0, p=1.0)      |x| < 90
    #* 4 :  A.Downscale(scale_min=x, scale_max=x, p=1.0)     0.6 < x < 1
    #* 5 :  A.Blur(blur_limit=[x,x], p=1.0)                  x : 3~7
    #* 6 :  A.RGBShift(r_shift_limit = [x,x], g_shift_limit = [y,y], b_shift_limit = [z,z], p=1.0) |x,y,z| < 20
    #* 7 :  A.CLAHE( clip_limit=[1.5,1.5], p=1.0)            x : 1~4
    #* 8 :  A.RandomBrightnessContrast(brightness_limit=[0.1,0.1], contrast_limit=[0.1,0.1], p=1.0)    |x,y| < 0.2
    #* 9 :  A.RandomGamma(gamma_limit=(x,x), p=1.0)    x : 80 ~ 120

    #? Option : 가급적 고정값으로 사용
    #? 10 : A.GaussNoise(var_limit=[x,x], p=1.0)  10~20
    #? 11 : A.RandomRain(slant_lower=10, slant_upper=10, blur_value=2, brightness_coefficient=1.0, p=1.0) 
    #? 12 : A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.1, brightness_coeff=1.4, p=1.0)
    #? 13 : A.Sharpen(alpha=0.2, lightness=1.0, p=1.0) # 보류

    if option_algorithm:
        pickup_list = [ i for i in range(1,13+1) ]
        weight_ = (40,40,50,10,10,10,10,10,10,10,10,10,10)
        
    else:
        pickup_list = [ i for i in range(1,9+1) ]
        weight_ = (30,30,40,10,10,10,10,10,10)

    k_weight = [i for i in range(1,max_algorithm+1)]
    pick_algorithm_number = random.choices(pickup_list, weights=weight_, k=random.choices(k_weight, weights=k_weight, k=1)[0])

    answer_list = []

    for num_ in pick_algorithm_number:
        
        if num_ == 1:
            answer_list.append(A.HorizontalFlip(p=1.0))
        elif num_ == 2:
            answer_list.append(A.VerticalFlip(p=1.0))
        elif num_ == 3:
            x = random.randrange(-90, 91)
            answer_list.append(A.Rotate(limit=[x,x], border_mode=0, p=1.0))
        elif num_ == 4:
            x = random.uniform(0.6, 1.0)
            answer_list.append(A.Downscale(scale_min=x, scale_max=x, p=1.0))
        elif num_ == 5:
            x = random.randrange(3,7+1)
            answer_list.append(A.Blur(blur_limit=[x,x], p=1.0))
        elif num_ == 6:
            x = random.uniform(-20, 20)
            y = random.uniform(-20, 20)
            z = random.uniform(-20, 20)
            answer_list.append(A.RGBShift(r_shift_limit = [x,x], g_shift_limit = [y,y], b_shift_limit = [z,z], p=1.0))
        elif num_ == 7:
            x = random.uniform(1,4)
            answer_list.append(A.CLAHE( clip_limit=[x,x], p=1.0 ))
        elif num_ == 8:
            x = random.uniform(-0.2, 0.2)
            y = random.uniform(-0.2, 0.2)
            answer_list.append(A.RandomBrightnessContrast(brightness_limit=[x,x], contrast_limit=[y,y], p=1.0))
        elif num_ == 9:
            x = random.randrange(80,120)
            answer_list.append(A.RandomGamma(gamma_limit=(x,x), p=1.0))
        elif num_ == 10:
            x = random.uniform(10, 20)
            answer_list.append(A.GaussNoise(var_limit=[x,x], p=1.0))
        elif num_ == 11:
            answer_list.append(A.RandomRain(slant_lower=10, slant_upper=10, blur_value=2, brightness_coefficient=1.0, p=1.0) )
        elif num_ == 12:
            answer_list.append(A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.1, brightness_coeff=1.4, p=1.0))
        elif num_ == 13:
            answer_list.append(A.Sharpen(alpha=0.2, lightness=1.0, p=1.0))

    transform =  A.Compose(answer_list)

    return transform


def data_generator_AUG(config):

    if not os.path.isdir(config['output_path']):
        os.mkdir(config['output_path'])

    img_folder_path_list = om.extract_folder(config['unet_data_path'], full_path=True)
    for idx, folder_ in enumerate(img_folder_path_list):
        img_path_list = om.extract_folder(folder_, ext=".jpg", full_path=True)

        if config['original_copy']:
            if os.path.isdir( os.path.join(config['output_path'], "{}_origin".format(str(idx).zfill(4))) ):
                shutil.rmtree(os.path.join(config['output_path'], "{}_origin".format(str(idx).zfill(4))))
            shutil.copytree(folder_, os.path.join(config['output_path'], "{}_origin".format(str(idx).zfill(4))))

        for jdx in range(config['max_copy_img']):

            # ANCHOR
            transform = logic_select(max_algorithm = config['max_algorithm'], option_algorithm = config['option_algorithm'])

            if os.path.isdir(os.path.join(config['output_path'], "{}_{}".format(str(idx).zfill(4), str(jdx).zfill(3)))):
                shutil.rmtree(os.path.join(config['output_path'], "{}_{}".format(str(idx).zfill(4), str(jdx).zfill(3))))
            os.mkdir(os.path.join(config['output_path'], "{}_{}".format(str(idx).zfill(4), str(jdx).zfill(3))))
            
            for img_path_ in img_path_list:
                image = cv2.imread(img_path_)
                image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                transformed = transform(image=image_RGB)
                transformed_image = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)

                output_path = os.path.join(config['output_path'], "{}_{}".format(str(idx).zfill(4), str(jdx).zfill(3)), os.path.basename(img_path_))
                cv2.imwrite(output_path, transformed_image)
                
        print("{} | {}".format(idx, len(img_folder_path_list)))   

if __name__ == "__main__":
    with open('/works/Anormal_Unet/utils/augmentor.yaml') as file:
        config_ = yaml.load(file, Loader=yaml.FullLoader)
    data_generator_AUG(config = config_)