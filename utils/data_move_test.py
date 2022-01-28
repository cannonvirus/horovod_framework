import os
import os_module as om
import random
import shutil
import re

out_path = "/works/Anormal_Unet/AIGC_test_data_o512_c64"

for path in om.extract_folder("/works/Anormal_Unet/AIGC_train_data_o512_c64", full_path=True):

    # if re.split("_", os.path.basename(path))[0] == "1":
    #     src = path
    #     dir_ = os.path.join(out_path, os.path.basename(path))
    #     shutil.move(src, dir_)

    prob = random.randint(0,1000)
    if prob <= 2:
        src = path
        dir_ = os.path.join(out_path, os.path.basename(path))
        shutil.move(src, dir_)