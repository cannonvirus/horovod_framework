import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn as nn
import numpy as np
from utils.Focal_Loss import WeightedFocalLoss, FocalLoss

def eval_net(net, loader, device, batch_size, freeze_mode, config): 
    """Evaluation without the densecrf with the dice coefficient"""
    
    net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    if config['loss_func'] == "MSE":
        criterion = nn.MSELoss().to(device)
    elif config['loss_func'] == "BCE":
        criterion = nn.BCELoss().to(device)
    elif config['loss_func'] == "CROSS":
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = FocalLoss().to(device)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:

            #! Model Inference & Loss Calculation
            if config['input_img_number'] == 1:
                imgs, true_labels = batch['image'], batch['label']
            else:
                imgs, crops, true_labels = batch['image'], batch['crop'], batch['label']
                crops = batch['crop'].to(device=device, dtype=torch.float32)
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_labels = true_labels.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                if config['input_img_number'] == 1:
                    anormal_pred = net(imgs)
                else:
                    anormal_pred = net(imgs, crops)

                # loss = criterion(anormal_pred, true_labels.reshape(-1,1))
                loss = criterion(anormal_pred, true_labels.long())

            tot += loss
            pbar.update()

    net.train()
    return tot / n_val
