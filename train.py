import argparse
import logging
import os
import yaml
import cv2
import sys
import re
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from model_unet import (
    UNet,
    UNet_ENC,
    UNet_ENC_Double,
    UNet_ENC_Double_Up,
    UNet_ENTRY_ENS,
    UNet_AIGC_ver2,
)
from model_resnet import *
from model_efficientnet import *
from model_esemble import *


from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, CustomDataset_ens, leaf_csv_reader
from torch.utils.data import DataLoader, random_split
from utils.Focal_Loss import WeightedFocalLoss, FocalLoss

import torch.nn.functional as F
import horovod.torch as hvd


def get_yaml():
    with open("config.yaml") as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return config_dict


def train_net(
    net,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 0.001,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
):

    # 1. Create dataset
    args = get_yaml()
    save_cp = args["save_checkpoints"]

    #! DataLoader ------------------------------------------------------------------------------------
    _, label_encoder, _ = leaf_csv_reader(enc_dec_mode=0)
    glob_train = sorted(glob("/works/lg_test/data/train/*"))
    glob_test = sorted(glob("/works/lg_test/data/test/*"))
    labelsss = pd.read_csv("/works/lg_test/data/train.csv")["label"]
    train_sp, val_sp = train_test_split(glob_train, test_size=0.2, stratify=labelsss)

    train_dataset = CustomDataset_ens(train_sp, label_encoder)
    val_dataset = CustomDataset_ens(val_sp, label_encoder)
    test_dataset = CustomDataset_ens(glob_test, label_encoder, mode="test")

    # dataset = BasicDataset(args['input_img_path'],
    #                        args['scale'],
    #                        args['scale_sub'],
    #                        args['input_time_series'])

    # n_val = int(len(dataset) * args['validation_ratio'])
    # n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])

    if args["horovod"]:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )

        train_loader_args = dict(
            batch_size=args["batch_size"],
            num_workers=args["num_worker"],
            pin_memory=False,
            sampler=train_sampler,
        )

        train_loader = DataLoader(train_dataset, **train_loader_args)
    else:
        train_loader_args = dict(
            batch_size=args["batch_size"],
            num_workers=args["num_worker"],
            pin_memory=False,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **train_loader_args)

    val_loader_args = dict(
        batch_size=args["batch_size"], num_workers=args["num_worker"], pin_memory=False
    )
    val_loader = DataLoader(val_dataset, drop_last=True, **val_loader_args)
    #! ------------------------------------------------------------------------------------------------------------

    writer = SummaryWriter(
        comment=f'LR_{args["learning_rate"]}_BS_{batch_size}_SCALE_{args["scale"]}'
    )
    logging.info(
        f"""Starting training:
        Epochs:          {args["epoch"]}
        Batch size:      {batch_size}
        Learning rate:   {args["learning_rate"]}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Amp:             {args['amp']}
        Images scaling:  {args["scale"]}
    """
    )

    #! Set up the optimizer, the learning rate scheduler
    optimizer = optim.Adam(
        net.parameters(),
        lr=args["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-8,
        amsgrad=False,
    )
    if args["horovod"]:
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=net.named_parameters()
        )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=10)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    #! Select Loss function
    if args["loss_func"] == "MSE":
        criterion = nn.MSELoss().to(device)
    elif args["loss_func"] == "BCE":
        criterion = nn.BCELoss().to(device)
    elif args["loss_func"] == "CROSS":
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = FocalLoss(alpha=0.25).to(device)

    #! horovod parameter ~ target gpu
    if args["horovod"]:
        hvd.broadcast_parameters(net.state_dict(), root_rank=0)

    # 5. Begin training
    global_step = 0

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(
            total=len(train_dataset), desc=f"Epoch {epoch}/{epochs}", unit="img"
        ) as pbar:
            for batch in train_loader:
                imgs = batch["image"].to(device=device, dtype=torch.float32)
                true_labels = batch["label"].to(device=device, dtype=torch.float32)

                #! Model Inference & Loss Calculation
                if args["input_img_number"] == 1:
                    # one input
                    anormal_pred = net(imgs)
                    loss = criterion(anormal_pred, true_labels.long())
                    # loss = criterion(anormal_pred, true_labels.reshape(-1,1))
                else:
                    # double input
                    crops = batch["crop"].to(device=device, dtype=torch.float32)
                    anormal_pred, feature = net(imgs, crops)
                    loss = criterion(anormal_pred, true_labels.reshape(-1, 1))

                #! Loss Backward & Optimizer step
                if args["horovod"]:
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)

                if args["amp"] == True:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                #! ???
                pbar.update(imgs.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                #! Evaluation -------------------------------------------------------------------------
                if (
                    global_step % int(len(train_dataset) * args["validation_ratio"])
                    == 0
                    and hvd.rank() == 0
                ):
                    val_score = eval_net(
                        net,
                        val_loader,
                        device,
                        batch_size,
                        freeze_mode=args["freezing_mode"],
                        config=args,
                    )
                    scheduler.step(val_score)
                    logging.info("Validation Dice Coeff: {:.4f}".format(val_score))
                #! ------------------------------------------------------------------------------------

        if hvd.rank() == 0:
            writer.add_scalar("Loss hvd = 0", epoch_loss / (len(train_loader)), epoch)
        elif hvd.rank() == 1:
            writer.add_scalar("Loss hvd = 1", epoch_loss / (len(train_loader)), epoch)
        else:
            writer.add_scalar("Loss hvd else", epoch_loss / (len(train_loader)), epoch)

        #! Save Model -------------------------------------------------------------------------
        if args["horovod"]:
            if epoch % args["save_model_epoch"] == 0 and hvd.rank() == 0:
                try:
                    os.mkdir(args["checkpoint_path"])
                    logging.info("Created checkpoint directory")
                except OSError:
                    pass
                torch.save(
                    net.state_dict(),
                    args["checkpoint_path"] + f"lgtest_esemble_{epoch}.pth",
                )
                logging.info(f"Checkpoint {epoch} saved !")
        else:
            if epoch % args["save_model_epoch"] == 0:
                try:
                    os.mkdir(args["checkpoint_path"])
                    logging.info("Created checkpoint directory")
                except OSError:
                    pass
                torch.save(
                    net.state_dict(), args["checkpoint_path"] + f"Cow_epoch{epoch}.pth"
                )
                logging.info(f"Checkpoint {epoch} saved !")
        #! ----------------------------------------------------------------------------------

    writer.close()


if __name__ == "__main__":
    args = get_yaml()

    # * horovod 초기화
    if args["horovod"]:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # net = UNet_ENC(n_channels=6, n_classes=3, bilinear=True, scale = args['scale']
    #! Resnet -----------------------------------------------------------------------------
    # net = ResNet(Bottleneck, [3, 4, 6, 3])
    # if args['resnet_pretrain_on']:
    #     pretrain_dict = torch.load("/works/Anormal_Unet/resnet/resnet50_pretrain.pth", map_location=device)
    #     pretrain_dict.pop("fc.weight")
    #     pretrain_dict.pop("fc.bias")
    #     network_ = deepcopy(pretrain_dict)

    #     for k,v in pretrain_dict.items():
    #         new_k = ""
    #         for id_, tx in enumerate(re.split("[.]", k)):
    #             if id_ == 0:
    #                 new_k = tx + "_crp"
    #             else:
    #                 new_k += ".{}".format(tx)
    #         network_[new_k] = v

    #     net.load_state_dict(network_, strict=False)
    # net.load_state_dict(pretrain_dict, strict=False)
    #! Resnet -----------------------------------------------------------------------------

    # ? efficientnet ------------------
    # net = EfficientNet.from_pretrained('efficientnet-b7', num_classes=111)
    # ? efficientnet ------------------

    # ? ensemble model ----------------
    e_model_1 = EfficientNet.from_pretrained("efficientnet-b3", num_classes=1024)
    e_model_2 = EfficientNet.from_pretrained("efficientnet-b4", num_classes=1024)
    e_model_3 = EfficientNet.from_pretrained("efficientnet-b2", num_classes=1024)
    net = EnsembleModel(e_model_1, e_model_2, e_model_3, 111, device)

    # ? ensemble model ----------------

    if args["load_pretrain"]:
        net.load_state_dict(torch.load(args["load_pretrain"], map_location=device))
        logging.info(f'Model loaded from {args["load_pretrain"]}')

    net.to(device=device)

    #! Freezing --------------------------------------------------------------------------------------------------------
    # if args["freezing_mode"]:
    #     logging.info(f'Mode : Freezing mode ~!!')
    #     for idx, child in enumerate(net.children()):
    #         if child._get_name() in ("Conv2d", 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'AdaptiveAvgPool2d'):
    #             for param in child.parameters():
    #                 param.requires_grad = False
    #! Freezing --------------------------------------------------------------------------------------------------------

    try:
        train_net(
            net=net,
            epochs=args["epoch"],
            batch_size=args["batch_size"],
            learning_rate=args["learning_rate"],
            device=device,
            img_scale=args["learning_rate"],
            val_percent=args["validation_ratio"],
            amp=args["amp"],
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        sys.exit(0)
